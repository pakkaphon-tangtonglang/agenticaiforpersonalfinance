"""Search Thai mutual funds by Thai/English name, fund house or code (SEC data).

Order of attempts:
1. Fund house in the query ("กรุงศรี ปันผล", "kbank") → that house's funds,
   filtered by the remaining words.
2. Otherwise a partial name/code search ("ปันผล", "K-USA").
3. Nothing found → the LLM suggests name terms ("หุ้นอเมริกา" → "US", ...),
   each searched in SEC data. Results always come from the SEC.

Each result is one unit class with its latest NAV; classes without a recent
NAV (closed or inactive) are left out.
"""

import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from datetime import date
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from finance_ai.tools import sec_fund_api
from finance_ai.tools.fund_nav_client import SINGLE_CLASS_NAME, latest_navs_for_project

MAX_FUNDS = 6  # fund projects whose NAV is looked up per search
MAX_RESULTS = 12
MIN_QUERY_LENGTH = 2
REGISTERED = "Registered"
PROFILE_TTL_SECONDS = 86400.0  # names and classes change rarely
_AMC_NAME_NOISE = re.compile(
    r"บริษัทหลักทรัพย์จัดการกองทุน(?:รวม)?|บริษัท|จำกัด|\(มหาชน\)|\(ประเทศไทย\)|"
    r"ยกเลิกการใช้งานเนื่องจากควบรวมกิจการ|จัดการกองทุน"
)
# Everyday names people use for fund houses that differ from the official name.
_AMC_ALIASES: dict[str, str] = {
    "kbank": "กสิกรไทย",
    "kasset": "กสิกรไทย",
    "กสิกร": "กสิกรไทย",
    "scb": "ไทยพาณิชย์",
    "scbam": "ไทยพาณิชย์",
    "ktam": "กรุงไทย",
    "bbl": "บัวหลวง",
    "bblam": "บัวหลวง",
    "one": "วรรณ",
    "kkp": "เกียรตินาคินภัทร",
}
_PAYOUT_SUFFIX = re.compile(r"\((A|D)\)$")
# English first words that are ordinary words, not a fund house ("thai stock").
_GENERIC_ENGLISH_WORDS = frozenset({"thai", "asset", "land", "first", "bangkok", "siam", "krung"})
_MIN_ENGLISH_NAME = 4
# A leading "กองทุน(รวม)" / "fund" is how people ask, not part of fund names.
_FUND_WORD = re.compile(r"^(?:กองทุนรวม|กองทุน|mutual fund|fund)\s*", re.IGNORECASE)


class FundAmc(BaseModel):
    """A fund house with the short Thai name people use ("กรุงศรี")."""

    unique_id: str
    short_name_th: str
    name_en: str


class FundSearchResult(BaseModel):
    """One fund unit class with its latest NAV.

    Example:
        >>> FundSearchResult(symbol="K-VALUE-A(A)", proj_id="M0023_2538",
        ...     fund_code="K-VALUE", name_th="กองทุนเปิดเค หุ้นปันผล", name_en="",
        ...     amc="กสิกรไทย", payout="A", nav=Decimal("7.36"),
        ...     nav_date=date(2026, 10, 30)).payout
        'A'
    """

    symbol: str
    proj_id: str
    fund_code: str
    name_th: str
    name_en: str
    amc: str
    payout: str  # "A" accumulate, "D" dividend, "" unknown
    nav: Decimal
    nav_date: date


class FundSearchOutcome(BaseModel):
    """Search results plus the extra terms the LLM suggested (for transparency)."""

    results: list[FundSearchResult]
    expanded_terms: list[str] = []


def search_funds(query: str, expand: Callable[[str], list[str]] | None = None) -> FundSearchOutcome:
    """Find Thai fund classes for a query, using the LLM only when nothing matches.

    Args:
        query: Thai/English name, fund house, or code ("กรุงศรี ปันผล", "K-USA").
        expand: Query → extra search terms (LLM); None turns expansion off.

    Returns:
        Up to MAX_RESULTS classes with NAV, and any expansion terms used.

    Example:
        >>> search_funds("ปันผล").results[0].fund_code  # doctest: +SKIP
        'K-VALUE'
    """
    cleaned = _FUND_WORD.sub("", " ".join(query.split())).strip() or " ".join(query.split())
    if len(cleaned) < MIN_QUERY_LENGTH or not sec_fund_api.is_configured():
        return FundSearchOutcome(results=[])
    rows = _profile_rows(cleaned)
    terms: list[str] = []
    if not rows and expand is not None:
        terms = expand(cleaned)
        rows = _expanded_rows(terms)
    return FundSearchOutcome(results=_with_navs(_rank(rows, cleaned)), expanded_terms=terms)


def find_amc(query: str) -> tuple[FundAmc | None, str]:
    """Split a fund house out of the query: ("กรุงศรี ปันผล") → (กรุงศรี, "ปันผล").

    Example:
        >>> amc, rest = find_amc("kbank หุ้นปันผล")  # doctest: +SKIP
        >>> (amc.short_name_th, rest)
        ('กสิกรไทย', 'หุ้นปันผล')
    """
    folded = query.casefold()
    for alias, short_name in _AMC_ALIASES.items():
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", folded):
            amc = _amc_by_short_name(short_name)
            if amc:
                return amc, _remove(query, alias)
    for amc in sorted(list_amcs(), key=lambda a: -len(a.short_name_th)):
        if len(amc.short_name_th) >= MIN_QUERY_LENGTH and amc.short_name_th.casefold() in folded:
            return amc, _remove(query, amc.short_name_th)
        english = _distinctive_english_name(amc)
        if english and re.search(rf"\b{re.escape(english)}\b", folded):
            return amc, _remove(query, english)
    return None, query


def _distinctive_english_name(amc: FundAmc) -> str:
    """First word of the English name when it identifies the house ("krungsri"), else ""."""
    first = amc.name_en.split()[0].casefold() if amc.name_en else ""
    return "" if len(first) < _MIN_ENGLISH_NAME or first in _GENERIC_ENGLISH_WORDS else first


def list_amcs() -> list[FundAmc]:
    """Active fund houses with short Thai names (cached for a day)."""
    rows = sec_fund_api.fetch_items(sec_fund_api.AMCS_PATH, {}, ttl_seconds=PROFILE_TTL_SECONDS)
    amcs = [_to_amc(row) for row in rows if "ยกเลิก" not in str(row.get("comp_name_th", ""))]
    return [amc for amc in amcs if amc.short_name_th]


def _to_amc(row: dict[str, Any]) -> FundAmc:
    """Official AMC row → FundAmc with the short name ("กรุงศรี")."""
    short = " ".join(_AMC_NAME_NOISE.sub(" ", str(row.get("comp_name_th", ""))).split())
    return FundAmc(
        unique_id=str(row.get("unique_id", "")),
        short_name_th=short,
        name_en=str(row.get("comp_name_en", "")).strip(),
    )


def _amc_by_short_name(short_name: str) -> FundAmc | None:
    """The AMC whose short name is exactly this."""
    return next((amc for amc in list_amcs() if amc.short_name_th == short_name), None)


def _remove(text: str, word: str) -> str:
    """Drop one word (case-insensitive) and tidy spaces."""
    return " ".join(re.sub(re.escape(word), " ", text, count=1, flags=re.IGNORECASE).split())


def _profile_rows(query: str) -> list[dict[str, Any]]:
    """Fund-house search when the query names one, else a name/code search."""
    amc, rest = find_amc(query)
    if amc is None:
        return _name_rows(query)
    rows = sec_fund_api.fetch_items(
        sec_fund_api.PROFILES_PATH,
        {"company_info": amc.unique_id, "fund_status": REGISTERED},
        max_pages=5,
        ttl_seconds=PROFILE_TTL_SECONDS,
    )
    words = rest.casefold().split()
    return [row for row in rows if all(word in _searchable(row) for word in words)]


def _name_rows(term: str) -> list[dict[str, Any]]:
    """Registered fund classes whose Thai/English name or code contains the term."""
    return sec_fund_api.fetch_items(
        sec_fund_api.PROFILES_PATH,
        {"project_info": term, "fund_status": REGISTERED},
        max_pages=3,
        ttl_seconds=PROFILE_TTL_SECONDS,
    )


def _expanded_rows(terms: list[str]) -> list[dict[str, Any]]:
    """Rows for LLM-suggested terms, searched in parallel and taken in turns.

    Suggested English terms must match a whole word ("US" must not match "PLUS"),
    and results alternate between terms so one term cannot fill the list.
    """
    if not terms:
        return []
    with ThreadPoolExecutor(max_workers=len(terms)) as pool:
        per_term = list(pool.map(_name_rows, terms))
    matched = [
        [row for row in rows if _matches_whole_word(row, term)]
        for term, rows in zip(terms, per_term)
    ]
    return [row for group in zip_longest(*matched) for row in group if row is not None]


def _matches_whole_word(row: dict[str, Any], term: str) -> bool:
    """Thai terms match anywhere; ASCII terms only as whole words or code parts."""
    if not term.isascii():
        return term.casefold() in _searchable(row)
    # Letters may not touch the term ("PLUS" ≠ "US"); digits may ("US500" has "US").
    pattern = rf"(?<![a-z]){re.escape(term.casefold())}(?![a-z])"
    return re.search(pattern, _searchable(row)) is not None


def _searchable(row: dict[str, Any]) -> str:
    """Name fields of a profile row, folded for matching."""
    fields = ("proj_name_th", "proj_name_en", "proj_abbr_name", "fund_class_name")
    return " ".join(str(row.get(field) or "") for field in fields).casefold()


def _rank(rows: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Exact fund-code matches first, keeping the API order otherwise."""
    wanted = query.strip().upper()
    return sorted(rows, key=lambda row: str(row.get("proj_abbr_name", "")).upper() != wanted)


def _with_navs(rows: list[dict[str, Any]]) -> list[FundSearchResult]:
    """Attach the latest NAV per class for up to MAX_FUNDS funds, mixing fund houses."""
    projects = _diverse_projects(rows)
    with ThreadPoolExecutor(max_workers=MAX_FUNDS) as pool:
        navs = dict(zip(projects, pool.map(latest_navs_for_project, projects)))
    results = [
        result
        for proj_id, class_rows in projects.items()
        for result in _class_results(class_rows, navs[proj_id])
    ]
    return results[:MAX_RESULTS]


def _diverse_projects(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group rows by fund, then take funds in turns per fund house.

    The SEC lists one house's funds together, so the first rows are often a
    single house; taking turns shows several houses (exact code matches,
    ranked first, stay first).
    """
    by_project: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        proj_id = str(row.get("proj_id") or "")
        if proj_id:
            by_project.setdefault(proj_id, []).append(row)
    by_house: dict[str, list[str]] = {}
    for proj_id, project_rows in by_project.items():
        by_house.setdefault(str(project_rows[0].get("unique_id") or ""), []).append(proj_id)
    turns = [proj_id for group in zip_longest(*by_house.values()) for proj_id in group if proj_id]
    return {proj_id: by_project[proj_id] for proj_id in turns[:MAX_FUNDS]}


def _class_results(rows: list[dict[str, Any]], navs: dict[str, Any]) -> list[FundSearchResult]:
    """One result per class of a fund that has a recent NAV (duplicates removed)."""
    results: dict[str, FundSearchResult] = {}
    for row in rows:
        class_name = str(row.get("fund_class_name") or SINGLE_CLASS_NAME).strip().upper()
        quote = navs.get(class_name)
        if quote is None:
            continue
        result = _to_result(row, class_name, quote)
        results.setdefault(result.symbol, result)
    return list(results.values())


def payout_type(class_code: str, description: str) -> str:
    """ "A" accumulate, "D" dividend, "" unknown — from "(A)/(D)" or the class description.

    Example:
        >>> payout_type("KFSEQ-D", "ต้องการรับผลตอบแทนสม่ำเสมอจากเงินปันผล")
        'D'
    """
    suffix = _PAYOUT_SUFFIX.search(class_code)
    if suffix:
        return suffix.group(1)
    if "สะสม" in description:
        return "A"
    return "D" if "เงินปันผล" in description else ""


def _to_result(row: dict[str, Any], class_name: str, quote: Any) -> FundSearchResult:
    """Profile row + NAV → result; single-class funds use the fund code as symbol."""
    fund_code = str(row.get("proj_abbr_name") or "").strip().upper()
    symbol = fund_code if class_name == SINGLE_CLASS_NAME else class_name
    amc = _to_amc({"comp_name_th": row.get("comp_name_th", "")}).short_name_th
    return FundSearchResult(
        symbol=symbol,
        proj_id=str(row.get("proj_id") or ""),
        fund_code=fund_code,
        name_th=str(row.get("proj_name_th") or "").strip(),
        name_en=str(row.get("proj_name_en") or "").strip(),
        amc=amc,
        payout=payout_type(symbol, str(row.get("fund_class_description") or "")),
        nav=quote.nav,
        nav_date=quote.nav_date,
    )
