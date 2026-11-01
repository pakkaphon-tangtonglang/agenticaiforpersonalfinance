"""Latest NAV of Thai mutual funds from the SEC Open API (Fund API V2).

NAV rows (GET /v2/fund/daily-info/nav) carry a `fund_class_name`. Multi-class
funds name each class ("K-VALUE-A(A)"); single-class funds use "main", so
their NAV can only be found through the fund project (`proj_id`). Lookups ask
for the last two weeks and keep the newest row per class.
"""

from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel

from finance_ai.tools import sec_fund_api

NAV_LOOKBACK_DAYS = 14  # covers long Thai holidays (e.g. Songkran)
SINGLE_CLASS_NAME = "MAIN"  # SEC's class name for single-class funds
NAV_TTL_SECONDS = 600.0  # NAV changes once a day
_MAX_NAV_PAGES = 3


class FundNavQuote(BaseModel):
    """Latest NAV of one fund unit class.

    Example:
        >>> FundNavQuote(class_name="K-VALUE-A(A)", proj_id="M0023_2538",
        ...     nav=Decimal("7.3609"), nav_date=date(2026, 10, 30)).class_name
        'K-VALUE-A(A)'
    """

    class_name: str
    proj_id: str
    nav: Decimal
    nav_date: date


def is_fund_data_available() -> bool:
    """True when an SEC API key is configured."""
    return sec_fund_api.is_configured()


def clear_cache() -> None:
    """Forget cached SEC answers (used by tests)."""
    sec_fund_api.clear_cache()


def latest_navs_for_project(proj_id: str) -> dict[str, FundNavQuote]:
    """Latest NAV of every active class of one fund, keyed by upper-case class name.

    Example:
        >>> sorted(latest_navs_for_project("M0023_2538"))  # doctest: +SKIP
        ['K-VALUE-A(A)', 'K-VALUE-A(D)', 'K-VALUE-Z(A)']
    """
    return _newest_per_class(_recent_nav_rows({"proj_id": proj_id}))


def fetch_latest_nav(symbol: str) -> Decimal | None:
    """Latest NAV for a stored fund symbol: a class code or a single-class fund code.

    Tries the class name first ("K-USA-A(A)"), then treats the symbol as a
    fund code ("K-VALUERMF") whose only class is "main".

    Example:
        >>> fetch_latest_nav("K-VALUERMF")  # doctest: +SKIP
        Decimal('34.443')
    """
    wanted = symbol.strip().upper()
    if not wanted:
        return None
    by_class = _newest_per_class(_recent_nav_rows({"fund_class_name": wanted}))
    if wanted in by_class:
        return by_class[wanted].nav
    return _single_class_nav(wanted)


def _single_class_nav(fund_code: str) -> Decimal | None:
    """NAV of a fund whose code is `fund_code` and whose NAV class is "main"."""
    profiles = sec_fund_api.fetch_items(
        sec_fund_api.PROFILES_PATH, {"project_info": fund_code}, ttl_seconds=86400.0
    )
    proj_ids = {
        str(row.get("proj_id"))
        for row in profiles
        if str(row.get("proj_abbr_name", "")).strip().upper() == fund_code
    }
    for proj_id in sorted(proj_ids):
        quotes = latest_navs_for_project(proj_id)
        quote = quotes.get(SINGLE_CLASS_NAME) or quotes.get(fund_code)
        if quote is not None:
            return quote.nav
    return None


def _recent_nav_rows(filters: dict[str, str]) -> list[dict[str, Any]]:
    """NAV rows from the last NAV_LOOKBACK_DAYS for the given filters."""
    start = (date.today() - timedelta(days=NAV_LOOKBACK_DAYS)).isoformat()
    return sec_fund_api.fetch_items(
        sec_fund_api.NAV_PATH,
        {**filters, "start_nav_date": start},
        max_pages=_MAX_NAV_PAGES,
        ttl_seconds=NAV_TTL_SECONDS,
    )


def _newest_per_class(rows: list[dict[str, Any]]) -> dict[str, FundNavQuote]:
    """Keep the most recent valid row for each class name."""
    newest: dict[str, FundNavQuote] = {}
    for row in rows:
        quote = _to_quote(row)
        if quote is None:
            continue
        current = newest.get(quote.class_name)
        if current is None or quote.nav_date > current.nav_date:
            newest[quote.class_name] = quote
    return newest


def _to_quote(row: dict[str, Any]) -> FundNavQuote | None:
    """Parse one API row; rows without a positive NAV are skipped."""
    try:
        nav = Decimal(str(row["last_val"]))
        quote = FundNavQuote(
            class_name=str(row["fund_class_name"]).strip().upper(),
            proj_id=str(row.get("proj_id") or ""),
            nav=nav,
            nav_date=date.fromisoformat(str(row["nav_date"])),
        )
    except (KeyError, TypeError, ValueError, InvalidOperation):
        return None
    return quote if nav.is_finite() and nav > 0 else None
