"""Decide when an asset search is unsure, and which questions narrow it down.

Pure functions over search results (no I/O), so the rules are easy to test.
The web page shows the questions one at a time; each option carries a filter
(`{"key": "category", "value": "fund"}`) that the page applies to the results.
A typed "something else" answer re-runs the search with the user's words.

Unsure means: nothing found, or more than a handful of results that differ in
asset type, fund house, or dividend policy.
"""

from collections import Counter
from typing import Any

from pydantic import BaseModel

GOLD = "gold"
THAI_STOCK = "thai_stock"
FUND = "fund"
FOREIGN_STOCK = "foreign_stock"
CRYPTO = "crypto"
OTHER = "other"
CATEGORY_LABELS: dict[str, str] = {
    GOLD: "ทองคำ",
    THAI_STOCK: "หุ้นไทย",
    FUND: "กองทุนรวมไทย",
    FOREIGN_STOCK: "หุ้นต่างประเทศ",
    CRYPTO: "คริปโต",
    OTHER: "อื่นๆ (สัญญาล่วงหน้า ดัชนี ETF)",
}
PAYOUT_LABELS: dict[str, str] = {"A": "สะสมมูลค่า (ไม่จ่ายปันผล)", "D": "รับเงินปันผล"}
FEW_RESULTS = 4  # at or below this many results, just show them
MAX_AMC_OPTIONS = 4


class ClarifyOption(BaseModel):
    """One answer; selecting it keeps results whose `key` equals `value`."""

    label: str
    key: str
    value: str


class ClarifyQuestion(BaseModel):
    """One question shown in the pop-up."""

    id: str
    question: str
    options: list[ClarifyOption]


class Clarification(BaseModel):
    """Questions to ask, plus a hint for the free-text answer."""

    reason: str
    questions: list[ClarifyQuestion]
    custom_placeholder: str = "พิมพ์ชื่อหรือตัวย่ออื่น…"


def stock_category(symbol: str, quote_type: str) -> str:
    """Category of a Yahoo search result.

    Example:
        >>> stock_category("PTT.BK", "EQUITY")
        'thai_stock'
    """
    kind = quote_type.strip().upper()
    if kind == "CRYPTOCURRENCY":
        return CRYPTO
    if kind != "EQUITY":
        return OTHER
    return THAI_STOCK if symbol.strip().upper().endswith(".BK") else FOREIGN_STOCK


def build_clarification(results: list[dict[str, Any]]) -> Clarification | None:
    """Questions for an unsure search, or None when the results are clear enough.

    Args:
        results: Search results with "category" and, for funds, "amc"/"payout".

    Returns:
        A Clarification with 1-3 questions, or None.

    Example:
        >>> build_clarification([]).reason
        'not_found'
    """
    if not results:
        # Nothing to filter yet: ask the type, then for new words to search with.
        retry = ClarifyQuestion(
            id="retry", question="ลองพิมพ์ชื่อใหม่ (ชื่อเต็ม ตัวย่อ หรือรหัสกองทุน)", options=[]
        )
        return Clarification(
            reason="not_found",
            questions=[_category_question(list(CATEGORY_LABELS)), retry],
            custom_placeholder="เช่น Apple, PTT, กรุงศรีปันผล, K-USA…",
        )
    if len(results) <= FEW_RESULTS:
        return None
    questions = [
        question
        for question in (
            _category_from(results),
            _amc_from(results),
            _payout_from(results),
        )
        if question is not None
    ]
    return Clarification(reason="many", questions=questions) if questions else None


def _category_question(
    categories: list[str], counts: Counter[str] | None = None
) -> ClarifyQuestion:
    """'What are you looking for?' with one option per category."""
    options = [
        ClarifyOption(label=_with_count(CATEGORY_LABELS[c], counts, c), key="category", value=c)
        for c in categories
    ]
    return ClarifyQuestion(id="category", question="กำลังหาสินทรัพย์ประเภทไหน", options=options)


def _category_from(results: list[dict[str, Any]]) -> ClarifyQuestion | None:
    """Ask about asset type when results mix types."""
    counts = Counter(str(result.get("category", OTHER)) for result in results)
    if len(counts) < 2:
        return None
    ordered = [category for category in CATEGORY_LABELS if category in counts]
    return _category_question(ordered, counts)


def _amc_from(results: list[dict[str, Any]]) -> ClarifyQuestion | None:
    """Ask about fund house when funds come from several houses."""
    counts = Counter(str(r["amc"]) for r in results if r.get("category") == FUND and r.get("amc"))
    if len(counts) < 2:
        return None
    options = [
        ClarifyOption(label=f"{amc} ({count})", key="amc", value=amc)
        for amc, count in counts.most_common(MAX_AMC_OPTIONS)
    ]
    return ClarifyQuestion(id="amc", question="กองทุนของ บลจ. ไหน", options=options)


def _payout_from(results: list[dict[str, Any]]) -> ClarifyQuestion | None:
    """Ask accumulate vs dividend when both kinds of fund class are present."""
    payouts = {str(r.get("payout")) for r in results if r.get("category") == FUND}
    if not {"A", "D"} <= payouts:
        return None
    options = [ClarifyOption(label=PAYOUT_LABELS[p], key="payout", value=p) for p in ("A", "D")]
    return ClarifyQuestion(id="payout", question="อยากได้ชนิดหน่วยแบบไหน", options=options)


def _with_count(label: str, counts: Counter[str] | None, key: str) -> str:
    """'กองทุนรวมไทย (5)' when counts are known."""
    return f"{label} ({counts[key]})" if counts else label
