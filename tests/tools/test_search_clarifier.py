"""Tests for search_clarifier (when an asset search should ask the user)."""

from typing import Any

import pytest

from finance_ai.tools.search_clarifier import build_clarification, stock_category


@pytest.mark.parametrize(
    ("symbol", "quote_type", "expected"),
    [
        ("PTT.BK", "EQUITY", "thai_stock"),
        ("AAPL", "EQUITY", "foreign_stock"),
        ("BTC-USD", "CRYPTOCURRENCY", "crypto"),
        ("GC=F", "FUTURE", "other"),
    ],
)
def test_stock_category(symbol: str, quote_type: str, expected: str) -> None:
    """Yahoo results are grouped by type and market."""
    assert stock_category(symbol, quote_type) == expected


def fund(amc: str, payout: str = "") -> dict[str, Any]:
    """A fund search result."""
    return {"category": "fund", "amc": amc, "payout": payout}


def test_not_found_asks_type_then_new_words() -> None:
    """No results: ask the type, then for a new search (free text only)."""
    clarify = build_clarification([])
    assert clarify is not None and clarify.reason == "not_found"
    assert [q.id for q in clarify.questions] == ["category", "retry"]
    assert not clarify.questions[1].options


def test_few_results_are_shown_without_questions() -> None:
    """Four or fewer results need no questions."""
    assert build_clarification([fund("ก"), fund("ข"), {"category": "thai_stock"}]) is None


def test_mixed_results_ask_type_house_and_payout() -> None:
    """Stocks + funds from several houses with A and D classes → three questions."""
    results = [
        {"category": "thai_stock"},
        fund("กสิกรไทย", "A"),
        fund("กสิกรไทย", "D"),
        fund("กรุงศรี", "A"),
        fund("ไทยพาณิชย์"),
    ]
    clarify = build_clarification(results)
    assert clarify is not None
    assert [q.id for q in clarify.questions] == ["category", "amc", "payout"]
    category, amc, payout = clarify.questions
    assert [o.label for o in category.options] == ["หุ้นไทย (1)", "กองทุนรวมไทย (4)"]
    assert amc.options[0].label == "กสิกรไทย (2)"
    assert [(o.key, o.value) for o in payout.options] == [("payout", "A"), ("payout", "D")]


def test_one_house_one_type_needs_no_questions() -> None:
    """Many funds from one house, all accumulate: nothing to ask."""
    assert build_clarification([fund("กสิกรไทย", "A") for _ in range(6)]) is None


def test_amc_options_capped() -> None:
    """At most four fund houses are offered, most results first."""
    houses = ["ก", "ข", "ค", "ง", "จ", "ก", "ก"]
    clarify = build_clarification([fund(house) for house in houses])
    assert clarify is not None
    assert [o.value for o in clarify.questions[0].options] == ["ก", "ข", "ค", "ง"]
