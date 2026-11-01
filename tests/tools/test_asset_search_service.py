"""Tests for asset_search_service (stocks + Thai funds in one search box)."""

from datetime import date
from decimal import Decimal
from unittest.mock import patch

from finance_ai.tools.asset_search_service import search_all_assets
from finance_ai.tools.fund_search_service import FundSearchOutcome, FundSearchResult
from finance_ai.tools.market_data_models import AssetSymbolMatch

_SEARCH = "finance_ai.tools.asset_search_service.search_funds"


def ptt() -> AssetSymbolMatch:
    """A Yahoo match."""
    return AssetSymbolMatch(symbol="PTT.BK", name="PTT", exchange="SET", quote_type="EQUITY")


def k_value() -> FundSearchResult:
    """A fund search result."""
    return FundSearchResult(
        symbol="K-VALUE-A(A)", proj_id="P1", fund_code="K-VALUE", name_th="กองทุนเปิดเค หุ้นปันผล",
        name_en="", amc="กสิกรไทย", payout="A", nav=Decimal("7.3609"), nav_date=date(2026, 10, 30),
    )  # fmt: skip


def test_stocks_and_funds_together() -> None:
    """Both sources are returned with categories; no LLM call when something matched."""
    asked: list[str] = []

    def expand(query: str) -> list[str]:
        asked.append(query)
        return []

    with patch(_SEARCH, return_value=FundSearchOutcome(results=[k_value()])) as search:
        response = search_all_assets("ปันผล", lambda q: [ptt()], expand=expand)
    assert [r["category"] for r in response["results"]] == ["thai_stock", "fund"]
    fund = response["results"][1]
    assert (fund["type"], fund["nav"], fund["amc"], fund["payout"]) == (
        "FUND",
        "7.3609",
        "กสิกรไทย",
        "A",
    )
    assert search.call_count == 1
    assert not asked
    assert response["clarify"] is None


def test_llm_expansion_only_when_both_empty() -> None:
    """With nothing found, the fund search runs again with the expander."""
    outcomes = [
        FundSearchOutcome(results=[]),
        FundSearchOutcome(results=[k_value()], expanded_terms=["ปันผล"]),
    ]

    def expand(_query: str) -> list[str]:
        return ["ปันผล"]

    with patch(_SEARCH, side_effect=outcomes) as search:
        response = search_all_assets("หุ้นจ่ายเงิน", lambda q: [], expand=expand)
    assert search.call_args_list[1].kwargs == {"expand": expand}
    assert response["expanded_terms"] == ["ปันผล"]
    assert len(response["results"]) == 1


def test_not_found_returns_questions() -> None:
    """Nothing anywhere → a not_found clarification."""
    with patch(_SEARCH, return_value=FundSearchOutcome(results=[])):
        response = search_all_assets("zzqqxx", lambda q: [], expand=lambda q: [])
    assert response["results"] == []
    assert response["clarify"]["reason"] == "not_found"
