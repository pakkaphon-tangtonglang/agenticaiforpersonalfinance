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


_GOLD_PRICE = "finance_ai.tools.asset_search_service.thai_gold_price_per_baht_weight"


_WORLD_PRICE = "finance_ai.tools.world_gold_service.fetch_current_price"


def test_gold_query_shows_calculated_and_api_gold_first() -> None:
    """ทองคำ offers the calculated Thai bar, then API world gold (Yahoo's GC=F row dropped)."""
    futures = AssetSymbolMatch(symbol="GC=F", name="Gold", exchange="CMX", quote_type="FUTURE")
    micro = AssetSymbolMatch(symbol="MGC=F", name="Micro", exchange="CMX", quote_type="FUTURE")
    with (
        patch(_SEARCH, return_value=FundSearchOutcome(results=[k_value()])),
        patch(_GOLD_PRICE, return_value=Decimal("61500.00")),
        patch(_WORLD_PRICE, return_value=Decimal("4012.5")),
    ):
        response = search_all_assets("ทองคำ", lambda q: [futures, micro], expand=lambda q: [])
    rows = [(r["symbol"], r["type"], r["price"]) for r in response["results"][:2]]
    assert rows == [("THGOLD965", "GOLD", "61500.00"), ("GC=F", "WORLD_GOLD", "4012.5")]
    assert [r["symbol"] for r in response["results"][2:]] == ["MGC=F", "K-VALUE-A(A)"]


def test_other_query_has_no_gold_bar() -> None:
    """The gold bar only shows for gold searches (and its price is not fetched)."""
    with (
        patch(_SEARCH, return_value=FundSearchOutcome(results=[])),
        patch(_GOLD_PRICE) as gold_price,
    ):
        response = search_all_assets("ปตท", lambda q: [ptt()], expand=lambda q: [])
    assert [r["symbol"] for r in response["results"]] == ["PTT.BK"]
    gold_price.assert_not_called()


def test_ask_ai_always_expands_fund_terms() -> None:
    """ask_ai=True passes the expander with always_expand, even when stocks matched."""
    with patch(_SEARCH, return_value=FundSearchOutcome(results=[])) as search:
        search_all_assets("หุ้นอเมริกา", lambda q: [ptt()], expand=str.split, ask_ai=True)
    assert search.call_args.kwargs == {"expand": str.split, "always_expand": True}
