"""Tests for the search_thai_funds agent tool."""

from datetime import date
from decimal import Decimal
from unittest.mock import patch

from finance_ai.agents.asset_monitoring_agent import ASSET_MONITORING_TOOLS
from finance_ai.agents.fund_search_tools import search_thai_funds
from finance_ai.tools.fund_search_service import FundSearchOutcome, FundSearchResult

_SEARCH = "finance_ai.agents.fund_search_tools.search_funds"


def k_value_dividend() -> FundSearchResult:
    """A dividend class of K-VALUE."""
    return FundSearchResult(
        symbol="K-VALUE-A(D)",
        proj_id="P1",
        fund_code="K-VALUE",
        name_th="กองทุนเปิดเค หุ้นปันผล",
        name_en="",
        amc="กสิกรไทย",
        payout="D",
        nav=Decimal("6.5077"),
        nav_date=date(2026, 10, 30),
    )


def test_returns_funds_with_thai_labels() -> None:
    """Funds come back as plain strings the agent can show and reuse."""
    with patch(_SEARCH, return_value=FundSearchOutcome(results=[k_value_dividend()])):
        result = search_thai_funds.invoke({"query": "เค ปันผล"})
    assert result["count"] == 1
    assert result["funds"][0] == {
        "symbol": "K-VALUE-A(D)",
        "fund_code": "K-VALUE",
        "name": "กองทุนเปิดเค หุ้นปันผล",
        "amc": "กสิกรไทย",
        "payout": "รับเงินปันผล",
        "nav": "6.5077",
        "nav_date": "2026-10-30",
    }


def test_nothing_found_asks_again() -> None:
    """No match with fund data available → ask for another name."""
    with (
        patch(_SEARCH, return_value=FundSearchOutcome(results=[])),
        patch("finance_ai.tools.fund_nav_client.is_fund_data_available", return_value=True),
    ):
        result = search_thai_funds.invoke({"query": "zzz"})
    assert result["needs_clarification"] is True


def test_without_key_explains() -> None:
    """No SEC key → a plain explanation instead of a question."""
    result = search_thai_funds.invoke({"query": "ปันผล"})
    assert "SEC_API_KEY" in result["error"]


def test_registered_on_asset_agent() -> None:
    """The asset agent can search funds before adding one."""
    assert "search_thai_funds" in [tool.name for tool in ASSET_MONITORING_TOOLS]
