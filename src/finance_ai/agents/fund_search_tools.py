"""LangGraph tool: find Thai mutual funds by name, fund house or code (SEC data).

Lets the asset agent turn "กองทุนกรุงศรีปันผล" into real fund class codes with
their latest NAV before calling add_portfolio_holding. Results come from the
SEC Open API only; the LLM expansion merely suggests search terms.
"""

from typing import Any

from langchain_core.tools import tool

from finance_ai.agents.clarification import clarification
from finance_ai.tools.fund_query_expander import expand_fund_query
from finance_ai.tools.fund_search_service import FundSearchResult, search_funds
from finance_ai.tools.search_clarifier import PAYOUT_LABELS

ASK_FUND_NAME = "ขอชื่อกองทุน ชื่อ บลจ. หรือรหัสกองทุนอีกครั้ง เช่น กรุงศรีปันผล, K-USA"
NO_FUND_DATA = "ยังไม่ได้ตั้งค่า SEC_API_KEY จึงค้นหากองทุนรวมไม่ได้"


@tool
def search_thai_funds(query: str) -> dict[str, Any]:
    """ค้นหากองทุนรวมไทยจากชื่อกองทุน ชื่อ บลจ. หรือรหัสกองทุน พร้อม NAV ล่าสุด (ข้อมูล ก.ล.ต.).

    ใช้ก่อน add_portfolio_holding เมื่อผู้ใช้พูดถึงกองทุนด้วยชื่อ เช่น "กองทุนกรุงศรีปันผล",
    "กองทุน S&P 500", "SSF". ใช้ค่า symbol ที่ได้เป็น symbol ของ add_portfolio_holding
    (asset_type="mutual_fund"). ถ้าได้หลายรายการ ให้แสดงให้ผู้ใช้เลือกก่อน ห้ามเลือกเอง.

    Args:
        query: ชื่อกองทุน ชื่อ บลจ. หรือรหัส เช่น "กรุงศรี ปันผล", "K-USA".

    Returns:
        {"count", "funds": [{symbol, fund_code, name, amc, payout, nav, nav_date}],
        "expanded_terms"} or a clarification when nothing was found.

    Example:
        >>> search_thai_funds.invoke({"query": "กรุงศรีปันผล"})  # doctest: +SKIP
    """
    outcome = search_funds(query, expand=expand_fund_query)
    if not outcome.results:
        from finance_ai.tools.fund_nav_client import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
            is_fund_data_available,
        )

        if not is_fund_data_available():
            return {"error": NO_FUND_DATA}
        return clarification(f"No Thai fund matches '{query}'.", ASK_FUND_NAME)
    return {
        "count": len(outcome.results),
        "funds": [_fund_row(fund) for fund in outcome.results],
        "expanded_terms": outcome.expanded_terms,
    }


def _fund_row(fund: FundSearchResult) -> dict[str, str]:
    """One fund class for the agent (strings only)."""
    return {
        "symbol": fund.symbol,
        "fund_code": fund.fund_code,
        "name": fund.name_th or fund.name_en,
        "amc": fund.amc,
        "payout": PAYOUT_LABELS.get(fund.payout, "ไม่ระบุ"),
        "nav": format(fund.nav.normalize(), "f"),
        "nav_date": fund.nav_date.isoformat(),
    }
