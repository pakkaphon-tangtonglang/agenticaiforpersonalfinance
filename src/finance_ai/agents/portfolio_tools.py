"""LangGraph tool for the user's investment portfolio (what they own).

Holdings are not the watchlist: the watchlist is assets the user only
follows; holdings have quantity, cost and gain/loss.
"""

from decimal import Decimal
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from finance_ai.agents.session_helper import get_tool_session
from finance_ai.tools.investment_calculator import HoldingSummary
from finance_ai.tools.investment_service import get_portfolio_summary

EMPTY_PORTFOLIO_MESSAGE: str = (
    "ยังไม่มีรายการถือครองในพอร์ต (นำเข้าได้จากหน้าสินทรัพย์ หรือบอกว่าซื้ออะไรไว้)"
)


@tool
def get_portfolio_holdings(
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """ดูพอร์ตการลงทุน: สินทรัพย์ที่ผู้ใช้ถือจริง (จำนวน ต้นทุน มูลค่า กำไร/ขาดทุน).

    ใช้เมื่อผู้ใช้ถามถึง "รายการถือหุ้น", "หุ้นที่ถืออยู่", "พอร์ต", "ถืออะไรอยู่",
    "กำไรขาดทุนพอร์ต". ไม่ใช่ Watchlist (รายการที่แค่ติดตาม) — ใช้ manage_watchlist แทน
    เฉพาะเมื่อผู้ใช้พูดถึงการติดตาม/watchlist.

    Args:
        user_id: UUID ของผู้ใช้ (injected จาก graph state).
        db_session_factory: Session factory (injected จาก graph state).

    Returns:
        Dict with holding_count, totals, holdings (money as strings, None when
        no price is stored) and a message when the portfolio is empty.

    Example:
        >>> get_portfolio_holdings.invoke({"user_id": uid, "db_session_factory": f})
    """
    with get_tool_session(db_session_factory) as session:
        summary = get_portfolio_summary(session, user_id)
    return {
        "holding_count": summary.holding_count,
        "total_cost": _money(summary.total_cost),
        "total_current_value": _money(summary.total_current_value),
        "total_unrealized_gain_loss": _money(summary.total_unrealized_gain_loss),
        "holdings": [holding_to_dict(holding) for holding in summary.holdings],
        "message": "" if summary.holdings else EMPTY_PORTFOLIO_MESSAGE,
    }


def holding_to_dict(holding: HoldingSummary) -> dict[str, Any]:
    """Convert one holding to plain values (money as strings, quantity unpadded).

    Used by the agent tool and the /assets/holdings endpoint.

    Args:
        holding: Holding summary from the investment service.

    Returns:
        Dict of display-ready fields (None where no price is stored).
    """
    return {
        "symbol": holding.symbol,
        "name": holding.name,
        "asset_type": holding.asset_type_label,
        "quantity": _plain(holding.quantity),
        "average_cost_per_unit": _money(holding.average_cost_per_unit),
        "total_cost": _money(holding.total_cost),
        "current_price_per_unit": _money(holding.current_price_per_unit),
        "current_value": _money(holding.current_value),
        "unrealized_gain_loss": _money(holding.unrealized_gain_loss),
        "gain_loss_percentage": _plain(holding.gain_loss_percentage),
    }


def _money(amount: Decimal | None) -> str | None:
    """Money with 2 decimals, or None when unknown."""
    return None if amount is None else str(amount.quantize(Decimal("0.01")))


def _plain(value: Decimal | None) -> str | None:
    """Decimal without trailing zeros (100.0000 → 100), or None."""
    if value is None:
        return None
    text = format(value.normalize(), "f")
    return text
