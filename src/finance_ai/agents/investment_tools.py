"""LangGraph tool wrappers for investment portfolio tracking.

Tools accept string inputs from LLM, parse them to proper types,
and persist records via the investment service layer. InjectedState
provides user_id and db_session_factory from the agent graph state.
"""

from datetime import date
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from finance_ai.agents.expense_tools import parse_date_value
from finance_ai.agents.session_helper import get_tool_session
from finance_ai.agents.tax_tools import parse_decimal_value
from finance_ai.tools.investment_calculator import (
    validate_asset_type,
    validate_price_per_unit,
    validate_quantity,
    validate_symbol,
)
from finance_ai.tools.investment_constants import ASSET_TYPES


@tool
def view_portfolio(
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """View the user's complete investment portfolio.

    Use this tool when the user wants to see their portfolio,
    holdings, or investment summary.

    Args:
        user_id: UUID of the user (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict with portfolio summary data from the database.
    """
    from finance_ai.tools.investment_service import (  # noqa: PLC0415
        get_portfolio_summary,
    )

    with get_tool_session(db_session_factory) as session:
        summary = get_portfolio_summary(session, user_id)

    return {
        "action": "view_portfolio",
        "total_value": str(summary.total_current_value),
        "total_cost": str(summary.total_cost),
        "total_gain_loss": str(summary.total_unrealized_gain_loss),
        "holding_count": summary.holding_count,
        "holdings": [
            {
                "symbol": h.symbol,
                "name": h.name,
                "quantity": str(h.quantity),
                "current_value": str(h.current_value),
            }
            for h in summary.holdings
        ],
    }


@tool
def add_holding(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    symbol: str,
    asset_type: str,
    name: str,
    quantity: str,
    price_per_unit: str,
    purchase_date: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """Add a new investment holding to the portfolio.

    Use this tool when the user wants to add a stock or mutual fund.

    Args:
        symbol: Asset symbol (e.g., "PTT.BK" for stock, "K-EQUITY" for fund).
        asset_type: "stock" or "mutual_fund".
        name: Display name of the asset (e.g., "PTT", "K Equity Fund").
        quantity: Number of units (e.g., "100").
        price_per_unit: Purchase price per unit in THB (e.g., "35.50").
        purchase_date: Date in YYYY-MM-DD format. Defaults to today.
        user_id: UUID of the user (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict with validated and persisted holding data.
    """
    normalized_type = validate_asset_type(asset_type)
    normalized_symbol = validate_symbol(symbol, normalized_type)
    parsed_quantity = parse_decimal_value(quantity, "quantity")
    validate_quantity(parsed_quantity)
    parsed_price = parse_decimal_value(price_per_unit, "price_per_unit")
    validate_price_per_unit(parsed_price)
    parsed_date = (
        parse_date_value(purchase_date, "purchase_date") if purchase_date else date.today()
    )
    type_label = ASSET_TYPES.get(normalized_type, normalized_type)

    _persist_holding(
        db_session_factory,
        user_id,
        normalized_type,
        normalized_symbol,
        name,
        parsed_quantity,
        parsed_price,
        parsed_date,
    )

    return {
        "action": "add_holding",
        "symbol": normalized_symbol,
        "asset_type": normalized_type,
        "asset_type_label": type_label,
        "name": name,
        "quantity": str(parsed_quantity),
        "price_per_unit": str(parsed_price),
        "purchase_date": parsed_date.isoformat(),
    }


def _persist_holding(  # noqa: PLR0913
    db_session_factory: Any,
    user_id: str,
    asset_type: str,
    symbol: str,
    name: str,
    quantity: Any,
    price_per_unit: Any,
    purchase_date: date,
) -> None:
    """Save investment holding to database via the service layer.

    Args:
        db_session_factory: Session factory or None.
        user_id: UUID of the user.
        asset_type: Normalized asset type key.
        symbol: Normalized asset symbol.
        name: Display name of the asset.
        quantity: Number of units (Decimal).
        price_per_unit: Price per unit (Decimal).
        purchase_date: Date of purchase.
    """
    from finance_ai.tools.investment_service import (  # noqa: PLC0415
        add_investment_holding,
    )

    with get_tool_session(db_session_factory) as session:
        add_investment_holding(
            session,
            user_id,
            asset_type,
            symbol,
            name,
            quantity,
            price_per_unit,
            purchase_date,
        )


@tool
def import_csv(
    csv_content: str,
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """Import investment holdings from CSV data.

    Use this tool when the user provides CSV data to import portfolio holdings.

    Args:
        csv_content: Raw CSV string with header row and holding data.
        user_id: UUID of the user (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict with import results including row count.
    """
    from finance_ai.tools.investment_service import (  # noqa: PLC0415
        import_holdings_from_csv,
    )

    with get_tool_session(db_session_factory) as session:
        holdings = import_holdings_from_csv(session, user_id, csv_content)
        imported = [{"symbol": h.symbol, "name": h.name or ""} for h in holdings]

    return {
        "action": "import_csv",
        "row_count": len(imported),
        "imported": imported,
    }


@tool
def refresh_prices(
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """Refresh market prices for all holdings in the portfolio.

    Use this tool when the user wants to update prices from the market.

    Args:
        user_id: UUID of the user (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict with the number of holdings updated.
    """
    from finance_ai.tools.investment_service import (  # noqa: PLC0415
        refresh_portfolio_prices,
    )

    with get_tool_session(db_session_factory) as session:
        updated_count = refresh_portfolio_prices(session, user_id)

    return {
        "action": "refresh_prices",
        "updated_count": updated_count,
    }


@tool
def lookup_holding(
    symbol: str,
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """Look up details for a specific holding by symbol.

    Use this tool when the user asks about a specific stock or fund.

    Args:
        symbol: Asset symbol to look up (e.g., "PTT.BK").
        user_id: UUID of the user (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict with holding details if found, or not_found status.
    """
    from finance_ai.tools.investment_service import (  # noqa: PLC0415
        get_user_holding_by_symbol,
    )

    normalized_symbol = symbol.strip().upper()
    with get_tool_session(db_session_factory) as session:
        holding = get_user_holding_by_symbol(
            session,
            user_id,
            normalized_symbol,
        )

    if holding is None:
        return {
            "action": "lookup_holding",
            "symbol": normalized_symbol,
            "status": "not_found",
        }

    return {
        "action": "lookup_holding",
        "symbol": holding.symbol,
        "name": holding.name,
        "quantity": str(holding.quantity),
        "average_cost": str(holding.average_cost_per_unit),
        "current_price": str(holding.current_price_per_unit),
        "current_value": str(holding.current_value),
        "gain_loss": str(holding.unrealized_gain_loss),
        "status": "found",
    }


@tool
def get_investment_advice(
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """Gather portfolio data for LLM-based investment recommendations.

    Use this tool when the user asks for investment advice or recommendations.
    Returns portfolio summary data for the LLM to analyze.

    Args:
        user_id: UUID of the user (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict with portfolio data for LLM analysis.
    """
    from finance_ai.tools.investment_service import (  # noqa: PLC0415
        get_portfolio_summary,
    )

    with get_tool_session(db_session_factory) as session:
        summary = get_portfolio_summary(session, user_id)

    return {
        "action": "get_investment_advice",
        "total_value": str(summary.total_current_value),
        "total_cost": str(summary.total_cost),
        "total_gain_loss": str(summary.total_unrealized_gain_loss),
        "holding_count": summary.holding_count,
        "holdings": [
            {
                "symbol": h.symbol,
                "name": h.name,
                "asset_type": h.asset_type,
                "quantity": str(h.quantity),
                "average_cost": str(h.average_cost_per_unit),
                "current_value": str(h.current_value),
                "gain_loss": str(h.unrealized_gain_loss),
            }
            for h in summary.holdings
        ],
    }
