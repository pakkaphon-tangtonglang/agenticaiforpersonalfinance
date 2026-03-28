"""LangGraph tool wrappers for cross-agent data sharing.

These tools allow one agent to query data from another agent's domain.
For example, Tax Agent can fetch portfolio data, Planning Agent can
fetch expense summaries. All tools use InjectedState for user_id
and db_session_factory.
"""

from datetime import date
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from finance_ai.agents.session_helper import get_tool_session


@tool
def get_expense_summary_cross(
    year: str = "",
    month: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """Get expense summary from the Expense domain.

    Use this tool to view a user's monthly expense breakdown
    when analyzing from another agent (Tax, Planning, etc.).

    Args:
        year: Year to query (e.g., "2026"). Defaults to current year.
        month: Month to query (e.g., "3"). Defaults to current month.
        user_id: UUID of the user (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict with expense summary including category breakdown.
    """
    today = date.today()
    parsed_year = int(year) if year else today.year
    parsed_month = int(month) if month else today.month
    return _fetch_expense_summary(db_session_factory, user_id, parsed_year, parsed_month)


def _fetch_expense_summary(
    db_session_factory: Any,
    user_id: str,
    year: int,
    month: int,
) -> dict[str, Any]:
    """Fetch expense summary via cross-agent service.

    Args:
        db_session_factory: Session factory or None.
        user_id: UUID of the user.
        year: Year to query.
        month: Month to query.

    Returns:
        Expense summary dict.
    """
    from finance_ai.tools.cross_agent_service import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        get_expense_summary,
    )

    with get_tool_session(db_session_factory) as session:
        return get_expense_summary(session, user_id, year, month)


@tool
def get_portfolio_summary_cross(
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """Get investment portfolio summary from the Investment domain.

    Use this tool to view a user's portfolio when analyzing from
    another agent (Tax for capital gains, Planning for net worth).

    Args:
        user_id: UUID of the user (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict with portfolio value, gains/losses, and holdings.
    """
    from finance_ai.tools.cross_agent_service import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        get_portfolio_summary,
    )

    with get_tool_session(db_session_factory) as session:
        return get_portfolio_summary(session, user_id)


@tool
def get_goals_summary_cross(
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """Get financial goals summary from the Planning domain.

    Use this tool to view a user's financial goals and progress
    when analyzing from another agent (Expense, Investment).

    Args:
        user_id: UUID of the user (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict with goal counts, progress, and per-goal details.
    """
    from finance_ai.tools.cross_agent_service import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        get_goals_summary,
    )

    with get_tool_session(db_session_factory) as session:
        return get_goals_summary(session, user_id)


@tool
def get_income_summary_cross(
    tax_year: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """Get income summary from the Income domain.

    Use this tool to view a user's income data when analyzing
    from another agent (Planning for budget, Tax for filing).

    Args:
        tax_year: Tax year (e.g., "2025"). Defaults to current year.
        user_id: UUID of the user (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict with total income and per-source breakdown.
    """
    from finance_ai.tools.cross_agent_service import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        get_income_summary,
    )

    parsed_year = int(tax_year) if tax_year else date.today().year
    with get_tool_session(db_session_factory) as session:
        return get_income_summary(session, user_id, parsed_year)


@tool
def get_tax_summary_cross(
    tax_year: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """Get tax filing summary from the Tax domain.

    Use this tool to view a user's tax filing data when analyzing
    from another agent (Investment for tax-loss harvesting, Planning).

    Args:
        tax_year: Tax year (e.g., "2025"). Defaults to current year.
        user_id: UUID of the user (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict with tax filing data or not_found status.
    """
    from finance_ai.tools.cross_agent_service import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        get_tax_filing_summary,
    )

    parsed_year = int(tax_year) if tax_year else date.today().year
    with get_tool_session(db_session_factory) as session:
        return get_tax_filing_summary(session, user_id, parsed_year)


# Pre-grouped tool lists for each agent
TAX_CROSS_TOOLS = [get_portfolio_summary_cross, get_expense_summary_cross]

PLANNING_CROSS_TOOLS = [
    get_expense_summary_cross,
    get_income_summary_cross,
    get_portfolio_summary_cross,
]

EXPENSE_CROSS_TOOLS = [get_goals_summary_cross]

ASSET_MONITORING_CROSS_TOOLS = [get_tax_summary_cross]
