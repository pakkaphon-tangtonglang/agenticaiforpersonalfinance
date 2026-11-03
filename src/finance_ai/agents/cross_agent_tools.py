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
def update_savings_goal_cross(
    search_term: str,
    current_amount: str = "",
    added_amount: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """Update the saved amount of the best-matching active financial goal.

    Give exactly one amount:
    - current_amount: the NEW TOTAL saved, when the user states what they have now
      (e.g. "มีเงินออม 200,000 บาท" after creating a car goal).
    - added_amount: a DEPOSIT added on top of what is saved, when the user saves more
      (e.g. "เพิ่มเงินเข้ากองทุนฉุกเฉินอีก 10,000"). Never pass a deposit as current_amount.

    Args:
        search_term: Keyword to find the goal (e.g., "รถ", "บ้าน", "ฉุกเฉิน").
                     Pass empty string to update the most recently created goal.
        current_amount: New total saved in THB (e.g., "200000"), or "".
        added_amount: Amount deposited in THB (e.g., "10000"), or "".
        user_id: UUID of the user (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict with the updated goal, or an error / no_goals status.
    """
    from finance_ai.tools.planning_service import (  # noqa: PLC0415
        get_active_goals,
        resolve_goal_amount,
    )

    try:
        mode, amount = resolve_goal_amount(current_amount, added_amount)
    except ValueError as error:
        return {"status": "error", "message": str(error)}
    with get_tool_session(db_session_factory) as session:
        goals = get_active_goals(session, user_id)
        if not goals:
            return {"status": "no_goals", "message": "ไม่พบเป้าหมายที่ยังไม่เสร็จ"}
        updated = _apply_goal_update(
            session, _find_best_goal_match(goals, search_term), mode, amount
        )
    return {
        "status": "updated",
        "goal_id": str(updated.id),
        "goal_name": updated.name,
        "goal_type": updated.goal_type,
        "current_amount": str(updated.current_amount),
        "target_amount": str(updated.target_amount),
        "is_completed": updated.is_completed,
    }


def _apply_goal_update(session: Any, goal: Any, mode: str, amount: Any) -> Any:
    """Set a goal's saved total or add a deposit to it.

    Args:
        session: Database session.
        goal: Goal to update.
        mode: "set" or "add".
        amount: New total or deposit (Decimal).

    Returns:
        Updated goal.
    """
    from finance_ai.tools.planning_service import (  # noqa: PLC0415
        add_to_goal_progress,
        update_goal_progress,
    )

    service = add_to_goal_progress if mode == "add" else update_goal_progress
    return service(session, goal.id, amount)


def _find_best_goal_match(goals: list[Any], search_term: str) -> Any:
    """Find the most relevant goal by keyword match.

    Falls back to the first active goal when no keyword matches.

    Args:
        goals: List of FinancialGoal instances.
        search_term: Keyword hint (may be empty).

    Returns:
        Best matching FinancialGoal.
    """
    if search_term:
        term = search_term.lower()
        for goal in goals:
            if term in goal.name.lower() or goal.name.lower() in term:
                return goal
    return goals[0]


# Pre-grouped tool lists for each agent
TAX_CROSS_TOOLS = [get_expense_summary_cross]

PLANNING_CROSS_TOOLS = [
    get_expense_summary_cross,
    get_income_summary_cross,
]

EXPENSE_CROSS_TOOLS = [get_goals_summary_cross, update_savings_goal_cross]

ASSET_MONITORING_CROSS_TOOLS: list[Any] = []
