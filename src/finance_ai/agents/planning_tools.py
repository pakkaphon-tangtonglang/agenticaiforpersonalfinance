"""LangGraph tool wrappers for financial goal planning.

Tools accept string inputs from LLM, parse them to proper types,
and persist records via the planning service layer. InjectedState
provides user_id and db_session_factory from the agent graph state.
"""

from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from finance_ai.agents.expense_tools import parse_date_value
from finance_ai.agents.session_helper import get_tool_session
from finance_ai.agents.tax_tools import parse_decimal_value
from finance_ai.tools.planning_constants import (
    DEFAULT_PRIORITY,
    GOAL_TYPES,
)


@tool
def create_financial_goal(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    goal_type: str,
    name: str,
    target_amount: str,
    current_amount: str = "0",
    target_date: str = "",
    priority: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """Create a new financial goal for the user.

    Use this tool when the user wants to set a savings or financial goal.
    If the user already has some money saved toward this goal, set current_amount.

    Args:
        goal_type: Goal type key (savings, emergency_fund, retirement,
            home_purchase, education, investment, debt_payoff, travel).
        name: Goal name/description (e.g., "ออมเงินฉุกเฉิน").
        target_amount: Target amount in THB (e.g., "100000").
        current_amount: Amount already saved toward this goal (default "0").
            Set this when user says "มีเงินอยู่แล้ว X บาท".
        target_date: Optional target date in YYYY-MM-DD format.
        priority: Priority level 1-5 (default 3). 1=lowest, 5=highest.
        user_id: UUID of the user (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict confirming the created goal with parsed values.
    """
    parsed_amount = parse_decimal_value(target_amount, "target_amount")
    parsed_current = (
        parse_decimal_value(current_amount, "current_amount") if current_amount else None
    )
    parsed_date = parse_date_value(target_date, "target_date") if target_date else None
    parsed_priority = int(priority) if priority else DEFAULT_PRIORITY
    goal_type_label = GOAL_TYPES.get(goal_type.strip().lower(), goal_type)

    goal = _persist_goal(
        db_session_factory,
        user_id,
        goal_type,
        name,
        parsed_amount,
        parsed_date,
        parsed_priority,
        parsed_current,
    )

    return {
        "status": "created",
        "goal_id": goal.id if goal else "",
        "goal_type": goal_type.strip().lower(),
        "goal_type_label": goal_type_label,
        "name": name,
        "target_amount": str(parsed_amount),
        "current_amount": str(parsed_current) if parsed_current is not None else "0",
        "target_date": parsed_date.isoformat() if parsed_date else None,
        "priority": parsed_priority,
    }


def _persist_goal(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    db_session_factory: Any,
    user_id: str,
    goal_type: str,
    name: str,
    target_amount: Any,
    target_date: Any,
    priority: int,
    current_amount: Any = None,
) -> Any:
    """Save goal to database via the service layer.

    Args:
        db_session_factory: Session factory or None.
        user_id: UUID of the user.
        goal_type: Goal type key.
        name: Goal name.
        target_amount: Target amount (Decimal).
        target_date: Target date or None.
        priority: Priority level.
        current_amount: Amount already saved (Decimal), or None for 0.

    Returns:
        Created FinancialGoal instance.
    """
    from decimal import Decimal  # noqa: PLC0415

    from finance_ai.tools.planning_service import create_goal  # noqa: PLC0415

    initial = current_amount if current_amount is not None else Decimal("0")
    with get_tool_session(db_session_factory) as session:
        return create_goal(
            session,
            user_id,
            goal_type,
            name,
            target_amount,
            target_date,
            priority,
            initial,
        )


@tool
def view_financial_goals(
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """View all financial goals with progress for the user.

    Use this tool when the user asks to see their financial goals or progress.

    Args:
        user_id: UUID of the user (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict with goal summary including per-goal progress.
    """
    return _fetch_goals_summary(db_session_factory, user_id)


def _fetch_goals_summary(
    db_session_factory: Any,
    user_id: str,
) -> dict[str, Any]:
    """Query goals summary from database.

    Args:
        db_session_factory: Session factory or None.
        user_id: UUID of the user.

    Returns:
        Dict with goals summary data.
    """
    from finance_ai.tools.planning_service import (  # noqa: PLC0415
        summarize_goals_for_user,
    )

    with get_tool_session(db_session_factory) as session:
        result = summarize_goals_for_user(session, user_id)

    return {
        "total_goals": result.total_goals,
        "active_goals": result.active_goals,
        "completed_goals": result.completed_goals,
        "total_target_amount": str(result.total_target_amount),
        "total_current_amount": str(result.total_current_amount),
        "overall_percentage": str(result.overall_percentage),
        "goals": [
            {
                "goal_id": g.goal_id,
                "name": g.name,
                "goal_type": g.goal_type,
                "goal_type_label": g.goal_type_label,
                "target_amount": str(g.target_amount),
                "current_amount": str(g.current_amount),
                "remaining_amount": str(g.remaining_amount),
                "percentage": str(g.percentage),
                "is_completed": g.is_completed,
                "priority_label": g.priority_label,
            }
            for g in result.goals
        ],
    }


@tool
def update_goal_progress(
    goal_id: str,
    current_amount: str,
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """Update the current saved amount for a financial goal.

    Use this tool when the user reports saving progress toward a goal.

    Args:
        goal_id: UUID of the goal to update.
        current_amount: New current amount in THB (e.g., "50000").
        user_id: UUID of the user (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict confirming the updated goal.
    """
    parsed_amount = parse_decimal_value(current_amount, "current_amount")

    goal = _update_goal(db_session_factory, goal_id, parsed_amount)

    return {
        "status": "updated",
        "goal_id": goal_id,
        "current_amount": str(parsed_amount),
        "is_completed": goal.is_completed if goal else False,
    }


def _update_goal(
    db_session_factory: Any,
    goal_id: str,
    new_amount: Any,
) -> Any:
    """Update goal progress in database.

    Args:
        db_session_factory: Session factory or None.
        goal_id: UUID of the goal.
        new_amount: New current amount (Decimal).

    Returns:
        Updated FinancialGoal instance.
    """
    from finance_ai.tools.planning_service import (  # noqa: PLC0415
        update_goal_progress as service_update,
    )

    with get_tool_session(db_session_factory) as session:
        return service_update(session, goal_id, new_amount)


@tool
def calculate_saving_plan(
    target_amount: str,
    current_amount: str = "0",
    months: str = "",
    monthly_saving: str = "",
) -> dict[str, Any]:
    """Calculate a saving plan for a financial goal.

    Use this tool when the user asks how much to save monthly,
    or how long it takes to reach a goal.

    Args:
        target_amount: Target goal amount in THB (e.g., "500000").
        current_amount: Amount already saved (default "0").
        months: Number of months to reach the goal (for monthly calculation).
        monthly_saving: Monthly saving amount (for time calculation).

    Returns:
        Dict with saving plan calculations.
    """
    from finance_ai.tools.planning_calculator import (  # noqa: PLC0415
        calculate_monthly_saving_needed,
        calculate_time_to_goal,
    )

    parsed_target = parse_decimal_value(target_amount, "target_amount")
    parsed_current = parse_decimal_value(current_amount, "current_amount")

    result: dict[str, Any] = {
        "target_amount": str(parsed_target),
        "current_amount": str(parsed_current),
        "remaining_amount": str(parsed_target - parsed_current),
    }

    if months:
        parsed_months = int(months)
        monthly = calculate_monthly_saving_needed(parsed_target, parsed_current, parsed_months)
        result["months"] = parsed_months
        result["monthly_saving_needed"] = str(monthly)

    if monthly_saving:
        parsed_monthly = parse_decimal_value(monthly_saving, "monthly_saving")
        time_needed = calculate_time_to_goal(parsed_target, parsed_current, parsed_monthly)
        result["monthly_saving"] = str(parsed_monthly)
        result["months_needed"] = time_needed

    return result
