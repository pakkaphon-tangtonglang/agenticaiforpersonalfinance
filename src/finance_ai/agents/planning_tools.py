"""LangGraph tool wrappers for financial goal planning.

Tools accept string inputs from LLM, parse them to proper types,
and persist records via the planning service layer. InjectedState
provides user_id and db_session_factory from the agent graph state.
"""

from decimal import Decimal
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
    current_amount: str = "",
    added_amount: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """Update the saved amount of a financial goal.

    Give exactly one amount:
    - current_amount: the NEW TOTAL saved, when the user states what they have now
      (e.g. "ตอนนี้เก็บได้ 50,000 แล้ว").
    - added_amount: a DEPOSIT added on top of what is saved, when the user saves more
      (e.g. "เพิ่มอีก 10,000", "ฝากเพิ่ม 5,000"). Never pass a deposit as current_amount.

    Args:
        goal_id: UUID of the goal to update.
        current_amount: New total saved in THB (e.g., "50000"), or "".
        added_amount: Amount deposited in THB (e.g., "10000"), or "".
        user_id: UUID of the user (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict confirming the goal's new saved amount.
    """
    from finance_ai.tools.planning_service import resolve_goal_amount  # noqa: PLC0415

    mode, amount = resolve_goal_amount(current_amount, added_amount)
    goal = _update_goal(db_session_factory, goal_id, amount, mode)
    return {
        "status": "updated",
        "goal_id": goal_id,
        "current_amount": str(goal.current_amount),
        "is_completed": goal.is_completed,
    }


def _update_goal(
    db_session_factory: Any,
    goal_id: str,
    amount: Decimal,
    mode: str,
) -> Any:
    """Set or add to a goal's saved amount in the database.

    Args:
        db_session_factory: Session factory or None.
        goal_id: UUID of the goal.
        amount: New total ("set") or deposit ("add").
        mode: "set" or "add".

    Returns:
        Updated FinancialGoal instance.
    """
    from finance_ai.tools.planning_service import (  # noqa: PLC0415
        add_to_goal_progress,
        update_goal_progress as service_update,
    )

    service = add_to_goal_progress if mode == "add" else service_update
    with get_tool_session(db_session_factory) as session:
        return service(session, goal_id, amount)


def _add_inflation_adjusted_plan(
    result: dict[str, Any],
    target: Decimal,
    current: Decimal,
    months: int,
    annual_inflation_percent: str,
) -> None:
    """Add the inflation-adjusted target and monthly saving to a plan result."""
    from finance_ai.tools.planning_calculator import (  # noqa: PLC0415
        calculate_monthly_saving_needed,
        inflate_target,
    )
    from finance_ai.tools.planning_constants import (  # noqa: PLC0415
        DEFAULT_ANNUAL_INFLATION,
    )

    inflation = (
        parse_decimal_value(annual_inflation_percent, "annual_inflation_percent") / Decimal("100")
        if annual_inflation_percent
        else DEFAULT_ANNUAL_INFLATION
    )
    future_target = inflate_target(target, months, inflation)
    result["annual_inflation_rate"] = format(inflation.normalize(), "f")
    result["target_amount_inflation_adjusted"] = str(future_target)
    result["monthly_saving_needed_inflation_adjusted"] = str(
        calculate_monthly_saving_needed(future_target, current, months)
    )


@tool
def calculate_saving_plan(
    target_amount: str,
    current_amount: str = "0",
    months: str = "",
    monthly_saving: str = "",
    annual_inflation_percent: str = "",
) -> dict[str, Any]:
    """Calculate a saving plan for a financial goal.

    Use this tool when the user asks how much to save monthly,
    or how long it takes to reach a goal. With a deadline (months), the
    result also shows the target and monthly saving adjusted for inflation,
    because the goal will cost more by then. Present both figures.

    Args:
        target_amount: Target goal amount in today's THB (e.g., "500000").
        current_amount: Amount already saved (default "0").
        months: Number of months to reach the goal (for monthly calculation).
        monthly_saving: Monthly saving amount (for time calculation).
        annual_inflation_percent: Yearly inflation in percent (default 2,
            the midpoint of the Bank of Thailand's 1-3% target).

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
        _add_inflation_adjusted_plan(
            result, parsed_target, parsed_current, parsed_months, annual_inflation_percent
        )

    if monthly_saving:
        parsed_monthly = parse_decimal_value(monthly_saving, "monthly_saving")
        time_needed = calculate_time_to_goal(parsed_target, parsed_current, parsed_monthly)
        result["monthly_saving"] = str(parsed_monthly)
        result["months_needed"] = time_needed

    return result


@tool
def plan_pay_yourself_first(
    monthly_income: str,
    savings_rate_percent: str = "",
) -> dict[str, Any]:
    """Split monthly income into "save first" and "free to spend".

    Use this tool when the user asks how to start saving, how much to save
    each month, or for a simple budget. Saving a fixed share on payday works
    better than saving what is left at month end (pay yourself first).

    Args:
        monthly_income: Monthly income in THB (e.g., "30000").
        savings_rate_percent: Share to save first in percent (default 20).

    Returns:
        Dict with save_first and spendable amounts, or an 'error' message.
    """
    from finance_ai.tools.planning_calculator import (  # noqa: PLC0415
        calculate_pay_yourself_first,
    )
    from finance_ai.tools.planning_constants import (  # noqa: PLC0415
        DEFAULT_PAY_YOURSELF_FIRST_RATE,
    )

    try:
        income = parse_decimal_value(monthly_income, "monthly_income")
        rate = (
            parse_decimal_value(savings_rate_percent, "savings_rate_percent") / Decimal("100")
            if savings_rate_percent
            else DEFAULT_PAY_YOURSELF_FIRST_RATE
        )
        plan = calculate_pay_yourself_first(income, rate)
    except ValueError as error:
        return {"error": str(error)}
    return {
        "monthly_income": str(plan.monthly_income),
        "savings_rate_percent": format((plan.savings_rate * 100).normalize(), "f"),
        "save_first": str(plan.save_first),
        "spendable": str(plan.spendable),
    }
