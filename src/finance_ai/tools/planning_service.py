"""Database-integrated financial goal planning service.

Orchestrates goal operations by delegating to FinancialGoalCRUD
and converting ORM models to calculator-compatible records.
"""

from datetime import date
from decimal import Decimal
from typing import Optional

from sqlalchemy.orm import Session

from finance_ai.database.crud.financial_goal_crud import FinancialGoalCRUD
from finance_ai.database.models.financial_goal import FinancialGoal
from finance_ai.tools.planning_calculator import (
    GoalRecord,
    GoalSummaryResult,
    summarize_goals,
)
from finance_ai.tools.planning_constants import (
    DEFAULT_PRIORITY,
    validate_goal_amount,
    validate_goal_type,
    validate_priority,
)


def convert_goal_to_record(goal: FinancialGoal) -> GoalRecord:
    """Convert a FinancialGoal ORM model to a GoalRecord.

    Args:
        goal: FinancialGoal model instance.

    Returns:
        GoalRecord suitable for pure calculator functions.

    Example:
        >>> record = convert_goal_to_record(goal)
        >>> record.goal_type
        'savings'
    """
    return GoalRecord(
        goal_id=goal.id,
        goal_type=goal.goal_type,
        name=goal.name,
        target_amount=goal.target_amount,
        current_amount=goal.current_amount,
        target_date=goal.target_date,
        priority=goal.priority,
        is_completed=goal.is_completed,
    )


def create_goal(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    session: Session,
    user_id: str,
    goal_type: str,
    name: str,
    target_amount: Decimal,
    target_date: Optional[date] = None,
    priority: int = DEFAULT_PRIORITY,
    current_amount: Decimal = Decimal("0"),
) -> FinancialGoal:
    """Validate inputs and create a new financial goal.

    Args:
        session: Database session.
        user_id: UUID of the user.
        goal_type: Goal type key.
        name: Goal name/description.
        target_amount: Target amount in THB.
        target_date: Optional target completion date.
        priority: Priority level (1-5).
        current_amount: Amount already saved (default 0).

    Returns:
        Created FinancialGoal instance.

    Raises:
        ValueError: If goal_type, amount, or priority is invalid.

    Example:
        >>> goal = create_goal(
        ...     session, "user-1", "savings", "เงินฉุกเฉิน",
        ...     Decimal("100000"), date(2027, 12, 31), 4, Decimal("50000"),
        ... )
    """
    normalized_type = validate_goal_type(goal_type)
    validate_goal_amount(target_amount)
    validate_priority(priority)
    safe_current = max(current_amount, Decimal("0"))
    is_completed = safe_current >= target_amount

    crud = FinancialGoalCRUD()
    goal = crud.create(
        session,
        user_id=user_id,
        goal_type=normalized_type,
        name=name,
        target_amount=target_amount,
        current_amount=safe_current,
        target_date=target_date,
        priority=priority,
        is_completed=is_completed,
    )
    session.commit()
    session.refresh(goal)
    return goal


def get_user_goals(
    session: Session,
    user_id: str,
) -> list[FinancialGoal]:
    """Get all financial goals for a user.

    Args:
        session: Database session.
        user_id: UUID of the user.

    Returns:
        List of FinancialGoal instances.

    Example:
        >>> goals = get_user_goals(session, "user-1")
    """
    crud = FinancialGoalCRUD()
    return crud.get_by_user(session, user_id)


def get_active_goals(
    session: Session,
    user_id: str,
) -> list[FinancialGoal]:
    """Get all active (incomplete) financial goals for a user.

    Args:
        session: Database session.
        user_id: UUID of the user.

    Returns:
        List of active FinancialGoal instances.

    Example:
        >>> goals = get_active_goals(session, "user-1")
    """
    crud = FinancialGoalCRUD()
    return crud.get_active_goals(session, user_id)


def update_goal_progress(
    session: Session,
    goal_id: str,
    new_amount: Decimal,
) -> FinancialGoal:
    """Update the current amount of a financial goal.

    Args:
        session: Database session.
        goal_id: UUID of the goal.
        new_amount: New current amount in THB.

    Returns:
        Updated FinancialGoal instance.

    Raises:
        ValueError: If goal not found or amount is negative.

    Example:
        >>> goal = update_goal_progress(session, "goal-1", Decimal("50000"))
    """
    if new_amount < Decimal("0"):
        raise ValueError(f"Current amount cannot be negative. Received: {new_amount} THB")

    crud = FinancialGoalCRUD()
    goal = crud.get_by_id(session, goal_id)
    if goal is None:
        raise ValueError(f"Goal not found: {goal_id}")

    goal.current_amount = new_amount
    if new_amount >= goal.target_amount:
        goal.is_completed = True
    session.commit()
    session.refresh(goal)
    return goal


def add_to_goal_progress(
    session: Session,
    goal_id: str,
    added_amount: Decimal,
) -> FinancialGoal:
    """Add a deposit to a goal's saved amount (instead of replacing it).

    Args:
        session: Database session.
        goal_id: UUID of the goal.
        added_amount: Amount deposited, in THB (must be positive).

    Returns:
        Updated FinancialGoal instance.

    Raises:
        ValueError: If the deposit is not positive or the goal is not found.

    Example:
        >>> goal = add_to_goal_progress(session, "goal-1", Decimal("10000"))
    """
    if added_amount <= Decimal("0"):
        raise ValueError(f"Deposit must be positive. Received: {added_amount} THB")
    goal = FinancialGoalCRUD().get_by_id(session, goal_id)
    if goal is None:
        raise ValueError(f"Goal not found: {goal_id}")
    return update_goal_progress(session, goal_id, goal.current_amount + added_amount)


def resolve_goal_amount(current_amount: str, added_amount: str) -> tuple[str, Decimal]:
    """Decide whether a goal update sets a new total or adds a deposit.

    Args:
        current_amount: New total saved ("มีเงินออม 120,000"), or "".
        added_amount: Amount deposited ("เพิ่มอีก 10,000"), or "".

    Returns:
        ("set", total) or ("add", deposit).

    Raises:
        ValueError: Unless exactly one of the two is given, or if it is not a number.

    Example:
        >>> resolve_goal_amount("", "10,000")
        ('add', Decimal('10000'))
    """
    total, deposit = current_amount.strip(), added_amount.strip()
    if bool(total) == bool(deposit):
        raise ValueError(
            "Give exactly one of current_amount (new total saved) or added_amount (deposit). "
            f"Received: current_amount={current_amount!r}, added_amount={added_amount!r}"
        )
    mode, raw = ("set", total) if total else ("add", deposit)
    try:
        return mode, Decimal(raw.replace(",", ""))
    except ArithmeticError as error:
        raise ValueError(f"Amount must be a number. Received: {raw!r}") from error


def mark_goal_completed(
    session: Session,
    goal_id: str,
) -> FinancialGoal:
    """Mark a financial goal as completed.

    Args:
        session: Database session.
        goal_id: UUID of the goal.

    Returns:
        Updated FinancialGoal instance.

    Raises:
        ValueError: If goal not found.

    Example:
        >>> goal = mark_goal_completed(session, "goal-1")
    """
    crud = FinancialGoalCRUD()
    goal = crud.mark_completed(session, goal_id)
    if goal is None:
        raise ValueError(f"Goal not found: {goal_id}")
    return goal


def delete_goal(
    session: Session,
    goal_id: str,
) -> bool:
    """Delete a financial goal.

    Args:
        session: Database session.
        goal_id: UUID of the goal.

    Returns:
        True if goal was deleted, False if not found.

    Example:
        >>> delete_goal(session, "goal-1")
        True
    """
    crud = FinancialGoalCRUD()
    deleted = crud.delete(session, goal_id)
    session.commit()
    return deleted


def summarize_goals_for_user(
    session: Session,
    user_id: str,
) -> GoalSummaryResult:
    """Summarize all financial goals for a user.

    Queries DB, converts to GoalRecord list, then delegates
    to the pure summarize_goals function.

    Args:
        session: Database session.
        user_id: UUID of the user.

    Returns:
        GoalSummaryResult with per-goal progress.

    Example:
        >>> summary = summarize_goals_for_user(session, "user-1")
        >>> summary.total_goals
        3
    """
    goals = get_user_goals(session, user_id)
    records = [convert_goal_to_record(g) for g in goals]
    return summarize_goals(records)
