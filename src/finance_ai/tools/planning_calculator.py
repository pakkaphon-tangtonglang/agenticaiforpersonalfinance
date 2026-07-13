"""Pure planning calculation functions for financial goal analysis.

All functions accept typed values directly with no database dependency,
making them easy to test and reuse across different contexts.
"""

import math
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field

from finance_ai.tools.planning_constants import (
    GOAL_TYPES,
    PRIORITY_LEVELS,
    validate_goal_amount,
)


class GoalRecord(BaseModel):
    """A single financial goal record for calculation purposes.

    Attributes:
        goal_id: Unique identifier of the goal.
        goal_type: Goal type key.
        name: Goal name/description.
        target_amount: Target amount in THB.
        current_amount: Current saved amount in THB.
        target_date: Optional target completion date.
        priority: Priority level (1-5).
        is_completed: Whether the goal is completed.
    """

    goal_id: str
    goal_type: str
    name: str
    target_amount: Decimal
    current_amount: Decimal = Decimal("0")
    target_date: date | None = None
    priority: int = 3
    is_completed: bool = False


class GoalProgress(BaseModel):
    """Progress information for a single financial goal.

    Attributes:
        goal_id: Unique identifier of the goal.
        name: Goal name.
        goal_type: Goal type key.
        goal_type_label: Thai display label.
        target_amount: Target amount in THB.
        current_amount: Current saved amount in THB.
        remaining_amount: Amount remaining to reach target.
        percentage: Progress percentage (0-100).
        is_completed: Whether the goal is completed.
        is_on_track: Whether the goal is on track (if target_date set).
        priority_label: Thai label for priority level.
    """

    goal_id: str
    name: str
    goal_type: str
    goal_type_label: str
    target_amount: Decimal
    current_amount: Decimal
    remaining_amount: Decimal
    percentage: Decimal = Field(default=Decimal("0.00"))
    is_completed: bool = False
    is_on_track: bool | None = None
    priority_label: str = ""


class GoalSummaryResult(BaseModel):
    """Summary of all financial goals for a user.

    Attributes:
        total_goals: Total number of goals.
        active_goals: Number of active (incomplete) goals.
        completed_goals: Number of completed goals.
        total_target_amount: Sum of all target amounts.
        total_current_amount: Sum of all current amounts.
        overall_percentage: Overall progress percentage.
        goals: List of individual goal progress records.
    """

    total_goals: int
    active_goals: int
    completed_goals: int
    total_target_amount: Decimal
    total_current_amount: Decimal
    overall_percentage: Decimal = Field(default=Decimal("0.00"))
    goals: list[GoalProgress]


def calculate_goal_progress(
    goal: GoalRecord,
) -> GoalProgress:
    """Calculate progress for a single financial goal.

    Args:
        goal: Goal record to calculate progress for.

    Returns:
        GoalProgress with percentage, remaining amount, and status.

    Example:
        >>> goal = GoalRecord(
        ...     goal_id="1", goal_type="savings", name="ออมเงิน",
        ...     target_amount=Decimal("100000"), current_amount=Decimal("40000"),
        ... )
        >>> progress = calculate_goal_progress(goal)
        >>> progress.percentage
        Decimal('40.00')
    """
    remaining = max(goal.target_amount - goal.current_amount, Decimal("0"))

    percentage = _calculate_percentage(goal.current_amount, goal.target_amount)
    goal_type_label = GOAL_TYPES.get(goal.goal_type, goal.goal_type)
    priority_label = PRIORITY_LEVELS.get(goal.priority, "")

    return GoalProgress(
        goal_id=goal.goal_id,
        name=goal.name,
        goal_type=goal.goal_type,
        goal_type_label=goal_type_label,
        target_amount=goal.target_amount,
        current_amount=goal.current_amount,
        remaining_amount=remaining,
        percentage=percentage,
        is_completed=goal.is_completed,
        priority_label=priority_label,
    )


def _calculate_percentage(
    current: Decimal,
    target: Decimal,
) -> Decimal:
    """Calculate percentage of current vs target.

    Args:
        current: Current amount.
        target: Target amount.

    Returns:
        Percentage with 2 decimal places, capped at 100.00.

    Example:
        >>> _calculate_percentage(Decimal("50"), Decimal("200"))
        Decimal('25.00')
    """
    if target <= Decimal("0"):
        return Decimal("0.00")
    raw = (current / target * Decimal("100")).quantize(Decimal("0.01"))
    if raw > Decimal("100.00"):
        return Decimal("100.00")
    return raw


def calculate_monthly_saving_needed(
    target_amount: Decimal,
    current_amount: Decimal,
    months_remaining: int,
) -> Decimal:
    """Calculate monthly saving needed to reach a goal.

    Args:
        target_amount: Target amount in THB.
        current_amount: Amount already saved.
        months_remaining: Number of months until target date.

    Returns:
        Monthly saving amount (2 decimal places).

    Raises:
        ValueError: If months_remaining is not positive.

    Example:
        >>> calculate_monthly_saving_needed(
        ...     Decimal("100000"), Decimal("40000"), 12
        ... )
        Decimal('5000.00')
    """
    validate_goal_amount(target_amount)

    if months_remaining <= 0:
        raise ValueError(f"Months remaining must be positive. Received: {months_remaining}")

    remaining = target_amount - current_amount
    if remaining <= Decimal("0"):
        return Decimal("0.00")

    return (remaining / Decimal(str(months_remaining))).quantize(Decimal("0.01"))


def calculate_time_to_goal(
    target_amount: Decimal,
    current_amount: Decimal,
    monthly_saving: Decimal,
) -> int:
    """Calculate months needed to reach a goal at given saving rate.

    Args:
        target_amount: Target amount in THB.
        current_amount: Amount already saved.
        monthly_saving: Monthly saving amount in THB.

    Returns:
        Number of months needed (rounded up).

    Raises:
        ValueError: If monthly_saving is not positive.

    Example:
        >>> calculate_time_to_goal(
        ...     Decimal("100000"), Decimal("40000"), Decimal("5000")
        ... )
        12
    """
    remaining = target_amount - current_amount
    if remaining <= Decimal("0"):
        return 0

    if monthly_saving <= Decimal("0"):
        raise ValueError(f"Monthly saving must be positive. Received: {monthly_saving} THB")

    return math.ceil(remaining / monthly_saving)


def summarize_goals(
    goals: list[GoalRecord],
) -> GoalSummaryResult:
    """Build a complete summary of all financial goals.

    Args:
        goals: List of goal records to summarize.

    Returns:
        GoalSummaryResult with totals and per-goal progress.

    Example:
        >>> summarize_goals([])
        GoalSummaryResult(total_goals=0, ...)
    """
    progress_list = [calculate_goal_progress(goal) for goal in goals]
    active = [g for g in goals if not g.is_completed]
    completed = [g for g in goals if g.is_completed]

    total_target = sum((g.target_amount for g in goals), Decimal("0"))
    total_current = sum((g.current_amount for g in goals), Decimal("0"))
    overall_pct = _calculate_percentage(total_current, total_target)

    return GoalSummaryResult(
        total_goals=len(goals),
        active_goals=len(active),
        completed_goals=len(completed),
        total_target_amount=total_target,
        total_current_amount=total_current,
        overall_percentage=overall_pct,
        goals=progress_list,
    )
