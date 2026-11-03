"""Dashboard add / edit / delete for financial goals (the chat uses planning_service)."""

from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.database.models.financial_goal import FinancialGoal
from finance_ai.tools.planning_constants import DEFAULT_PRIORITY, GOAL_TYPES, MAX_GOAL_AMOUNT

CENT = Decimal("0.01")


class GoalNotFoundError(LookupError):
    """The goal does not exist or belongs to another user."""


class GoalInput(BaseModel):
    """A new or edited goal from the dashboard form."""

    name: str = Field(min_length=1, max_length=255)
    goal_type: str
    target_amount: Decimal = Field(gt=0, le=MAX_GOAL_AMOUNT)
    current_amount: Decimal = Field(default=Decimal("0"), ge=0, le=MAX_GOAL_AMOUNT)
    target_date: Optional[date] = None

    @field_validator("name")
    @classmethod
    def _trimmed(cls, value: str) -> str:
        """No blank goal name."""
        if not value.strip():
            raise ValueError("Goal name is required (e.g. 'เงินสำรองฉุกเฉิน').")
        return value.strip()

    @field_validator("target_amount", "current_amount")
    @classmethod
    def _to_satang(cls, value: Decimal) -> Decimal:
        """Store money with 2 decimal places."""
        return value.quantize(CENT, rounding=ROUND_HALF_UP)

    @field_validator("goal_type")
    @classmethod
    def _known_type(cls, value: str) -> str:
        """Only the types the form offers."""
        if value not in GOAL_TYPES:
            raise ValueError(f"Unknown goal type '{value}'. Allowed: {', '.join(GOAL_TYPES)}")
        return value


class GoalEntry(GoalInput):
    """A saved goal as the dashboard shows it."""

    goal_id: str
    goal_type_label: str
    percentage: Decimal
    is_completed: bool


def _to_entry(goal: FinancialGoal) -> GoalEntry:
    """ORM row → dashboard entry with the saved percentage (capped at 100)."""
    percentage = min(goal.current_amount / goal.target_amount * 100, Decimal("100"))
    return GoalEntry(
        goal_id=goal.id, name=goal.name, goal_type=goal.goal_type,
        goal_type_label=GOAL_TYPES.get(goal.goal_type, goal.goal_type),
        target_amount=goal.target_amount, current_amount=goal.current_amount,
        target_date=goal.target_date, percentage=percentage.quantize(CENT),
        is_completed=goal.is_completed,
    )  # fmt: skip


def list_goal_entries(session: Session, user_id: str) -> list[GoalEntry]:
    """All the user's goals, unfinished first, then by creation.

    Example:
        >>> [goal.name for goal in list_goal_entries(session, uid)]
        ['เงินสำรองฉุกเฉิน']
    """
    statement = (
        select(FinancialGoal)
        .where(FinancialGoal.user_id == user_id)
        .order_by(FinancialGoal.is_completed, FinancialGoal.created_at)
    )
    return [_to_entry(goal) for goal in session.execute(statement).scalars()]


def _owned_goal(session: Session, user_id: str, goal_id: str) -> FinancialGoal:
    """The user's goal, or GoalNotFoundError."""
    goal = session.get(FinancialGoal, goal_id)
    if goal is None or goal.user_id != user_id:
        raise GoalNotFoundError(f"Goal not found: {goal_id}")
    return goal


def _apply(goal: FinancialGoal, data: GoalInput) -> None:
    """Copy the form onto the row; reaching the target completes it."""
    for field, value in data.model_dump().items():
        setattr(goal, field, value)
    goal.is_completed = data.current_amount >= data.target_amount


def create_goal_entry(session: Session, user_id: str, data: GoalInput) -> GoalEntry:
    """Save a goal typed into the dashboard (committed).

    Example:
        >>> create_goal_entry(session, uid, form).goal_type_label
        'เงินสำรองฉุกเฉิน'
    """
    goal = FinancialGoal(user_id=user_id, priority=DEFAULT_PRIORITY)
    _apply(goal, data)
    session.add(goal)
    session.commit()
    return _to_entry(goal)


def update_goal_entry(session: Session, user_id: str, goal_id: str, data: GoalInput) -> GoalEntry:
    """Replace a goal's values (committed).

    Raises:
        GoalNotFoundError: If the goal is missing or not the user's.
    """
    goal = _owned_goal(session, user_id, goal_id)
    _apply(goal, data)
    session.commit()
    return _to_entry(goal)


def delete_goal_entry(session: Session, user_id: str, goal_id: str) -> None:
    """Delete a goal (committed).

    Raises:
        GoalNotFoundError: If the goal is missing or not the user's.
    """
    session.delete(_owned_goal(session, user_id, goal_id))
    session.commit()
