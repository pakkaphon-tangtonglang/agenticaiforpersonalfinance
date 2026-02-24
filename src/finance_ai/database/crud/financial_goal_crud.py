"""CRUD operations for the FinancialGoal model."""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.database.crud.base_crud import BaseCRUD
from finance_ai.database.models.financial_goal import FinancialGoal


class FinancialGoalCRUD(BaseCRUD[FinancialGoal]):
    """
    CRUD operations specific to FinancialGoal model.

    Example:
        >>> goal_crud = FinancialGoalCRUD()
        >>> goals = goal_crud.get_by_user(session, user_id)
    """

    def __init__(self) -> None:
        """Initialize FinancialGoalCRUD with FinancialGoal model."""
        super().__init__(FinancialGoal)

    def get_by_user(self, session: Session, user_id: str) -> list[FinancialGoal]:
        """
        Get all financial goals for a user.

        Args:
            session: Database session.
            user_id: UUID of the user.

        Returns:
            List of FinancialGoal instances.
        """
        statement = select(FinancialGoal).where(FinancialGoal.user_id == user_id)
        return list(session.execute(statement).scalars().all())

    def get_active_goals(self, session: Session, user_id: str) -> list[FinancialGoal]:
        """
        Get all incomplete financial goals for a user.

        Args:
            session: Database session.
            user_id: UUID of the user.

        Returns:
            List of active (not completed) FinancialGoal instances.
        """
        statement = select(FinancialGoal).where(
            FinancialGoal.user_id == user_id,
            FinancialGoal.is_completed.is_(False),
        )
        return list(session.execute(statement).scalars().all())

    def mark_completed(self, session: Session, goal_id: str) -> Optional[FinancialGoal]:
        """
        Mark a financial goal as completed.

        Args:
            session: Database session.
            goal_id: UUID of the goal.

        Returns:
            Updated FinancialGoal or None if not found.
        """
        goal = self.get_by_id(session, goal_id)
        if goal is None:
            return None
        goal.is_completed = True
        session.commit()
        session.refresh(goal)
        return goal
