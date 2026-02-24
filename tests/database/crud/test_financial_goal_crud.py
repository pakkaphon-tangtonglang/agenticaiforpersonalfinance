"""Tests for FinancialGoalCRUD operations."""

from decimal import Decimal

from sqlalchemy.orm import Session

from finance_ai.database.crud.financial_goal_crud import FinancialGoalCRUD
from finance_ai.database.models.user import User


class TestFinancialGoalCRUD:
    """Tests for FinancialGoal-specific CRUD operations."""

    def test_get_by_user(self, test_session: Session, sample_user: User) -> None:
        """Test getting all goals for a user."""
        crud = FinancialGoalCRUD()
        crud.create(
            test_session,
            user_id=sample_user.id,
            goal_type="retirement",
            name="Retire Early",
            target_amount=Decimal("10000000.00"),
        )
        crud.create(
            test_session,
            user_id=sample_user.id,
            goal_type="house",
            name="Buy House",
            target_amount=Decimal("5000000.00"),
        )
        goals = crud.get_by_user(test_session, sample_user.id)
        assert len(goals) == 2

    def test_get_active_goals(self, test_session: Session, sample_user: User) -> None:
        """Test filtering out completed goals."""
        crud = FinancialGoalCRUD()
        crud.create(
            test_session,
            user_id=sample_user.id,
            goal_type="emergency",
            name="Emergency Fund",
            target_amount=Decimal("300000.00"),
        )
        crud.create(
            test_session,
            user_id=sample_user.id,
            goal_type="car",
            name="Old Car",
            target_amount=Decimal("800000.00"),
            is_completed=True,
        )
        active = crud.get_active_goals(test_session, sample_user.id)
        assert len(active) == 1
        assert active[0].name == "Emergency Fund"

    def test_mark_completed(self, test_session: Session, sample_user: User) -> None:
        """Test marking a goal as completed."""
        crud = FinancialGoalCRUD()
        goal = crud.create(
            test_session,
            user_id=sample_user.id,
            goal_type="education",
            name="MBA",
            target_amount=Decimal("1000000.00"),
        )
        result = crud.mark_completed(test_session, goal.id)
        assert result is not None
        assert result.is_completed is True

    def test_mark_completed_not_found(self, test_session: Session) -> None:
        """Test that marking nonexistent goal returns None."""
        crud = FinancialGoalCRUD()
        result = crud.mark_completed(test_session, "fake-id")
        assert result is None
