"""Tests for the FinancialGoal database model."""

from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from finance_ai.database.models.financial_goal import FinancialGoal
from finance_ai.database.models.user import User


class TestFinancialGoalModel:
    """Tests for FinancialGoal model creation, defaults, and relationships."""

    def test_create_financial_goal(self, test_session: Session, sample_user: User) -> None:
        """Test creating a financial goal with all fields."""
        goal = FinancialGoal(
            user_id=sample_user.id,
            goal_type="retirement",
            name="Retirement Fund",
            target_amount=Decimal("10000000.00"),
            current_amount=Decimal("500000.00"),
            target_date=date(2050, 12, 31),
            priority=1,
        )
        test_session.add(goal)
        test_session.commit()
        test_session.refresh(goal)
        assert goal.id is not None
        assert goal.target_amount == Decimal("10000000.00")
        assert goal.goal_type == "retirement"

    def test_financial_goal_defaults(self, test_session: Session, sample_user: User) -> None:
        """Test that defaults are correctly applied."""
        goal = FinancialGoal(
            user_id=sample_user.id,
            goal_type="emergency",
            name="Emergency Fund",
            target_amount=Decimal("300000.00"),
        )
        test_session.add(goal)
        test_session.commit()
        test_session.refresh(goal)
        assert goal.current_amount == Decimal("0")
        assert goal.priority == 1
        assert goal.is_completed is False
        assert goal.target_date is None

    def test_financial_goal_relationship_to_user(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Test that goal links back to its user."""
        goal = FinancialGoal(
            user_id=sample_user.id,
            goal_type="house",
            name="Buy a House",
            target_amount=Decimal("5000000.00"),
        )
        test_session.add(goal)
        test_session.commit()
        test_session.refresh(goal)
        assert goal.user.id == sample_user.id
        assert goal in sample_user.financial_goals

    def test_multiple_goals_for_user(self, test_session: Session, sample_user: User) -> None:
        """Test creating multiple goals for one user."""
        goal_data = [
            ("retirement", "Retire Early", Decimal("10000000.00")),
            ("education", "MBA Degree", Decimal("1000000.00")),
            ("car", "New Car", Decimal("800000.00")),
        ]
        for goal_type, name, amount in goal_data:
            goal = FinancialGoal(
                user_id=sample_user.id,
                goal_type=goal_type,
                name=name,
                target_amount=amount,
            )
            test_session.add(goal)
        test_session.commit()
        assert len(sample_user.financial_goals) == 3
