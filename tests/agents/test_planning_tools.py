"""Tests for LangGraph planning tool wrappers."""

from collections.abc import Callable
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session

from finance_ai.agents.planning_tools import (
    calculate_saving_plan,
    create_financial_goal,
    update_goal_progress,
    view_financial_goals,
)
from finance_ai.database.models.user import User


class TestCreateFinancialGoal:
    """Tests for the create_financial_goal LangGraph tool."""

    def test_creates_goal_in_database(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Creates a goal record in the database."""
        result = create_financial_goal.invoke(
            {
                "goal_type": "savings",
                "name": "เงินฉุกเฉิน",
                "target_amount": "100000",
                "target_date": "2027-12-31",
                "priority": "4",
            },
            config={
                "configurable": {
                    "user_id": sample_user.id,
                    "db_session_factory": db_session_factory,
                }
            },
        )
        assert result["status"] == "created"
        assert result["goal_type"] == "savings"
        assert result["name"] == "เงินฉุกเฉิน"
        assert result["target_amount"] == "100000"

    def test_default_priority(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Uses default priority when not specified."""
        result = create_financial_goal.invoke(
            {
                "goal_type": "retirement",
                "name": "เกษียณ",
                "target_amount": "5000000",
            },
            config={
                "configurable": {
                    "user_id": sample_user.id,
                    "db_session_factory": db_session_factory,
                }
            },
        )
        assert result["priority"] == 3

    def test_invalid_goal_type_raises(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Raises error for invalid goal type."""
        with pytest.raises(Exception, match="Unknown goal type"):
            create_financial_goal.invoke(
                {
                    "goal_type": "crypto",
                    "name": "Test",
                    "target_amount": "100000",
                },
                config={
                    "configurable": {
                        "user_id": sample_user.id,
                        "db_session_factory": db_session_factory,
                    }
                },
            )


class TestViewFinancialGoals:
    """Tests for the view_financial_goals LangGraph tool."""

    def test_shows_created_goals(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Shows goals that were created via the tool."""
        create_financial_goal.invoke(
            {
                "goal_type": "savings",
                "name": "เงินฉุกเฉิน",
                "target_amount": "100000",
                "priority": "4",
            },
            config={
                "configurable": {
                    "user_id": sample_user.id,
                    "db_session_factory": db_session_factory,
                }
            },
        )
        result = view_financial_goals.invoke(
            {},
            config={
                "configurable": {
                    "user_id": sample_user.id,
                    "db_session_factory": db_session_factory,
                }
            },
        )
        assert result["total_goals"] >= 1
        names = [g["name"] for g in result["goals"]]
        assert "เงินฉุกเฉิน" in names


class TestUpdateGoalProgress:
    """Tests for the update_goal_progress LangGraph tool."""

    def test_updates_amount(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Updates the current amount of a goal."""
        created = create_financial_goal.invoke(
            {
                "goal_type": "savings",
                "name": "TestUpdate",
                "target_amount": "100000",
            },
            config={
                "configurable": {
                    "user_id": sample_user.id,
                    "db_session_factory": db_session_factory,
                }
            },
        )
        result = update_goal_progress.invoke(
            {
                "goal_id": created["goal_id"],
                "current_amount": "50000",
            },
            config={
                "configurable": {
                    "user_id": sample_user.id,
                    "db_session_factory": db_session_factory,
                }
            },
        )
        assert result["status"] == "updated"
        assert result["current_amount"] == "50000"

    def test_auto_completes(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Marks goal as completed when target is reached."""
        created = create_financial_goal.invoke(
            {
                "goal_type": "savings",
                "name": "TestComplete",
                "target_amount": "100000",
            },
            config={
                "configurable": {
                    "user_id": sample_user.id,
                    "db_session_factory": db_session_factory,
                }
            },
        )
        result = update_goal_progress.invoke(
            {
                "goal_id": created["goal_id"],
                "current_amount": "100000",
            },
            config={
                "configurable": {
                    "user_id": sample_user.id,
                    "db_session_factory": db_session_factory,
                }
            },
        )
        assert result["is_completed"] is True


class TestCalculateSavingPlan:
    """Tests for the calculate_saving_plan LangGraph tool."""

    def test_monthly_calculation(self) -> None:
        """Calculates monthly saving needed."""
        result = calculate_saving_plan.invoke(
            {
                "target_amount": "100000",
                "current_amount": "40000",
                "months": "12",
            }
        )
        assert result["monthly_saving_needed"] == "5000.00"
        assert result["remaining_amount"] == "60000"

    def test_time_calculation(self) -> None:
        """Calculates time to reach goal."""
        result = calculate_saving_plan.invoke(
            {
                "target_amount": "100000",
                "current_amount": "40000",
                "monthly_saving": "5000",
            }
        )
        assert result["months_needed"] == 12

    def test_both_calculations(self) -> None:
        """Returns both monthly and time calculations."""
        result = calculate_saving_plan.invoke(
            {
                "target_amount": "100000",
                "current_amount": "0",
                "months": "10",
                "monthly_saving": "8000",
            }
        )
        assert "monthly_saving_needed" in result
        assert "months_needed" in result

    def test_default_current_amount(self) -> None:
        """Uses 0 as default current amount."""
        result = calculate_saving_plan.invoke(
            {
                "target_amount": "100000",
                "months": "10",
            }
        )
        assert result["current_amount"] == "0"
        assert result["monthly_saving_needed"] == "10000.00"
