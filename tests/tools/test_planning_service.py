"""Tests for planning service with database integration."""

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session

from finance_ai.database.crud.financial_goal_crud import FinancialGoalCRUD
from finance_ai.database.models.financial_goal import FinancialGoal
from finance_ai.database.models.user import User
from finance_ai.tools.planning_service import (
    convert_goal_to_record,
    create_goal,
    delete_goal,
    get_active_goals,
    get_user_goals,
    mark_goal_completed,
    summarize_goals_for_user,
    update_goal_progress,
)


class TestConvertGoalToRecord:
    """Tests for converting FinancialGoal to GoalRecord."""

    def test_converts_all_fields(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Converts goal with all fields populated."""
        crud = FinancialGoalCRUD()
        goal = crud.create(
            test_session,
            user_id=sample_user.id,
            goal_type="savings",
            name="เงินฉุกเฉิน",
            target_amount=Decimal("100000"),
            current_amount=Decimal("30000"),
            target_date=date(2027, 12, 31),
            priority=4,
            is_completed=False,
        )
        record = convert_goal_to_record(goal)
        assert record.goal_type == "savings"
        assert record.name == "เงินฉุกเฉิน"
        assert record.target_amount == Decimal("100000")
        assert record.current_amount == Decimal("30000")
        assert record.target_date == date(2027, 12, 31)
        assert record.priority == 4
        assert record.is_completed is False


class TestCreateGoal:
    """Tests for create_goal service function."""

    def test_creates_valid_goal(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Creates a goal with valid inputs."""
        goal = create_goal(
            test_session,
            sample_user.id,
            "savings",
            "ออมเงินฉุกเฉิน",
            Decimal("100000"),
            date(2027, 12, 31),
            4,
        )
        assert goal.goal_type == "savings"
        assert goal.name == "ออมเงินฉุกเฉิน"
        assert goal.target_amount == Decimal("100000")
        assert goal.current_amount == Decimal("0")
        assert goal.is_completed is False

    def test_normalizes_goal_type(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Normalizes goal type to lowercase."""
        goal = create_goal(
            test_session,
            sample_user.id,
            "Savings",
            "Test",
            Decimal("50000"),
        )
        assert goal.goal_type == "savings"

    def test_invalid_goal_type_raises(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Raises ValueError for unknown goal type."""
        with pytest.raises(ValueError, match="Unknown goal type"):
            create_goal(
                test_session,
                sample_user.id,
                "crypto_moon",
                "Test",
                Decimal("50000"),
            )

    def test_invalid_amount_raises(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Raises ValueError for invalid amount."""
        with pytest.raises(ValueError, match="Goal amount"):
            create_goal(
                test_session,
                sample_user.id,
                "savings",
                "Test",
                Decimal("-100"),
            )

    def test_invalid_priority_raises(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Raises ValueError for out-of-range priority."""
        with pytest.raises(ValueError, match="Priority must be between"):
            create_goal(
                test_session,
                sample_user.id,
                "savings",
                "Test",
                Decimal("50000"),
                priority=10,
            )

    def test_default_priority(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Uses default priority when not specified."""
        goal = create_goal(
            test_session,
            sample_user.id,
            "savings",
            "Test",
            Decimal("50000"),
        )
        assert goal.priority == 3


class TestGetUserGoals:
    """Tests for get_user_goals service function."""

    def test_returns_empty_for_no_goals(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Returns empty list when user has no goals."""
        goals = get_user_goals(test_session, sample_user.id)
        assert goals == []

    def test_returns_all_goals(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Returns all goals for the user."""
        create_goal(
            test_session,
            sample_user.id,
            "savings",
            "Goal 1",
            Decimal("100000"),
        )
        create_goal(
            test_session,
            sample_user.id,
            "retirement",
            "Goal 2",
            Decimal("5000000"),
        )
        goals = get_user_goals(test_session, sample_user.id)
        assert len(goals) == 2


class TestGetActiveGoals:
    """Tests for get_active_goals service function."""

    def test_excludes_completed(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Excludes completed goals from results."""
        goal = create_goal(
            test_session,
            sample_user.id,
            "savings",
            "Goal 1",
            Decimal("100000"),
        )
        mark_goal_completed(test_session, goal.id)
        create_goal(
            test_session,
            sample_user.id,
            "retirement",
            "Goal 2",
            Decimal("5000000"),
        )
        active = get_active_goals(test_session, sample_user.id)
        assert len(active) == 1
        assert active[0].name == "Goal 2"


class TestUpdateGoalProgress:
    """Tests for update_goal_progress service function."""

    def test_updates_amount(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Updates current amount successfully."""
        goal = create_goal(
            test_session,
            sample_user.id,
            "savings",
            "Test",
            Decimal("100000"),
        )
        updated = update_goal_progress(test_session, goal.id, Decimal("50000"))
        assert updated.current_amount == Decimal("50000")

    def test_auto_completes_when_target_reached(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Marks goal as completed when target is reached."""
        goal = create_goal(
            test_session,
            sample_user.id,
            "savings",
            "Test",
            Decimal("100000"),
        )
        updated = update_goal_progress(test_session, goal.id, Decimal("100000"))
        assert updated.is_completed is True

    def test_negative_amount_raises(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Raises ValueError for negative amount."""
        goal = create_goal(
            test_session,
            sample_user.id,
            "savings",
            "Test",
            Decimal("100000"),
        )
        with pytest.raises(ValueError, match="cannot be negative"):
            update_goal_progress(test_session, goal.id, Decimal("-100"))

    def test_nonexistent_goal_raises(
        self,
        test_session: Session,
    ) -> None:
        """Raises ValueError for nonexistent goal."""
        with pytest.raises(ValueError, match="Goal not found"):
            update_goal_progress(test_session, "nonexistent-id", Decimal("50000"))


class TestMarkGoalCompleted:
    """Tests for mark_goal_completed service function."""

    def test_marks_completed(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Marks goal as completed."""
        goal = create_goal(
            test_session,
            sample_user.id,
            "savings",
            "Test",
            Decimal("100000"),
        )
        updated = mark_goal_completed(test_session, goal.id)
        assert updated.is_completed is True

    def test_nonexistent_raises(
        self,
        test_session: Session,
    ) -> None:
        """Raises ValueError for nonexistent goal."""
        with pytest.raises(ValueError, match="Goal not found"):
            mark_goal_completed(test_session, "nonexistent-id")


class TestDeleteGoal:
    """Tests for delete_goal service function."""

    def test_deletes_existing(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Deletes an existing goal and returns True."""
        goal = create_goal(
            test_session,
            sample_user.id,
            "savings",
            "Test",
            Decimal("100000"),
        )
        result = delete_goal(test_session, goal.id)
        assert result is True
        assert get_user_goals(test_session, sample_user.id) == []

    def test_returns_false_for_nonexistent(
        self,
        test_session: Session,
    ) -> None:
        """Returns False for nonexistent goal."""
        result = delete_goal(test_session, "nonexistent-id")
        assert result is False


class TestSummarizeGoalsForUser:
    """Tests for summarize_goals_for_user service function."""

    def test_empty_summary(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Returns zeroed summary for user with no goals."""
        summary = summarize_goals_for_user(test_session, sample_user.id)
        assert summary.total_goals == 0

    def test_includes_all_goals(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Summary includes all user goals."""
        create_goal(
            test_session,
            sample_user.id,
            "savings",
            "Goal 1",
            Decimal("100000"),
        )
        create_goal(
            test_session,
            sample_user.id,
            "retirement",
            "Goal 2",
            Decimal("5000000"),
        )
        summary = summarize_goals_for_user(test_session, sample_user.id)
        assert summary.total_goals == 2
        assert len(summary.goals) == 2
