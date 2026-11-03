"""Tests for adding, editing and deleting goals from the dashboard."""

from decimal import Decimal

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from finance_ai.database.models.user import User
from finance_ai.tools.goal_editor_service import (
    GoalInput,
    GoalNotFoundError,
    create_goal_entry,
    delete_goal_entry,
    list_goal_entries,
    update_goal_entry,
)


def _emergency(**changes: object) -> GoalInput:
    values: dict[str, object] = {
        "name": "เงินสำรองฉุกเฉิน", "goal_type": "emergency_fund",
        "target_amount": "100000", "current_amount": "25000", "target_date": "2027-12-31",
    }  # fmt: skip
    values.update(changes)
    return GoalInput.model_validate(values)


class TestGoalInput:
    """Form validation before any database work."""

    def test_unknown_type_rejected(self) -> None:
        """The form only offers known goal types."""
        with pytest.raises(ValidationError, match="Unknown goal type"):
            _emergency(goal_type="mystery")

    def test_zero_target_rejected(self) -> None:
        """A goal needs a positive target."""
        with pytest.raises(ValidationError):
            _emergency(target_amount="0")

    def test_blank_name_rejected(self) -> None:
        """A goal needs a name."""
        with pytest.raises(ValidationError):
            _emergency(name=" ")

    def test_no_date_ok(self) -> None:
        """The target date is optional."""
        assert _emergency(target_date=None).target_date is None


class TestGoalEditor:
    """Create, list, edit and delete through the dashboard."""

    def test_create_and_list(self, test_session: Session, sample_user: User) -> None:
        """A saved goal shows its label and percentage."""
        create_goal_entry(test_session, sample_user.id, _emergency())
        [goal] = list_goal_entries(test_session, sample_user.id)
        assert (goal.goal_type_label, goal.percentage) == ("เงินสำรองฉุกเฉิน", Decimal("25.00"))
        assert goal.is_completed is False

    def test_update_reaching_target_completes(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Saving the full target marks it completed (percentage capped at 100)."""
        goal = create_goal_entry(test_session, sample_user.id, _emergency())
        updated = update_goal_entry(
            test_session, sample_user.id, goal.goal_id, _emergency(current_amount="120000")
        )
        assert (updated.is_completed, updated.percentage) == (True, Decimal("100.00"))

    def test_delete(self, test_session: Session, sample_user: User) -> None:
        """A deleted goal is gone."""
        goal = create_goal_entry(test_session, sample_user.id, _emergency())
        delete_goal_entry(test_session, sample_user.id, goal.goal_id)
        assert list_goal_entries(test_session, sample_user.id) == []

    def test_other_users_goal_not_found(self, test_session: Session, sample_user: User) -> None:
        """Someone else's id is treated as missing."""
        goal = create_goal_entry(test_session, sample_user.id, _emergency())
        with pytest.raises(GoalNotFoundError):
            delete_goal_entry(test_session, "someone-else", goal.goal_id)
        with pytest.raises(GoalNotFoundError):
            update_goal_entry(test_session, "someone-else", goal.goal_id, _emergency())
