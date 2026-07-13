"""Tests for pure planning calculation functions."""

from decimal import Decimal

import pytest

from finance_ai.tools.planning_calculator import (
    GoalProgress,
    GoalRecord,
    GoalSummaryResult,
    calculate_goal_progress,
    calculate_monthly_saving_needed,
    calculate_time_to_goal,
    summarize_goals,
)


@pytest.fixture
def savings_goal() -> GoalRecord:
    """A savings goal at 40% progress."""
    return GoalRecord(
        goal_id="goal-1",
        goal_type="savings",
        name="ออมเงินฉุกเฉิน",
        target_amount=Decimal("100000"),
        current_amount=Decimal("40000"),
        priority=4,
    )


@pytest.fixture
def completed_goal() -> GoalRecord:
    """A completed goal."""
    return GoalRecord(
        goal_id="goal-2",
        goal_type="travel",
        name="ทริปญี่ปุ่น",
        target_amount=Decimal("50000"),
        current_amount=Decimal("50000"),
        is_completed=True,
        priority=2,
    )


@pytest.fixture
def zero_target_goal() -> GoalRecord:
    """A goal with zero target (edge case)."""
    return GoalRecord(
        goal_id="goal-3",
        goal_type="savings",
        name="Test",
        target_amount=Decimal("0"),
        current_amount=Decimal("0"),
    )


class TestCalculateGoalProgress:
    """Tests for calculate_goal_progress."""

    def test_normal_progress(self, savings_goal: GoalRecord) -> None:
        """Calculates correct percentage for partial progress."""
        result = calculate_goal_progress(savings_goal)
        assert result.percentage == Decimal("40.00")
        assert result.remaining_amount == Decimal("60000")
        assert result.is_completed is False

    def test_completed_goal(self, completed_goal: GoalRecord) -> None:
        """Shows 100% for completed goal."""
        result = calculate_goal_progress(completed_goal)
        assert result.percentage == Decimal("100.00")
        assert result.remaining_amount == Decimal("0")
        assert result.is_completed is True

    def test_over_target_caps_at_100(self) -> None:
        """Percentage caps at 100% when over target."""
        goal = GoalRecord(
            goal_id="g1",
            goal_type="savings",
            name="Test",
            target_amount=Decimal("100"),
            current_amount=Decimal("150"),
        )
        result = calculate_goal_progress(goal)
        assert result.percentage == Decimal("100.00")
        assert result.remaining_amount == Decimal("0")

    def test_zero_progress(self) -> None:
        """Shows 0% for no progress."""
        goal = GoalRecord(
            goal_id="g1",
            goal_type="savings",
            name="Test",
            target_amount=Decimal("100000"),
            current_amount=Decimal("0"),
        )
        result = calculate_goal_progress(goal)
        assert result.percentage == Decimal("0.00")
        assert result.remaining_amount == Decimal("100000")

    def test_zero_target_returns_zero_percent(self, zero_target_goal: GoalRecord) -> None:
        """Zero target amount yields 0% progress."""
        result = calculate_goal_progress(zero_target_goal)
        assert result.percentage == Decimal("0.00")

    def test_goal_type_label_is_thai(self, savings_goal: GoalRecord) -> None:
        """Goal type label is in Thai."""
        result = calculate_goal_progress(savings_goal)
        assert result.goal_type_label == "ออมเงิน"

    def test_priority_label_is_thai(self, savings_goal: GoalRecord) -> None:
        """Priority label is in Thai."""
        result = calculate_goal_progress(savings_goal)
        assert result.priority_label == "สูง"

    def test_returns_goal_progress_model(self, savings_goal: GoalRecord) -> None:
        """Returns GoalProgress instance."""
        result = calculate_goal_progress(savings_goal)
        assert isinstance(result, GoalProgress)


class TestCalculateMonthlySavingNeeded:
    """Tests for calculate_monthly_saving_needed."""

    def test_basic_calculation(self) -> None:
        """Calculates correct monthly amount."""
        result = calculate_monthly_saving_needed(Decimal("100000"), Decimal("40000"), 12)
        assert result == Decimal("5000.00")

    def test_already_reached_returns_zero(self) -> None:
        """Returns 0 when goal already met."""
        result = calculate_monthly_saving_needed(Decimal("100000"), Decimal("100000"), 12)
        assert result == Decimal("0.00")

    def test_over_target_returns_zero(self) -> None:
        """Returns 0 when over target."""
        result = calculate_monthly_saving_needed(Decimal("100000"), Decimal("120000"), 12)
        assert result == Decimal("0.00")

    def test_one_month_remaining(self) -> None:
        """Full remaining amount when 1 month left."""
        result = calculate_monthly_saving_needed(Decimal("100000"), Decimal("90000"), 1)
        assert result == Decimal("10000.00")

    def test_zero_months_raises(self) -> None:
        """Raises ValueError for zero months."""
        with pytest.raises(ValueError, match="Months remaining must be positive"):
            calculate_monthly_saving_needed(Decimal("100000"), Decimal("0"), 0)

    def test_negative_months_raises(self) -> None:
        """Raises ValueError for negative months."""
        with pytest.raises(ValueError, match="Months remaining must be positive"):
            calculate_monthly_saving_needed(Decimal("100000"), Decimal("0"), -5)

    def test_rounds_to_two_decimals(self) -> None:
        """Result is rounded to 2 decimal places."""
        result = calculate_monthly_saving_needed(Decimal("100000"), Decimal("0"), 7)
        assert result == Decimal("14285.71")


class TestCalculateTimeToGoal:
    """Tests for calculate_time_to_goal."""

    def test_exact_division(self) -> None:
        """Returns exact months when evenly divisible."""
        result = calculate_time_to_goal(Decimal("100000"), Decimal("40000"), Decimal("5000"))
        assert result == 12

    def test_rounds_up(self) -> None:
        """Rounds up when not evenly divisible."""
        result = calculate_time_to_goal(Decimal("100000"), Decimal("0"), Decimal("7000"))
        assert result == 15  # 100000/7000 = 14.28... -> 15

    def test_already_reached_returns_zero(self) -> None:
        """Returns 0 when goal already met."""
        result = calculate_time_to_goal(Decimal("100000"), Decimal("100000"), Decimal("5000"))
        assert result == 0

    def test_over_target_returns_zero(self) -> None:
        """Returns 0 when over target."""
        result = calculate_time_to_goal(Decimal("100000"), Decimal("120000"), Decimal("5000"))
        assert result == 0

    def test_zero_saving_raises(self) -> None:
        """Raises ValueError for zero monthly saving."""
        with pytest.raises(ValueError, match="Monthly saving must be positive"):
            calculate_time_to_goal(Decimal("100000"), Decimal("0"), Decimal("0"))

    def test_negative_saving_raises(self) -> None:
        """Raises ValueError for negative monthly saving."""
        with pytest.raises(ValueError, match="Monthly saving must be positive"):
            calculate_time_to_goal(Decimal("100000"), Decimal("0"), Decimal("-5000"))


class TestSummarizeGoals:
    """Tests for summarize_goals."""

    def test_empty_list(self) -> None:
        """Returns zeroed summary for empty list."""
        result = summarize_goals([])
        assert result.total_goals == 0
        assert result.active_goals == 0
        assert result.completed_goals == 0
        assert result.total_target_amount == Decimal("0")
        assert result.overall_percentage == Decimal("0.00")

    def test_single_goal(self, savings_goal: GoalRecord) -> None:
        """Summarizes a single active goal."""
        result = summarize_goals([savings_goal])
        assert result.total_goals == 1
        assert result.active_goals == 1
        assert result.completed_goals == 0
        assert result.total_target_amount == Decimal("100000")
        assert result.total_current_amount == Decimal("40000")
        assert result.overall_percentage == Decimal("40.00")

    def test_mixed_goals(
        self,
        savings_goal: GoalRecord,
        completed_goal: GoalRecord,
    ) -> None:
        """Summarizes mix of active and completed goals."""
        result = summarize_goals([savings_goal, completed_goal])
        assert result.total_goals == 2
        assert result.active_goals == 1
        assert result.completed_goals == 1
        assert result.total_target_amount == Decimal("150000")
        assert result.total_current_amount == Decimal("90000")

    def test_progress_list_populated(self, savings_goal: GoalRecord) -> None:
        """Goals list contains GoalProgress for each input."""
        result = summarize_goals([savings_goal])
        assert len(result.goals) == 1
        assert isinstance(result.goals[0], GoalProgress)
        assert result.goals[0].goal_id == "goal-1"

    def test_returns_summary_model(self) -> None:
        """Returns GoalSummaryResult instance."""
        result = summarize_goals([])
        assert isinstance(result, GoalSummaryResult)
