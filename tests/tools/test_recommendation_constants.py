"""Tests for recommendation constants."""

from decimal import Decimal

from finance_ai.tools.recommendation_constants import (
    EMERGENCY_FUND_MONTHS,
    EXPENSE_CONCENTRATION_THRESHOLD,
    GOAL_BEHIND_SCHEDULE_PERCENTAGE,
    GOAL_DEADLINE_WARNING_MONTHS,
    HEALTH_SCORE_BASE,
    HEALTH_SCORE_PENALTY_PER_PRIORITY,
    INVESTMENT_CONCENTRATION_THRESHOLD,
    OVERSPENDING_THRESHOLD,
    RECOMMENDATION_CATEGORIES,
    RECOMMENDATION_PRIORITIES,
    SAVINGS_RATE_WARNING_THRESHOLD,
    TAX_DEDUCTION_UTILIZATION_THRESHOLD,
)


class TestRecommendationCategories:
    """Tests for RECOMMENDATION_CATEGORIES constant."""

    def test_has_six_categories(self) -> None:
        """Should have exactly 6 recommendation categories."""
        assert len(RECOMMENDATION_CATEGORIES) == 6

    def test_all_values_are_thai_strings(self) -> None:
        """All category labels should be non-empty strings."""
        for key, value in RECOMMENDATION_CATEGORIES.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
            assert len(value) > 0

    def test_contains_expected_keys(self) -> None:
        """Should contain all expected category keys."""
        expected = {
            "expense_optimization",
            "tax_optimization",
            "investment_rebalancing",
            "goal_progress",
            "savings_rate",
            "emergency_fund",
        }
        assert set(RECOMMENDATION_CATEGORIES.keys()) == expected


class TestRecommendationPriorities:
    """Tests for RECOMMENDATION_PRIORITIES constant."""

    def test_has_five_levels(self) -> None:
        """Should have exactly 5 priority levels (1-5)."""
        assert len(RECOMMENDATION_PRIORITIES) == 5

    def test_keys_are_one_to_five(self) -> None:
        """Keys should be integers 1 through 5."""
        assert set(RECOMMENDATION_PRIORITIES.keys()) == {1, 2, 3, 4, 5}


class TestThresholds:
    """Tests for rule threshold constants."""

    def test_savings_rate_is_valid_ratio(self) -> None:
        """Savings rate threshold should be between 0 and 1."""
        assert Decimal("0") < SAVINGS_RATE_WARNING_THRESHOLD < Decimal("1")

    def test_overspending_is_valid_ratio(self) -> None:
        """Overspending threshold should be between 0 and 1."""
        assert Decimal("0") < OVERSPENDING_THRESHOLD < Decimal("1")

    def test_expense_concentration_is_valid_ratio(self) -> None:
        """Expense concentration threshold should be between 0 and 1."""
        assert Decimal("0") < EXPENSE_CONCENTRATION_THRESHOLD < Decimal("1")

    def test_investment_concentration_is_valid_ratio(self) -> None:
        """Investment concentration threshold should be between 0 and 1."""
        assert Decimal("0") < INVESTMENT_CONCENTRATION_THRESHOLD < Decimal("1")

    def test_tax_deduction_is_valid_ratio(self) -> None:
        """Tax deduction utilization threshold should be between 0 and 1."""
        assert Decimal("0") < TAX_DEDUCTION_UTILIZATION_THRESHOLD < Decimal("1")

    def test_emergency_fund_months_positive(self) -> None:
        """Emergency fund months should be positive."""
        assert EMERGENCY_FUND_MONTHS > 0

    def test_goal_deadline_months_positive(self) -> None:
        """Goal deadline warning months should be positive."""
        assert GOAL_DEADLINE_WARNING_MONTHS > 0

    def test_goal_behind_schedule_valid(self) -> None:
        """Goal behind schedule percentage should be valid."""
        assert Decimal("0") < GOAL_BEHIND_SCHEDULE_PERCENTAGE <= Decimal("100")

    def test_health_score_base_is_100(self) -> None:
        """Health score base should be 100."""
        assert HEALTH_SCORE_BASE == 100

    def test_health_score_penalty_positive(self) -> None:
        """Health score penalty per priority should be positive."""
        assert HEALTH_SCORE_PENALTY_PER_PRIORITY > 0
