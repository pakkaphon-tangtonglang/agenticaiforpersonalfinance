"""Tests for pillar-based financial health scoring."""

from decimal import Decimal
from typing import Any

import pytest

from finance_ai.tools.health_score_service import (
    calculate_health_score,
    linear_score,
    score_invest_pillar,
    score_plan_pillar,
    score_save_pillar,
    score_spend_pillar,
)

_DEFAULT_DATA: dict[str, Any] = {
    "income": "0",
    "expense": "0",
    "holdings": [],
    "gain_loss": "0",
    "goals": [],
    "tax_status": "not_found",
    "annual_income": "0",
}


def _build_data(**overrides: Any) -> dict[str, dict[str, Any]]:
    """Build a gathered-data dict shaped like gather_all_financial_data().

    Args:
        **overrides: Any key of _DEFAULT_DATA to replace.

    Returns:
        Dict keyed by domain name.
    """
    values = {**_DEFAULT_DATA, **overrides}
    total_value = sum(Decimal(h["current_value"]) for h in values["holdings"])
    return {
        "income_monthly": {"total_income": values["income"]},
        "expense": {"total_amount": values["expense"]},
        "portfolio": {
            "holdings": values["holdings"],
            "total_value": str(total_value),
            "total_gain_loss": values["gain_loss"],
        },
        "goals": {"goals": values["goals"]},
        "tax": {"status": values["tax_status"]},
        "income": {"total_income": values["annual_income"]},
    }


class TestLinearScore:
    """Tests for linear_score interpolation."""

    def test_at_best_returns_100(self) -> None:
        """At best returns 100."""
        assert linear_score(Decimal("0.5"), Decimal("0.5"), Decimal("1.2")) == 100

    def test_at_worst_returns_0(self) -> None:
        """At worst returns 0."""
        assert linear_score(Decimal("1.2"), Decimal("0.5"), Decimal("1.2")) == 0

    def test_midpoint_returns_50(self) -> None:
        """Midpoint returns 50."""
        assert linear_score(Decimal("0.85"), Decimal("0.5"), Decimal("1.2")) == 50

    def test_beyond_range_is_clamped(self) -> None:
        """Beyond range is clamped."""
        assert linear_score(Decimal("0.1"), Decimal("0.5"), Decimal("1.2")) == 100
        assert linear_score(Decimal("3"), Decimal("0.5"), Decimal("1.2")) == 0

    def test_ascending_direction(self) -> None:
        """Ascending direction."""
        assert linear_score(Decimal("0.10"), Decimal("0.20"), Decimal("0")) == 50

    def test_equal_bounds_raises(self) -> None:
        """Equal bounds raises."""
        with pytest.raises(ValueError, match="must differ"):
            linear_score(Decimal("1"), Decimal("1"), Decimal("1"))


class TestSpendPillar:
    """Tests for the spending pillar."""

    def test_low_expense_ratio_scores_100(self) -> None:
        """Low expense ratio scores 100."""
        assert score_spend_pillar(Decimal("50000"), Decimal("20000")) == 100

    def test_expense_above_income_scores_0(self) -> None:
        """Expense above income scores 0."""
        assert score_spend_pillar(Decimal("10000"), Decimal("15000")) == 0

    def test_missing_income_returns_none(self) -> None:
        """Missing income returns none."""
        assert score_spend_pillar(Decimal("0"), Decimal("1000")) is None

    def test_missing_expense_returns_none(self) -> None:
        """Missing expense returns none."""
        assert score_spend_pillar(Decimal("1000"), Decimal("0")) is None


class TestSavePillar:
    """Tests for the savings pillar."""

    def test_twenty_percent_savings_scores_100(self) -> None:
        """Twenty percent savings scores 100."""
        assert score_save_pillar(Decimal("10000"), Decimal("8000")) == 100

    def test_ten_percent_savings_scores_50(self) -> None:
        """Ten percent savings scores 50."""
        assert score_save_pillar(Decimal("10000"), Decimal("9000")) == 50

    def test_negative_savings_scores_0(self) -> None:
        """Negative savings scores 0."""
        assert score_save_pillar(Decimal("10000"), Decimal("12000")) == 0

    def test_missing_data_returns_none(self) -> None:
        """Missing data returns none."""
        assert score_save_pillar(Decimal("0"), Decimal("0")) is None


class TestInvestPillar:
    """Tests for the investment pillar."""

    def test_no_holdings_returns_none(self) -> None:
        """No holdings returns none."""
        assert score_invest_pillar(_build_data()["portfolio"]) is None

    def test_diversified_portfolio_scores_100(self) -> None:
        """Diversified portfolio scores 100."""
        holdings = [{"current_value": "100"} for _ in range(4)]
        assert score_invest_pillar(_build_data(holdings=holdings)["portfolio"]) == 100

    def test_single_holding_scores_floor(self) -> None:
        """Single holding scores floor."""
        holdings = [{"current_value": "100"}]
        assert score_invest_pillar(_build_data(holdings=holdings)["portfolio"]) == 30

    def test_loss_reduces_score(self) -> None:
        """Loss reduces score."""
        holdings = [{"current_value": "100"} for _ in range(4)]
        data = _build_data(holdings=holdings, gain_loss="-50")
        assert score_invest_pillar(data["portfolio"]) == 80


class TestPlanPillar:
    """Tests for the planning pillar."""

    def test_no_goals_no_income_returns_none(self) -> None:
        """No goals no income returns none."""
        data = _build_data()
        assert score_plan_pillar(data["goals"], data["tax"], data["income"]) is None

    def test_tax_not_filed_with_income_scores_0(self) -> None:
        """Tax not filed with income scores 0."""
        data = _build_data(annual_income="300000")
        assert score_plan_pillar(data["goals"], data["tax"], data["income"]) == 0

    def test_tax_filed_scores_100(self) -> None:
        """Tax filed scores 100."""
        data = _build_data(annual_income="300000", tax_status="filed")
        assert score_plan_pillar(data["goals"], data["tax"], data["income"]) == 100

    def test_goals_only(self) -> None:
        """Goals only."""
        goals = [{"goal_type": "custom", "percentage": "40.00", "is_completed": False}]
        data = _build_data(goals=goals)
        assert score_plan_pillar(data["goals"], data["tax"], data["income"]) == 40

    def test_emergency_goal_tax_and_goals_averaged(self) -> None:
        """Emergency goal tax and goals averaged."""
        goals = [{"goal_type": "emergency_fund", "percentage": "80.00", "is_completed": False}]
        data = _build_data(goals=goals, annual_income="300000", tax_status="filed")
        # emergency 80, active goals 80, tax 100 -> mean 86.67 -> 87
        assert score_plan_pillar(data["goals"], data["tax"], data["income"]) == 87

    def test_completed_goal_counts_as_100(self) -> None:
        """Completed goal counts as 100."""
        goals = [{"goal_type": "custom", "percentage": "100.00", "is_completed": True}]
        data = _build_data(goals=goals)
        assert score_plan_pillar(data["goals"], data["tax"], data["income"]) == 100


class TestCalculateHealthScore:
    """Tests for the overall health score."""

    def test_empty_data_is_insufficient(self) -> None:
        """Empty data is insufficient."""
        result = calculate_health_score(_build_data())
        assert result.score is None
        assert result.status == "insufficient_data"
        assert result.missing_data == ["รายรับเดือนนี้", "รายจ่ายเดือนนี้"]

    def test_income_only_is_insufficient(self) -> None:
        """Income only is insufficient."""
        result = calculate_health_score(_build_data(income="30000"))
        assert result.score is None
        assert result.missing_data == ["รายจ่ายเดือนนี้"]

    def test_insufficient_still_reports_known_pillars(self) -> None:
        """Insufficient still reports known pillars."""
        goals = [{"goal_type": "custom", "percentage": "40.00", "is_completed": False}]
        result = calculate_health_score(_build_data(goals=goals))
        assert result.score is None
        assert result.pillars["plan"] == 40

    def test_averages_only_known_pillars(self) -> None:
        """Averages only known pillars."""
        result = calculate_health_score(_build_data(income="10000", expense="9000"))
        # spend: ratio 0.9 -> 43; save: 10% -> 50; invest/plan unknown; 46.5 rounds half-up
        assert result.pillars == {"spend": 43, "save": 50, "invest": None, "plan": None}
        assert result.score == 47
        assert result.status == "scored"
        assert result.coverage == Decimal("0.5000")
        assert result.missing_data == []

    def test_full_coverage(self) -> None:
        """Full coverage."""
        holdings = [{"current_value": "100"} for _ in range(4)]
        goals = [{"goal_type": "emergency_fund", "percentage": "100.00", "is_completed": True}]
        data = _build_data(
            income="50000",
            expense="20000",
            holdings=holdings,
            goals=goals,
            tax_status="filed",
            annual_income="600000",
        )
        result = calculate_health_score(data)
        assert result.score == 100
        assert result.coverage == Decimal("1.0000")
