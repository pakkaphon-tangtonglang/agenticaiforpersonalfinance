"""Tests for recommendation service layer."""

from decimal import Decimal
from typing import Any

import pytest

from finance_ai.tools.recommendation_service import (
    Recommendation,
    RecommendationReport,
    analyze_expense_patterns,
    analyze_goal_progress,
    analyze_investment_risk,
    analyze_savings_rate,
    analyze_tax_optimization,
    calculate_health_score,
    generate_recommendations,
)

# -- Fixtures for test data --


def _make_expense_data(
    total: str = "0",
    categories: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build expense summary test data."""
    return {
        "domain": "expense",
        "total_amount": total,
        "transaction_count": len(categories) if categories else 0,
        "category_breakdown": categories or [],
    }


def _make_income_data(total: str = "0") -> dict[str, Any]:
    """Build income summary test data."""
    return {
        "domain": "income",
        "tax_year": 2026,
        "total_income": total,
        "source_count": 1 if total != "0" else 0,
        "sources": [],
    }


def _make_tax_data(
    status: str = "draft",
    gross: str = "0",
    deductions: str = "0",
) -> dict[str, Any]:
    """Build tax summary test data."""
    return {
        "domain": "tax",
        "tax_year": 2026,
        "status": status,
        "gross_income": gross,
        "total_deductions": deductions,
        "total_tax": "0",
    }


def _make_portfolio_data(
    total_value: str = "0",
    gain_loss: str = "0",
    holdings: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build portfolio summary test data."""
    return {
        "domain": "investment",
        "total_value": total_value,
        "total_cost": "0",
        "total_gain_loss": gain_loss,
        "holding_count": len(holdings) if holdings else 0,
        "holdings": holdings or [],
    }


def _make_goals_data(
    goals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build goals summary test data."""
    return {
        "domain": "planning",
        "total_goals": len(goals) if goals else 0,
        "active_goals": 0,
        "completed_goals": 0,
        "overall_percentage": "0",
        "goals": goals or [],
    }


# -- TestAnalyzeExpensePatterns --


class TestAnalyzeExpensePatterns:
    """Tests for analyze_expense_patterns function."""

    def test_no_data_returns_empty(self) -> None:
        """Empty expense and income data produces no recommendations."""
        result = analyze_expense_patterns(_make_expense_data(), _make_income_data())
        assert result == []

    def test_balanced_expenses_no_recommendations(self) -> None:
        """Balanced expenses within income produce no recommendations."""
        expense = _make_expense_data(
            "9000",
            [
                {"category": "food", "label": "อาหาร", "amount": "3000"},
                {"category": "transport", "label": "การเดินทาง", "amount": "3000"},
                {"category": "utilities", "label": "สาธารณูปโภค", "amount": "3000"},
            ],
        )
        income = _make_income_data("600000")  # 50k/month
        result = analyze_expense_patterns(expense, income)
        assert result == []

    def test_overspending_triggers_recommendation(self) -> None:
        """Expenses > 80% of monthly income triggers recommendation."""
        expense = _make_expense_data("45000")  # 45k/month
        income = _make_income_data("480000")  # 40k/month -> 45/40 = 112%
        result = analyze_expense_patterns(expense, income)
        assert len(result) >= 1
        assert any(r.category == "savings_rate" for r in result)

    def test_concentration_triggers_recommendation(self) -> None:
        """Single category > 40% of total triggers recommendation."""
        expense = _make_expense_data(
            "10000",
            [
                {"category": "food", "label": "อาหาร", "amount": "6000"},
                {"category": "transport", "label": "การเดินทาง", "amount": "4000"},
            ],
        )
        income = _make_income_data("1200000")
        result = analyze_expense_patterns(expense, income)
        assert any(r.category == "expense_optimization" for r in result)


# -- TestAnalyzeTaxOptimization --


class TestAnalyzeTaxOptimization:
    """Tests for analyze_tax_optimization function."""

    def test_no_income_returns_empty(self) -> None:
        """No income data returns no recommendations."""
        result = analyze_tax_optimization(_make_tax_data(), _make_income_data())
        assert result == []

    def test_no_filing_triggers_recommendation(self) -> None:
        """Missing tax filing triggers urgent recommendation."""
        tax = _make_tax_data(status="not_found")
        income = _make_income_data("600000")
        result = analyze_tax_optimization(tax, income)
        assert len(result) == 1
        assert result[0].priority == 5
        assert result[0].category == "tax_optimization"

    def test_low_deduction_triggers_recommendation(self) -> None:
        """Low deduction utilization triggers recommendation."""
        tax = _make_tax_data(
            status="draft",
            gross="1200000",
            deductions="60000",  # very low vs 30% of 1.2M = 360k
        )
        income = _make_income_data("1200000")
        result = analyze_tax_optimization(tax, income)
        assert any(r.category == "tax_optimization" for r in result)

    def test_good_deductions_no_recommendation(self) -> None:
        """Adequate deduction usage produces no recommendation."""
        tax = _make_tax_data(
            status="draft",
            gross="600000",
            deductions="180000",  # 100% of 30% * 600k
        )
        income = _make_income_data("600000")
        result = analyze_tax_optimization(tax, income)
        # No "ใช้สิทธิ์ลดหย่อนภาษีน้อย" recommendation
        assert not any("ใช้สิทธิ์" in r.title for r in result)


# -- TestAnalyzeInvestmentRisk --


class TestAnalyzeInvestmentRisk:
    """Tests for analyze_investment_risk function."""

    def test_no_holdings_returns_empty(self) -> None:
        """No holdings produces no recommendations."""
        result = analyze_investment_risk(_make_portfolio_data())
        assert result == []

    def test_concentration_triggers_recommendation(self) -> None:
        """Single holding > 30% of portfolio triggers recommendation."""
        portfolio = _make_portfolio_data(
            total_value="100000",
            holdings=[
                {"symbol": "PTT.BK", "current_value": "80000", "gain_loss": "0"},
                {"symbol": "AOT.BK", "current_value": "20000", "gain_loss": "0"},
            ],
        )
        result = analyze_investment_risk(portfolio)
        assert any(r.category == "investment_rebalancing" for r in result)
        assert any("PTT.BK" in r.title for r in result)

    def test_diversified_no_recommendation(self) -> None:
        """Well-diversified portfolio produces no concentration recommendations."""
        portfolio = _make_portfolio_data(
            total_value="100000",
            holdings=[
                {"symbol": "PTT.BK", "current_value": "25000", "gain_loss": "1000"},
                {"symbol": "AOT.BK", "current_value": "25000", "gain_loss": "500"},
                {"symbol": "K-EQUITY", "current_value": "25000", "gain_loss": "0"},
                {"symbol": "SCB-SSF", "current_value": "25000", "gain_loss": "-500"},
            ],
        )
        result = analyze_investment_risk(portfolio)
        # No concentration recommendations
        assert not any("สัดส่วนสูง" in r.title for r in result)

    def test_portfolio_loss_triggers_recommendation(self) -> None:
        """Portfolio with overall loss triggers recommendation."""
        portfolio = _make_portfolio_data(
            total_value="80000",
            gain_loss="-20000",
            holdings=[
                {"symbol": "PTT.BK", "current_value": "80000", "gain_loss": "-20000"},
            ],
        )
        result = analyze_investment_risk(portfolio)
        assert any(r.title == "พอร์ตขาดทุน" for r in result)


# -- TestAnalyzeGoalProgress --


class TestAnalyzeGoalProgress:
    """Tests for analyze_goal_progress function."""

    def test_no_goals_returns_empty(self) -> None:
        """No goals produces no recommendations."""
        result = analyze_goal_progress(_make_goals_data())
        assert result == []

    def test_behind_schedule_triggers_recommendation(self) -> None:
        """Goal with low progress triggers recommendation."""
        goals = _make_goals_data(
            goals=[
                {
                    "name": "ออมเงินฉุกเฉิน",
                    "goal_type": "emergency_fund",
                    "target_amount": "100000",
                    "current_amount": "10000",
                    "percentage": "10.00",
                    "is_completed": False,
                },
            ]
        )
        result = analyze_goal_progress(goals)
        assert len(result) == 1
        assert result[0].category == "goal_progress"
        assert result[0].priority == 4

    def test_completed_goals_ignored(self) -> None:
        """Completed goals are not flagged."""
        goals = _make_goals_data(
            goals=[
                {
                    "name": "ออมเงินท่องเที่ยว",
                    "goal_type": "travel",
                    "target_amount": "50000",
                    "current_amount": "50000",
                    "percentage": "100.00",
                    "is_completed": True,
                },
            ]
        )
        result = analyze_goal_progress(goals)
        assert result == []

    def test_on_track_goal_no_recommendation(self) -> None:
        """Goal with > 50% progress is not flagged."""
        goals = _make_goals_data(
            goals=[
                {
                    "name": "ออมเงินท่องเที่ยว",
                    "goal_type": "travel",
                    "target_amount": "100000",
                    "current_amount": "60000",
                    "percentage": "60.00",
                    "is_completed": False,
                },
            ]
        )
        result = analyze_goal_progress(goals)
        assert result == []


# -- TestAnalyzeSavingsRate --


class TestAnalyzeSavingsRate:
    """Tests for analyze_savings_rate function."""

    def test_no_income_returns_empty(self) -> None:
        """No income data returns only emergency fund if no goal."""
        result = analyze_savings_rate(_make_income_data(), _make_expense_data(), _make_goals_data())
        # Only emergency fund recommendation (no savings rate since income=0)
        assert all(r.category == "emergency_fund" for r in result)

    def test_low_savings_triggers_recommendation(self) -> None:
        """Savings < 20% triggers recommendation."""
        income = _make_income_data("600000")  # 50k/month
        expense = _make_expense_data("45000")  # 45k/month = 10% savings
        goals = _make_goals_data(
            goals=[
                {
                    "goal_type": "emergency_fund",
                    "is_completed": False,
                    "percentage": "0",
                    "name": "EF",
                    "target_amount": "100000",
                    "current_amount": "0",
                }
            ]
        )
        result = analyze_savings_rate(income, expense, goals)
        assert any(r.category == "savings_rate" for r in result)

    def test_good_savings_no_recommendation(self) -> None:
        """Savings >= 20% produces no savings rate recommendation."""
        income = _make_income_data("600000")  # 50k/month
        expense = _make_expense_data("30000")  # 30k/month = 40% savings
        goals = _make_goals_data(
            goals=[
                {
                    "goal_type": "emergency_fund",
                    "is_completed": False,
                    "percentage": "50",
                    "name": "EF",
                    "target_amount": "100000",
                    "current_amount": "50000",
                }
            ]
        )
        result = analyze_savings_rate(income, expense, goals)
        assert not any(r.category == "savings_rate" for r in result)

    def test_no_emergency_fund_triggers_recommendation(self) -> None:
        """No emergency fund goal triggers recommendation."""
        result = analyze_savings_rate(
            _make_income_data("600000"),
            _make_expense_data("20000"),
            _make_goals_data(),
        )
        assert any(r.category == "emergency_fund" for r in result)

    def test_has_emergency_fund_no_recommendation(self) -> None:
        """Having emergency fund goal suppresses that recommendation."""
        goals = _make_goals_data(
            goals=[
                {
                    "goal_type": "emergency_fund",
                    "is_completed": False,
                    "percentage": "50",
                    "name": "EF",
                    "target_amount": "100000",
                    "current_amount": "50000",
                }
            ]
        )
        result = analyze_savings_rate(
            _make_income_data("600000"),
            _make_expense_data("20000"),
            goals,
        )
        assert not any(r.category == "emergency_fund" for r in result)


# -- TestCalculateHealthScore --


class TestCalculateHealthScore:
    """Tests for calculate_health_score function."""

    def test_no_recommendations_returns_100(self) -> None:
        """No recommendations means perfect health score."""
        assert calculate_health_score([]) == 100

    def test_recommendations_reduce_score(self) -> None:
        """Each recommendation reduces score by priority * penalty."""
        recs = [
            Recommendation(
                category="tax_optimization",
                priority=5,
                title="test",
                description="test",
            ),
        ]
        score = calculate_health_score(recs)
        assert score == 75  # 100 - 5*5

    def test_score_never_below_zero(self) -> None:
        """Score should never go below 0."""
        recs = [
            Recommendation(
                category="test",
                priority=5,
                title="test",
                description="test",
            )
            for _ in range(10)
        ]
        score = calculate_health_score(recs)
        assert score == 0  # 100 - 50*5 = clamped to 0


# -- TestGenerateRecommendations (integration with DB) --


class TestGenerateRecommendations:
    """Integration tests for generate_recommendations."""

    def test_empty_user_returns_report(
        self,
        test_session: Any,
        sample_user: Any,
    ) -> None:
        """Empty user data returns a valid report structure."""
        report = generate_recommendations(test_session, sample_user.id, 2026, 3)
        assert isinstance(report, RecommendationReport)
        assert report.user_id == sample_user.id
        assert report.health_score <= 100
        assert report.health_score >= 0
        assert report.total_recommendations == len(report.recommendations)

    def test_recommendations_sorted_by_priority(
        self,
        test_session: Any,
        sample_user: Any,
    ) -> None:
        """Recommendations should be sorted by priority descending."""
        report = generate_recommendations(test_session, sample_user.id, 2026, 3)
        if len(report.recommendations) >= 2:
            for i in range(len(report.recommendations) - 1):
                assert report.recommendations[i].priority >= report.recommendations[i + 1].priority


# -- TestPydanticModels --


class TestRecommendationModel:
    """Tests for Recommendation Pydantic model."""

    def test_valid_recommendation(self) -> None:
        """Should create recommendation with valid fields."""
        rec = Recommendation(
            category="tax_optimization",
            priority=4,
            title="ยังไม่ได้ยื่นภาษี",
            description="ยังไม่พบข้อมูลการยื่นภาษี",
            action_items=["ยื่นภาษี"],
            estimated_impact="หลีกเลี่ยงค่าปรับ",
        )
        assert rec.category == "tax_optimization"
        assert rec.priority == 4

    def test_priority_must_be_1_to_5(self) -> None:
        """Priority outside 1-5 should raise validation error."""
        with pytest.raises(Exception):
            Recommendation(
                category="test",
                priority=0,
                title="test",
                description="test",
            )

    def test_default_action_items(self) -> None:
        """Default action_items should be empty list."""
        rec = Recommendation(
            category="test",
            priority=1,
            title="test",
            description="test",
        )
        assert rec.action_items == []
        assert rec.estimated_impact == ""
