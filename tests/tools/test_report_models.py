"""Tests for report Pydantic models."""

from decimal import Decimal

import pytest

from finance_ai.tools.report_models import (
    ExpenseBreakdownSection,
    FinancialReport,
    GoalProgressSection,
    InvestmentPortfolioSection,
    MonthlyOverviewSection,
    ReportHighlight,
    TaxStatusSection,
)


class TestReportHighlight:
    """Tests for ReportHighlight model."""

    def test_create_valid(self) -> None:
        """Should create with all fields."""
        h = ReportHighlight(
            highlight_type="achievement",
            title="อัตราการออมดี",
            description="ออมได้ 25% ของรายได้",
        )
        assert h.highlight_type == "achievement"
        assert h.title == "อัตราการออมดี"

    def test_requires_all_fields(self) -> None:
        """Should reject missing fields."""
        with pytest.raises(Exception):
            ReportHighlight()  # type: ignore[call-arg]


class TestMonthlyOverviewSection:
    """Tests for MonthlyOverviewSection model."""

    def test_defaults_to_zero(self) -> None:
        """All fields should default to Decimal 0."""
        s = MonthlyOverviewSection()
        assert s.total_income == Decimal("0")
        assert s.total_expenses == Decimal("0")
        assert s.net_savings == Decimal("0")
        assert s.savings_rate == Decimal("0")

    def test_create_with_values(self) -> None:
        """Should accept Decimal values."""
        s = MonthlyOverviewSection(
            total_income=Decimal("50000"),
            total_expenses=Decimal("35000"),
            net_savings=Decimal("15000"),
            savings_rate=Decimal("0.30"),
        )
        assert s.net_savings == Decimal("15000")


class TestExpenseBreakdownSection:
    """Tests for ExpenseBreakdownSection model."""

    def test_defaults(self) -> None:
        """Should default to zero/empty."""
        s = ExpenseBreakdownSection()
        assert s.total_amount == Decimal("0")
        assert s.transaction_count == 0
        assert s.categories == []

    def test_with_categories(self) -> None:
        """Should accept category list."""
        s = ExpenseBreakdownSection(
            total_amount=Decimal("35000"),
            transaction_count=42,
            categories=[{"category": "food", "amount": "15000"}],
        )
        assert len(s.categories) == 1


class TestInvestmentPortfolioSection:
    """Tests for InvestmentPortfolioSection model."""

    def test_defaults(self) -> None:
        """Should default to zero/empty."""
        s = InvestmentPortfolioSection()
        assert s.total_value == Decimal("0")
        assert s.gain_loss_percentage == Decimal("0")
        assert s.holdings == []

    def test_with_values(self) -> None:
        """Should calculate gain/loss correctly."""
        s = InvestmentPortfolioSection(
            total_value=Decimal("550000"),
            total_cost=Decimal("500000"),
            total_gain_loss=Decimal("50000"),
            gain_loss_percentage=Decimal("10.00"),
            holding_count=3,
        )
        assert s.holding_count == 3


class TestGoalProgressSection:
    """Tests for GoalProgressSection model."""

    def test_defaults(self) -> None:
        """Should default to zero."""
        s = GoalProgressSection()
        assert s.total_goals == 0
        assert s.active_goals == 0
        assert s.completed_goals == 0

    def test_with_goals(self) -> None:
        """Should accept goal list."""
        s = GoalProgressSection(
            total_goals=3,
            active_goals=2,
            completed_goals=1,
            overall_percentage=Decimal("60.00"),
            goals=[{"name": "Emergency Fund"}],
        )
        assert len(s.goals) == 1


class TestTaxStatusSection:
    """Tests for TaxStatusSection model."""

    def test_defaults(self) -> None:
        """Should default to not_filed with zero amounts."""
        s = TaxStatusSection()
        assert s.status == "not_filed"
        assert s.gross_income == Decimal("0")

    def test_with_values(self) -> None:
        """Should accept tax data."""
        s = TaxStatusSection(
            status="filed",
            gross_income=Decimal("600000"),
            total_deductions=Decimal("160000"),
            total_tax=Decimal("29000"),
            effective_rate=Decimal("4.83"),
            tax_year=2026,
        )
        assert s.tax_year == 2026


class TestFinancialReport:
    """Tests for FinancialReport model."""

    def test_create_minimal(self) -> None:
        """Should create with just required fields."""
        report = FinancialReport(
            user_id="test-user",
            generated_at="2026-03-10T00:00:00",
        )
        assert report.user_id == "test-user"
        assert report.health_score == 100
        assert report.highlights == []

    def test_health_score_bounds(self) -> None:
        """Health score must be 0-100."""
        with pytest.raises(Exception):
            FinancialReport(
                user_id="x",
                generated_at="2026-01-01",
                health_score=101,
            )

    def test_model_dump(self) -> None:
        """model_dump should include all sections."""
        report = FinancialReport(
            user_id="abc",
            generated_at="2026-03-10T00:00:00",
            report_type="monthly",
            year=2026,
            month=3,
        )
        data = report.model_dump()
        assert "monthly_overview" in data
        assert "expense_breakdown" in data
        assert "investment_portfolio" in data
        assert "goal_progress" in data
        assert "tax_status" in data
        assert "highlights" in data

    def test_default_sections_are_empty(self) -> None:
        """Default sections should have zero values."""
        report = FinancialReport(user_id="abc", generated_at="2026-01-01")
        assert report.monthly_overview.total_income == Decimal("0")
        assert report.expense_breakdown.transaction_count == 0
        assert report.investment_portfolio.holding_count == 0
        assert report.goal_progress.total_goals == 0
        assert report.tax_status.status == "not_filed"
