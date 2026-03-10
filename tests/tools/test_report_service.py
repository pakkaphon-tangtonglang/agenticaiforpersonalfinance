"""Tests for report service functions."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

from finance_ai.tools.report_models import (
    ExpenseBreakdownSection,
    FinancialReport,
    GoalProgressSection,
    InvestmentPortfolioSection,
    MonthlyOverviewSection,
    TaxStatusSection,
)
from finance_ai.tools.report_service import (
    _check_goal_highlights,
    _check_portfolio_highlight,
    _check_recommendation_highlights,
    _check_savings_highlight,
    _safe_divide,
    build_expense_breakdown,
    build_goal_progress,
    build_investment_portfolio,
    build_monthly_overview,
    build_tax_status,
    generate_financial_report,
)

# ---------------------------------------------------------------------------
# _safe_divide
# ---------------------------------------------------------------------------


class TestSafeDivide:
    """Tests for _safe_divide helper."""

    def test_normal_division(self) -> None:
        """Should divide normally."""
        assert _safe_divide(Decimal("100"), Decimal("4")) == Decimal("25")

    def test_zero_denominator(self) -> None:
        """Should return 0 when denominator is zero."""
        assert _safe_divide(Decimal("100"), Decimal("0")) == Decimal("0")

    def test_int_denominator(self) -> None:
        """Should accept int denominator."""
        assert _safe_divide(Decimal("120"), 12) == Decimal("10")


# ---------------------------------------------------------------------------
# build_monthly_overview
# ---------------------------------------------------------------------------


class TestBuildMonthlyOverview:
    """Tests for build_monthly_overview function."""

    def test_normal_case(self) -> None:
        """Should calculate monthly income, savings, and rate."""
        income = {"total_income": "600000"}
        expense = {"total_amount": "35000"}
        result = build_monthly_overview(income, expense)

        assert isinstance(result, MonthlyOverviewSection)
        assert result.total_income == Decimal("50000")
        assert result.total_expenses == Decimal("35000")
        assert result.net_savings == Decimal("15000")
        assert result.savings_rate == Decimal("0.3")

    def test_zero_income(self) -> None:
        """Should handle zero income gracefully."""
        result = build_monthly_overview({"total_income": "0"}, {"total_amount": "5000"})
        assert result.total_income == Decimal("0")
        assert result.savings_rate == Decimal("0")

    def test_empty_data(self) -> None:
        """Should handle empty dicts."""
        result = build_monthly_overview({}, {})
        assert result.total_income == Decimal("0")
        assert result.total_expenses == Decimal("0")


# ---------------------------------------------------------------------------
# build_expense_breakdown
# ---------------------------------------------------------------------------


class TestBuildExpenseBreakdown:
    """Tests for build_expense_breakdown function."""

    def test_with_categories(self) -> None:
        """Should enrich categories with percentages."""
        data = {
            "total_amount": "10000",
            "transaction_count": 5,
            "category_breakdown": [
                {"category": "food", "amount": "6000"},
                {"category": "transport", "amount": "4000"},
            ],
        }
        result = build_expense_breakdown(data)

        assert isinstance(result, ExpenseBreakdownSection)
        assert result.total_amount == Decimal("10000")
        assert result.transaction_count == 5
        assert len(result.categories) == 2
        assert result.categories[0]["percentage"] == "60.00"

    def test_empty_data(self) -> None:
        """Should handle empty expense data."""
        result = build_expense_breakdown({})
        assert result.total_amount == Decimal("0")
        assert result.categories == []


# ---------------------------------------------------------------------------
# build_investment_portfolio
# ---------------------------------------------------------------------------


class TestBuildInvestmentPortfolio:
    """Tests for build_investment_portfolio function."""

    def test_with_gain(self) -> None:
        """Should calculate positive gain/loss percentage."""
        data = {
            "total_value": "550000",
            "total_cost": "500000",
            "total_gain_loss": "50000",
            "holding_count": 3,
            "holdings": [{"symbol": "PTT.BK"}],
        }
        result = build_investment_portfolio(data)

        assert isinstance(result, InvestmentPortfolioSection)
        assert result.gain_loss_percentage == Decimal("10.00")
        assert result.holding_count == 3

    def test_zero_cost(self) -> None:
        """Should handle zero cost (no holdings)."""
        result = build_investment_portfolio({"total_cost": "0"})
        assert result.gain_loss_percentage == Decimal("0.00")

    def test_empty_data(self) -> None:
        """Should handle empty dict."""
        result = build_investment_portfolio({})
        assert result.total_value == Decimal("0")


# ---------------------------------------------------------------------------
# build_goal_progress
# ---------------------------------------------------------------------------


class TestBuildGoalProgress:
    """Tests for build_goal_progress function."""

    def test_with_goals(self) -> None:
        """Should calculate completed goals."""
        data = {
            "total_goals": 5,
            "active_goals": 3,
            "overall_percentage": "55.00",
            "goals": [{"name": "Emergency Fund"}],
        }
        result = build_goal_progress(data)

        assert isinstance(result, GoalProgressSection)
        assert result.completed_goals == 2
        assert result.overall_percentage == Decimal("55.00")

    def test_empty_data(self) -> None:
        """Should handle empty dict."""
        result = build_goal_progress({})
        assert result.total_goals == 0
        assert result.completed_goals == 0


# ---------------------------------------------------------------------------
# build_tax_status
# ---------------------------------------------------------------------------


class TestBuildTaxStatus:
    """Tests for build_tax_status function."""

    def test_with_data(self) -> None:
        """Should map tax data correctly."""
        data = {
            "status": "filed",
            "gross_income": "600000",
            "deductions": "160000",
            "tax_due": "29000",
            "effective_rate": "4.83",
        }
        result = build_tax_status(data, 2026)

        assert isinstance(result, TaxStatusSection)
        assert result.status == "filed"
        assert result.tax_year == 2026
        assert result.total_tax == Decimal("29000")

    def test_empty_data(self) -> None:
        """Should handle empty dict."""
        result = build_tax_status({}, 2026)
        assert result.status == "not_filed"
        assert result.gross_income == Decimal("0")


# ---------------------------------------------------------------------------
# Highlights
# ---------------------------------------------------------------------------


class TestCheckSavingsHighlight:
    """Tests for _check_savings_highlight."""

    def test_good_savings_rate(self) -> None:
        """Should return achievement for savings >= 20%."""
        overview = MonthlyOverviewSection(savings_rate=Decimal("0.25"))
        highlights = _check_savings_highlight(overview)
        assert len(highlights) == 1
        assert highlights[0].highlight_type == "achievement"

    def test_low_savings_rate(self) -> None:
        """Should return nothing for savings < 20%."""
        overview = MonthlyOverviewSection(savings_rate=Decimal("0.10"))
        assert _check_savings_highlight(overview) == []


class TestCheckGoalHighlights:
    """Tests for _check_goal_highlights."""

    def test_near_complete_goal(self) -> None:
        """Should highlight goals >= 80%."""
        data = {"goals": [{"name": "Emergency", "percentage": "85"}]}
        highlights = _check_goal_highlights(data)
        assert len(highlights) == 1
        assert "Emergency" in highlights[0].title

    def test_no_near_complete(self) -> None:
        """Should return nothing for low-progress goals."""
        data = {"goals": [{"name": "Car", "percentage": "30"}]}
        assert _check_goal_highlights(data) == []


class TestCheckPortfolioHighlight:
    """Tests for _check_portfolio_highlight."""

    def test_portfolio_loss(self) -> None:
        """Should warn on portfolio loss."""
        data = {"total_gain_loss": "-50000"}
        highlights = _check_portfolio_highlight(data)
        assert len(highlights) == 1
        assert highlights[0].highlight_type == "warning"

    def test_portfolio_gain(self) -> None:
        """Should return nothing for positive portfolio."""
        data = {"total_gain_loss": "50000"}
        assert _check_portfolio_highlight(data) == []


class TestCheckRecommendationHighlights:
    """Tests for _check_recommendation_highlights."""

    def test_high_priority_recommendations(self) -> None:
        """Should convert priority >= 4 recommendations to warnings."""
        rec = MagicMock()
        rec.priority = 4
        rec.title = "ค่าใช้จ่ายสูง"
        rec.description = "ค่าใช้จ่ายเกิน 80%"
        report = MagicMock()
        report.recommendations = [rec]

        highlights = _check_recommendation_highlights(report)
        assert len(highlights) == 1
        assert highlights[0].highlight_type == "warning"

    def test_low_priority_skipped(self) -> None:
        """Should skip priority < 4 recommendations."""
        rec = MagicMock()
        rec.priority = 2
        report = MagicMock()
        report.recommendations = [rec]

        assert _check_recommendation_highlights(report) == []


# ---------------------------------------------------------------------------
# generate_financial_report (integration)
# ---------------------------------------------------------------------------


class TestGenerateFinancialReport:
    """Tests for generate_financial_report main function."""

    @patch("finance_ai.tools.report_service._get_recommendations")
    @patch("finance_ai.tools.report_service._gather_data")
    def test_returns_complete_report(self, mock_gather: MagicMock, mock_recs: MagicMock) -> None:
        """Should assemble a complete FinancialReport."""
        mock_gather.return_value = {
            "income": {"total_income": "600000"},
            "expense": {
                "total_amount": "35000",
                "transaction_count": 10,
                "category_breakdown": [],
            },
            "portfolio": {
                "total_value": "500000",
                "total_cost": "450000",
                "total_gain_loss": "50000",
                "holding_count": 2,
                "holdings": [],
            },
            "goals": {
                "total_goals": 2,
                "active_goals": 1,
                "overall_percentage": "60",
                "goals": [],
            },
            "tax": {
                "status": "filed",
                "gross_income": "600000",
                "deductions": "160000",
                "tax_due": "29000",
                "effective_rate": "4.83",
            },
        }
        mock_rec_report = MagicMock()
        mock_rec_report.health_score = 85
        mock_rec_report.recommendations = []
        mock_recs.return_value = mock_rec_report

        session = MagicMock()
        report = generate_financial_report(session, "user-1", 2026, 3)

        assert isinstance(report, FinancialReport)
        assert report.user_id == "user-1"
        assert report.year == 2026
        assert report.month == 3
        assert report.health_score == 85
        assert report.monthly_overview.total_income == Decimal("50000")
        assert report.tax_status.status == "filed"

    @patch("finance_ai.tools.report_service._get_recommendations")
    @patch("finance_ai.tools.report_service._gather_data")
    def test_empty_data(self, mock_gather: MagicMock, mock_recs: MagicMock) -> None:
        """Should handle all-empty data gracefully."""
        mock_gather.return_value = {
            "income": {},
            "expense": {},
            "portfolio": {},
            "goals": {},
            "tax": {},
        }
        mock_rec_report = MagicMock()
        mock_rec_report.health_score = 100
        mock_rec_report.recommendations = []
        mock_recs.return_value = mock_rec_report

        session = MagicMock()
        report = generate_financial_report(session, "user-2", 2026, 1)

        assert report.health_score == 100
        assert report.monthly_overview.total_income == Decimal("0")
