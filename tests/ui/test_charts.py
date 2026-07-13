"""Tests for finance_ai.ui.charts — pure chart builder functions."""

from decimal import Decimal

import plotly.graph_objects as go
import pytest

from finance_ai.ui.charts import (
    _score_color,
    _thai_label,
    _to_float,
    create_expense_pie_chart,
    create_goal_progress_bar,
    create_health_gauge,
    create_income_vs_expense_bar,
    create_portfolio_pie_chart,
)


class TestToFloat:
    """Tests for _to_float helper."""

    def test_converts_decimal(self) -> None:
        """Decimal is converted to float."""
        assert _to_float(Decimal("123.45")) == 123.45

    def test_converts_string(self) -> None:
        """String number is converted to float."""
        assert _to_float("99.5") == 99.5

    def test_converts_int(self) -> None:
        """Integer is converted to float."""
        assert _to_float(42) == 42.0

    def test_passthrough_float(self) -> None:
        """Float passes through unchanged."""
        assert _to_float(3.14) == 3.14


class TestThaiLabel:
    """Tests for _thai_label helper."""

    def test_known_category(self) -> None:
        """Known category returns Thai label."""
        assert _thai_label("food") == "อาหาร"

    def test_unknown_category(self) -> None:
        """Unknown category returns the key itself."""
        assert _thai_label("misc") == "misc"


class TestScoreColor:
    """Tests for _score_color helper."""

    def test_high_score_green(self) -> None:
        """Score >= 70 returns green."""
        assert _score_color(85) == "#27AE60"

    def test_medium_score_orange(self) -> None:
        """Score 40-69 returns orange."""
        assert _score_color(55) == "#E67E22"

    def test_low_score_red(self) -> None:
        """Score < 40 returns red."""
        assert _score_color(20) == "#E74C3C"


class TestCreateExpensePieChart:
    """Tests for create_expense_pie_chart."""

    def test_returns_figure(self) -> None:
        """Function returns a Plotly Figure."""
        categories = [
            {"category": "food", "amount": "15000"},
            {"category": "transport", "amount": "5000"},
        ]
        fig = create_expense_pie_chart(categories)
        assert isinstance(fig, go.Figure)

    def test_empty_categories(self) -> None:
        """Empty list produces a figure without error."""
        fig = create_expense_pie_chart([])
        assert isinstance(fig, go.Figure)

    def test_has_pie_trace(self) -> None:
        """Figure contains a Pie trace."""
        categories = [{"category": "food", "amount": "1000"}]
        fig = create_expense_pie_chart(categories)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Pie)


class TestCreatePortfolioPieChart:
    """Tests for create_portfolio_pie_chart."""

    def test_returns_figure(self) -> None:
        """Function returns a Plotly Figure."""
        holdings = [
            {"symbol": "PTT.BK", "current_value": "100000"},
            {"symbol": "KBANK.BK", "current_value": "80000"},
        ]
        fig = create_portfolio_pie_chart(holdings)
        assert isinstance(fig, go.Figure)

    def test_empty_holdings(self) -> None:
        """Empty list produces a figure."""
        fig = create_portfolio_pie_chart([])
        assert isinstance(fig, go.Figure)


class TestCreateGoalProgressBar:
    """Tests for create_goal_progress_bar."""

    def test_returns_figure(self) -> None:
        """Function returns a Plotly Figure."""
        goals = [
            {"name": "เกษียณ", "percentage": "45"},
            {"name": "ซื้อบ้าน", "percentage": "80"},
        ]
        fig = create_goal_progress_bar(goals)
        assert isinstance(fig, go.Figure)

    def test_caps_at_100(self) -> None:
        """Percentage is capped at 100."""
        goals = [{"name": "test", "percentage": "150"}]
        fig = create_goal_progress_bar(goals)
        bar = fig.data[0]
        assert bar.x[0] == 100.0

    def test_empty_goals(self) -> None:
        """Empty list produces a figure."""
        fig = create_goal_progress_bar([])
        assert isinstance(fig, go.Figure)


class TestCreateHealthGauge:
    """Tests for create_health_gauge."""

    def test_returns_figure(self) -> None:
        """Function returns a Plotly Figure."""
        fig = create_health_gauge(75)
        assert isinstance(fig, go.Figure)

    def test_indicator_value(self) -> None:
        """Indicator shows the correct score value."""
        fig = create_health_gauge(42)
        indicator = fig.data[0]
        assert indicator.value == 42

    @pytest.mark.parametrize("score", [0, 50, 100])
    def test_boundary_scores(self, score: int) -> None:
        """Boundary scores create valid figures."""
        fig = create_health_gauge(score)
        assert isinstance(fig, go.Figure)


class TestCreateIncomeVsExpenseBar:
    """Tests for create_income_vs_expense_bar."""

    def test_returns_figure(self) -> None:
        """Function returns a Plotly Figure."""
        fig = create_income_vs_expense_bar(Decimal("50000"), Decimal("35000"))
        assert isinstance(fig, go.Figure)

    def test_three_bars(self) -> None:
        """Figure has 3 bars (income, expense, savings)."""
        fig = create_income_vs_expense_bar(Decimal("50000"), Decimal("35000"))
        bar = fig.data[0]
        assert len(bar.y) == 3

    def test_savings_calculated(self) -> None:
        """Third bar shows income - expenses."""
        fig = create_income_vs_expense_bar(Decimal("50000"), Decimal("35000"))
        bar = fig.data[0]
        assert bar.y[2] == 15000.0

    def test_zero_values(self) -> None:
        """Zero values produce a valid figure."""
        fig = create_income_vs_expense_bar(Decimal("0"), Decimal("0"))
        assert isinstance(fig, go.Figure)
