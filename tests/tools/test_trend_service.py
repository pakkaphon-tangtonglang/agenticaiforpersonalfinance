"""Tests for trend_service (dashboard insight cards: spending + price trends).

Expense summaries, holdings and price fetchers are patched where
trend_service looks them up, so no database or network is touched.
"""

# pylint: disable=redefined-outer-name

from datetime import date
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from finance_ai.tools.investment_calculator import HoldingSummary, PortfolioSummaryResult
from finance_ai.tools.price_series_client import PricePoint
from finance_ai.tools.trend_service import (
    BENCHMARK_SYMBOL,
    build_asset_series,
    build_dashboard_trends,
    percent_change,
)

MODULE = "finance_ai.tools.trend_service"


def month(total: str, **amounts: str) -> dict[str, Any]:
    """One month's expense summary."""
    breakdown = [
        {"category": category, "label": f"หมวด-{category}", "amount": amount}
        for category, amount in amounts.items()
    ]
    return {"total_amount": total, "category_breakdown": breakdown}


def points(*values: str) -> list[PricePoint]:
    """A daily series starting 2026-10-01."""
    return [PricePoint(day=date(2026, 10, 1 + i), value=Decimal(v)) for i, v in enumerate(values)]


def holding(symbol: str, asset_type: str, name: str = "") -> HoldingSummary:
    """A minimal holding summary."""
    return HoldingSummary(
        symbol=symbol,
        name=name or symbol,
        asset_type=asset_type,
        asset_type_label="",
        quantity=Decimal("1"),
        average_cost_per_unit=Decimal("1"),
        total_cost=Decimal("1"),
        allocation_percentage=Decimal("100"),
    )


class TestPercentChange:
    """percent_change: first → last point, 2 decimals."""

    def test_rise_and_fall(self) -> None:
        """+10% and -2.5% from the first point."""
        assert percent_change(points("40", "41", "44")) == Decimal("10.00")
        assert percent_change(points("40", "39")) == Decimal("-2.50")

    def test_too_short_or_zero_start_is_none(self) -> None:
        """One point, or a zero start, has no meaningful change."""
        assert percent_change(points("40")) is None
        assert percent_change(points("0", "5")) is None


class TestBuildAssetSeries:
    """Stocks use yfinance, funds use SEC NAV; empty series are dropped."""

    @patch(f"{MODULE}.fetch_nav_series")
    @patch(f"{MODULE}.fetch_price_series")
    def test_routes_by_asset_type(self, price: MagicMock, nav: MagicMock) -> None:
        """PTT.BK → price series, K-USA-A(A) → NAV series."""
        price.return_value = points("40", "44")
        nav.return_value = points("15", "15.3")

        series = build_asset_series(
            [holding("PTT.BK", "stock", "ปตท."), holding("K-USA-A(A)", "mutual_fund")]
        )

        assert [(s.symbol, s.kind, s.name, s.change_pct) for s in series] == [
            ("PTT.BK", "stock", "ปตท.", Decimal("10.00")),
            ("K-USA-A(A)", "mutual_fund", "K-USA-A(A)", Decimal("2.00")),
        ]
        price.assert_called_once_with("PTT.BK", days=30)
        nav.assert_called_once_with("K-USA-A(A)", days=30)

    @patch(f"{MODULE}.fetch_nav_series", return_value=[])
    @patch(f"{MODULE}.fetch_price_series")
    def test_missing_history_is_dropped(self, price: MagicMock, _nav: MagicMock) -> None:
        """No data (rate limit, no SEC key) → the holding has no line."""
        price.return_value = points("40")

        assert not build_asset_series(
            [holding("PTT.BK", "stock"), holding("KFSDIV", "mutual_fund")]
        )

    @patch(f"{MODULE}.fetch_price_series", return_value=points("1", "2"))
    def test_duplicate_symbols_fetched_once(self, price: MagicMock) -> None:
        """Two lots of the same stock share one line."""
        series = build_asset_series([holding("PTT.BK", "stock"), holding("PTT.BK", "stock")])

        assert len(series) == 1
        assert price.call_count == 1


class TestBuildDashboardTrends:
    """The whole payload for the insight cards."""

    @pytest.fixture
    def summaries(self) -> list[dict[str, Any]]:
        """Six months newest first: current month food spike."""
        return [
            month("6000", food="6000"),
            month("4000", food="4000"),
            month("3800", food="3800"),
            month("4200", food="4200"),
            month("0"),
            month("3000", food="3000"),
        ]

    @patch(f"{MODULE}.fetch_price_series", return_value=points("10", "10.5"))
    @patch(f"{MODULE}.get_portfolio_summary")
    @patch(f"{MODULE}.get_expense_summary")
    def test_payload(
        self,
        expense: MagicMock,
        portfolio: MagicMock,
        _price: MagicMock,
        summaries: list[dict[str, Any]],
    ) -> None:
        """Spending months oldest first, average, spikes, assets and benchmark."""
        expense.side_effect = summaries
        portfolio.return_value = PortfolioSummaryResult(
            total_cost=Decimal("1"),
            holdings=[holding("PTT.BK", "stock")],
            asset_type_breakdown=[],
            holding_count=1,
        )

        trends = build_dashboard_trends(MagicMock(), "u1", 2026, 3)

        assert [(m.year, m.month) for m in trends.spending] == [
            (2025, 10),
            (2025, 11),
            (2025, 12),
            (2026, 1),
            (2026, 2),
            (2026, 3),
        ]
        assert trends.spending[-1].total == Decimal("6000")
        assert trends.average_spending == Decimal("3750.00")
        assert [spike.category for spike in trends.spikes] == ["food"]
        assert [asset.symbol for asset in trends.assets] == ["PTT.BK"]
        assert trends.benchmark is not None
        assert trends.benchmark.symbol == BENCHMARK_SYMBOL
        assert trends.benchmark.kind == "benchmark"

    @patch(f"{MODULE}.fetch_price_series")
    @patch(f"{MODULE}.get_portfolio_summary")
    @patch(f"{MODULE}.get_expense_summary", return_value=month("0"))
    def test_new_user_has_no_assets_and_no_benchmark(
        self, _expense: MagicMock, portfolio: MagicMock, price: MagicMock
    ) -> None:
        """Without holdings nothing is fetched from yfinance at all."""
        portfolio.return_value = PortfolioSummaryResult(
            total_cost=Decimal("0"), holdings=[], asset_type_breakdown=[], holding_count=0
        )

        trends = build_dashboard_trends(MagicMock(), "u1", 2026, 3)

        assert not trends.assets
        assert trends.benchmark is None
        assert not trends.spikes
        assert trends.average_spending == Decimal("0.00")
        price.assert_not_called()
