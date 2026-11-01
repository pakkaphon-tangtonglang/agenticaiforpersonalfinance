"""Tests for GET /dashboard/trends (insight-card data)."""

from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from finance_ai.main import app
from finance_ai.tools.trend_models import DashboardTrends, MonthlySpending

BUILD_PATH = "finance_ai.main.build_dashboard_trends"


def empty_trends() -> DashboardTrends:
    """A payload with one month of spending and nothing else."""
    return DashboardTrends(
        spending=[MonthlySpending(year=2026, month=3, total=Decimal("1250.50"))],
        average_spending=Decimal("0.00"),
        spikes=[],
        assets=[],
        benchmark=None,
    )


class TestDashboardTrendsEndpoint:
    """The endpoint forwards the month and serialises money as strings."""

    @patch(BUILD_PATH)
    def test_explicit_month(self, build: MagicMock) -> None:
        """year/month are passed through; Decimals serialise as strings."""
        build.return_value = empty_trends()

        response = TestClient(app).get(
            "/dashboard/trends", params={"user_id": "trend-user-1", "year": 2026, "month": 3}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["spending"] == [{"year": 2026, "month": 3, "total": "1250.50"}]
        assert body["benchmark"] is None
        assert build.call_args.args[1:] == ("trend-user-1", 2026, 3)

    @patch(BUILD_PATH)
    def test_defaults_to_current_month(self, build: MagicMock) -> None:
        """Without year/month the current month is used."""
        build.return_value = empty_trends()
        today = date.today()

        TestClient(app).get("/dashboard/trends", params={"user_id": "trend-user-1"})

        assert build.call_args.args[2:] == (today.year, today.month)

    def test_invalid_month_is_rejected(self) -> None:
        """month must be 1-12."""
        response = TestClient(app).get(
            "/dashboard/trends", params={"user_id": "trend-user-1", "month": 13}
        )

        assert response.status_code == 422
