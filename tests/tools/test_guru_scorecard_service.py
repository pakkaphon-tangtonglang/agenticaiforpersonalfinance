"""Tests for the deterministic guru scorecard."""

from decimal import Decimal
from typing import Any

import pytest

from finance_ai.tools.guru_scorecard_service import (
    build_guru_scorecard,
    emergency_fund_months,
    equity_share,
    investing_rate,
    largest_holding_share,
    savings_rate,
)


def _financial_data(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    income: str = "50000",
    expense: str = "35000",
    invested: str = "5000",
    emergency_saved: str = "120000",
    holdings: list[dict[str, str]] | None = None,
    goal_types: tuple[str, ...] = ("emergency_fund", "retirement"),
) -> dict[str, Any]:
    """Build gathered financial data in the cross_agent_service shape."""
    holdings = (
        holdings
        if holdings is not None
        else [
            {"asset_type": "mutual_fund", "current_value": "60000"},
            {"asset_type": "stock", "current_value": "40000"},
        ]
    )
    total_value = sum(Decimal(holding["current_value"]) for holding in holdings)
    goals = [
        {
            "goal_type": goal_type,
            "current_amount": emergency_saved if goal_type == "emergency_fund" else "0",
            "target_amount": "300000",
            "is_completed": False,
        }
        for goal_type in goal_types
    ]
    return {
        "income_monthly": {"total_income": income},
        "expense": {
            "total_amount": expense,
            "category_breakdown": [{"category": "investment", "amount": invested}],
        },
        "goals": {"goals": goals},
        "portfolio": {"total_value": str(total_value), "holdings": holdings},
    }


class TestMetrics:
    """Tests for the shared Decimal metrics."""

    def test_savings_rate(self) -> None:
        """(income - expense) / income."""
        assert savings_rate(_financial_data()) == Decimal("0.3")

    def test_savings_rate_without_income_is_none(self) -> None:
        """No income means the rate is unknown, not zero."""
        assert savings_rate(_financial_data(income="0")) is None

    def test_emergency_fund_months(self) -> None:
        """Saved emergency money divided by monthly expense."""
        assert emergency_fund_months(_financial_data()) == Decimal("120000") / Decimal("35000")

    def test_emergency_fund_months_without_expenses_is_none(self) -> None:
        """No expense data means coverage cannot be measured."""
        assert emergency_fund_months(_financial_data(expense="0")) is None

    def test_investing_rate_uses_investment_category(self) -> None:
        """Money spent in the investment category over income."""
        assert investing_rate(_financial_data()) == Decimal("0.1")

    def test_largest_holding_share(self) -> None:
        """Biggest single holding over portfolio value."""
        assert largest_holding_share(_financial_data()) == Decimal("0.6")

    def test_equity_share(self) -> None:
        """Stock value over portfolio value."""
        assert equity_share(_financial_data()) == Decimal("0.4")

    def test_portfolio_metrics_without_holdings_are_none(self) -> None:
        """An empty portfolio gives None, never a division error."""
        data = _financial_data(holdings=[])

        assert largest_holding_share(data) is None
        assert equity_share(data) is None


class TestBuildGuruScorecard:
    """Tests for build_guru_scorecard."""

    @pytest.mark.parametrize("guru_id", ["buffett", "dalio", "niwes", "coach_noom"])
    def test_every_guru_has_checks(self, guru_id: str) -> None:
        """Each persona gets at least three checks with known statuses."""
        scorecard = build_guru_scorecard(guru_id, _financial_data(), risk_level=3)

        assert scorecard.guru_id == guru_id
        assert len(scorecard.checks) >= 3
        assert {check.status for check in scorecard.checks} <= {"good", "warn", "bad", "missing"}

    def test_coach_noom_flags_low_savings_and_thin_emergency_fund(self) -> None:
        """5% savings and 1 month of cover are both bad for โค้ชหนุ่ม."""
        data = _financial_data(income="40000", expense="38000", emergency_saved="38000")

        checks = {check.key: check for check in build_guru_scorecard("coach_noom", data).checks}

        assert checks["savings_rate"].status == "bad"
        assert checks["savings_rate"].value == "5.00%"
        assert checks["emergency_fund_months"].status == "bad"

    def test_dalio_flags_concentrated_portfolio(self) -> None:
        """One holding at 90% breaks Dalio's diversification check."""
        data = _financial_data(
            holdings=[
                {"asset_type": "stock", "current_value": "90000"},
                {"asset_type": "stock", "current_value": "10000"},
            ]
        )

        checks = {check.key: check for check in build_guru_scorecard("dalio", data).checks}

        assert checks["largest_holding_share"].status == "bad"
        assert checks["asset_type_count"].status == "warn"

    def test_niwes_equity_share_depends_on_risk_level(self) -> None:
        """High stock share suits a high risk level but not a low one."""
        data = _financial_data(holdings=[{"asset_type": "stock", "current_value": "100000"}])

        aggressive = build_guru_scorecard("niwes", data, risk_level=5).checks
        cautious = build_guru_scorecard("niwes", data, risk_level=1).checks

        assert next(c for c in aggressive if c.key == "equity_share").status == "good"
        assert next(c for c in cautious if c.key == "equity_share").status == "warn"

    def test_missing_data_is_reported_not_guessed(self) -> None:
        """No income and no holdings give 'missing' checks."""
        data = _financial_data(income="0", holdings=[])

        statuses = {
            check.key: check.status for check in build_guru_scorecard("buffett", data).checks
        }

        assert statuses["investing_rate"] == "missing"
        assert statuses["fund_share"] == "missing"

    def test_unknown_guru_raises_value_error(self) -> None:
        """Only registered gurus have scorecards."""
        with pytest.raises(ValueError, match="No scorecard for guru 'elon'"):
            build_guru_scorecard("elon", _financial_data())
