"""Tests for the portfolio (holdings) tool — not the watchlist."""

from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any

from sqlalchemy.orm import Session

from finance_ai.agents.portfolio_tools import get_portfolio_holdings
from finance_ai.database.models.investment_holding import InvestmentHolding
from finance_ai.database.models.user import User


def _invoke(user: User, factory: Callable[[], Session]) -> dict[str, Any]:
    result: dict[str, Any] = get_portfolio_holdings.invoke(
        {"user_id": user.id, "db_session_factory": factory}
    )
    return result


def _add_holding(session: Session, user: User, price: Decimal | None) -> None:
    session.add(
        InvestmentHolding(
            user_id=user.id,
            asset_type="stock",
            symbol="PTT.BK",
            name="PTT",
            quantity=Decimal("100"),
            average_cost_per_unit=Decimal("35.50"),
            total_cost=Decimal("3550.00"),
            purchase_date=date(2025, 1, 15),
            current_price_per_unit=price,
            current_value=None if price is None else price * 100,
            unrealized_gain_loss=None if price is None else price * 100 - Decimal("3550"),
        )
    )
    session.commit()


class TestGetPortfolioHoldings:
    """get_portfolio_holdings reads what the user owns."""

    def test_empty_portfolio_says_so(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """No holdings: count 0 and a message (never the watchlist)."""
        result = _invoke(sample_user, db_session_factory)
        assert result["holding_count"] == 0
        assert result["holdings"] == []
        assert "ยังไม่มี" in str(result["message"])

    def test_lists_holdings_with_gain(
        self,
        sample_user: User,
        test_session: Session,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Each holding has quantity, cost, value and gain/loss as strings."""
        _add_holding(test_session, sample_user, Decimal("42"))
        result = _invoke(sample_user, db_session_factory)
        assert result["holding_count"] == 1
        holding = result["holdings"][0]
        assert holding["symbol"] == "PTT.BK"
        assert holding["quantity"] == "100"
        assert holding["total_cost"] == "3550.00"
        assert holding["current_value"] == "4200.00"
        assert holding["unrealized_gain_loss"] == "650.00"
        assert result["total_current_value"] == "4200.00"

    def test_missing_price_is_none(
        self,
        sample_user: User,
        test_session: Session,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """A holding without a stored price reports None, not a made-up value."""
        _add_holding(test_session, sample_user, None)
        holding = _invoke(sample_user, db_session_factory)["holdings"][0]
        assert holding["current_value"] is None
        assert holding["unrealized_gain_loss"] is None
