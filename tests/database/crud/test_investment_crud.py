"""Tests for InvestmentHoldingCRUD operations."""

from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from finance_ai.database.crud.investment_crud import InvestmentHoldingCRUD
from finance_ai.database.models.user import User


class TestInvestmentHoldingCRUD:
    """Tests for InvestmentHolding-specific CRUD operations."""

    def _create_holding(
        self,
        crud: InvestmentHoldingCRUD,
        session: Session,
        user: User,
        symbol: str = "PTT.BK",
        asset_type: str = "stock",
    ) -> None:
        """Helper to create a test holding."""
        crud.create(
            session,
            user_id=user.id,
            asset_type=asset_type,
            symbol=symbol,
            quantity=Decimal("100.0000"),
            average_cost_per_unit=Decimal("35.0000"),
            total_cost=Decimal("3500.00"),
            purchase_date=date(2024, 1, 1),
        )

    def test_get_by_user(self, test_session: Session, sample_user: User) -> None:
        """Test getting all holdings for a user."""
        crud = InvestmentHoldingCRUD()
        self._create_holding(crud, test_session, sample_user, "PTT.BK")
        self._create_holding(crud, test_session, sample_user, "AOT.BK")
        holdings = crud.get_by_user(test_session, sample_user.id)
        assert len(holdings) == 2

    def test_get_by_symbol(self, test_session: Session, sample_user: User) -> None:
        """Test finding a holding by symbol."""
        crud = InvestmentHoldingCRUD()
        self._create_holding(crud, test_session, sample_user, "PTT.BK")
        found = crud.get_by_symbol(test_session, sample_user.id, "PTT.BK")
        assert found is not None
        assert found.symbol == "PTT.BK"

    def test_get_by_symbol_not_found(self, test_session: Session, sample_user: User) -> None:
        """Test that nonexistent symbol returns None."""
        crud = InvestmentHoldingCRUD()
        found = crud.get_by_symbol(test_session, sample_user.id, "NONE.BK")
        assert found is None

    def test_get_by_asset_type(self, test_session: Session, sample_user: User) -> None:
        """Test filtering holdings by asset type."""
        crud = InvestmentHoldingCRUD()
        self._create_holding(crud, test_session, sample_user, "PTT.BK", "stock")
        self._create_holding(crud, test_session, sample_user, "K-CHINA", "mutual_fund")
        stocks = crud.get_by_asset_type(test_session, sample_user.id, "stock")
        assert len(stocks) == 1
        assert stocks[0].symbol == "PTT.BK"

    def test_update_market_price(self, test_session: Session, sample_user: User) -> None:
        """Test updating market price recalculates values."""
        crud = InvestmentHoldingCRUD()
        self._create_holding(crud, test_session, sample_user, "PTT.BK")
        holding = crud.get_by_symbol(test_session, sample_user.id, "PTT.BK")
        assert holding is not None
        updated = crud.update_market_price(test_session, holding.id, Decimal("40.0000"))
        assert updated is not None
        assert updated.current_price_per_unit == Decimal("40.0000")
        assert updated.current_value == Decimal("4000.0000")
        assert updated.unrealized_gain_loss == Decimal("500.00")
        assert updated.last_price_update is not None

    def test_update_market_price_not_found(self, test_session: Session) -> None:
        """Test that updating price of nonexistent holding returns None."""
        crud = InvestmentHoldingCRUD()
        result = crud.update_market_price(test_session, "fake-id", Decimal("50.0000"))
        assert result is None
