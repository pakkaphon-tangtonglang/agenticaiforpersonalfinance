"""Tests for the InvestmentHolding database model."""

from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from finance_ai.database.models.investment_holding import InvestmentHolding
from finance_ai.database.models.user import User


class TestInvestmentHoldingModel:
    """Tests for InvestmentHolding model creation and relationships."""

    def test_create_stock_holding(self, test_session: Session, sample_user: User) -> None:
        """Test creating a stock holding with Thai symbol format."""
        holding = InvestmentHolding(
            user_id=sample_user.id,
            asset_type="stock",
            symbol="PTT.BK",
            name="PTT Public Company Limited",
            quantity=Decimal("100.0000"),
            average_cost_per_unit=Decimal("35.5000"),
            total_cost=Decimal("3550.00"),
            purchase_date=date(2024, 1, 15),
        )
        test_session.add(holding)
        test_session.commit()
        test_session.refresh(holding)
        assert holding.id is not None
        assert holding.symbol == "PTT.BK"
        assert holding.quantity == Decimal("100.0000")

    def test_create_mutual_fund_holding(self, test_session: Session, sample_user: User) -> None:
        """Test creating a mutual fund holding with fractional units."""
        holding = InvestmentHolding(
            user_id=sample_user.id,
            asset_type="mutual_fund",
            symbol="K-CHINA",
            name="K China Equity Fund",
            quantity=Decimal("1234.5678"),
            average_cost_per_unit=Decimal("10.2345"),
            total_cost=Decimal("12634.56"),
            purchase_date=date(2024, 3, 1),
        )
        test_session.add(holding)
        test_session.commit()
        test_session.refresh(holding)
        assert holding.quantity == Decimal("1234.5678")

    def test_holding_optional_market_fields(self, test_session: Session, sample_user: User) -> None:
        """Test that market price fields are optional."""
        holding = InvestmentHolding(
            user_id=sample_user.id,
            asset_type="stock",
            symbol="AOT.BK",
            quantity=Decimal("50.0000"),
            average_cost_per_unit=Decimal("65.0000"),
            total_cost=Decimal("3250.00"),
            purchase_date=date(2024, 6, 1),
        )
        test_session.add(holding)
        test_session.commit()
        test_session.refresh(holding)
        assert holding.current_price_per_unit is None
        assert holding.current_value is None
        assert holding.unrealized_gain_loss is None
        assert holding.last_price_update is None

    def test_holding_relationship_to_user(self, test_session: Session, sample_user: User) -> None:
        """Test that holding links back to its user."""
        holding = InvestmentHolding(
            user_id=sample_user.id,
            asset_type="bond",
            symbol="TH-BOND-5Y",
            quantity=Decimal("10.0000"),
            average_cost_per_unit=Decimal("1000.0000"),
            total_cost=Decimal("10000.00"),
            purchase_date=date(2024, 1, 1),
        )
        test_session.add(holding)
        test_session.commit()
        test_session.refresh(holding)
        assert holding.user.id == sample_user.id
        assert holding in sample_user.investment_holdings
