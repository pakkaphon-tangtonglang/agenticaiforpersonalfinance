"""Tests for the Transaction database model."""

from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from finance_ai.database.models.investment_holding import InvestmentHolding
from finance_ai.database.models.transaction import Transaction
from finance_ai.database.models.user import User


class TestTransactionModel:
    """Tests for Transaction model creation and relationships."""

    def test_create_buy_transaction(self, test_session: Session, sample_user: User) -> None:
        """Test creating a buy transaction linked to a holding."""
        holding = InvestmentHolding(
            user_id=sample_user.id,
            asset_type="stock",
            symbol="PTT.BK",
            quantity=Decimal("100.0000"),
            average_cost_per_unit=Decimal("35.0000"),
            total_cost=Decimal("3500.00"),
            purchase_date=date(2024, 1, 15),
        )
        test_session.add(holding)
        test_session.commit()
        transaction = Transaction(
            user_id=sample_user.id,
            holding_id=holding.id,
            transaction_type="buy",
            amount=Decimal("3500.00"),
            quantity=Decimal("100.0000"),
            price_per_unit=Decimal("35.0000"),
            transaction_date=date(2024, 1, 15),
        )
        test_session.add(transaction)
        test_session.commit()
        test_session.refresh(transaction)
        assert transaction.id is not None
        assert transaction.transaction_type == "buy"
        assert transaction.holding_id == holding.id

    def test_create_expense_transaction(self, test_session: Session, sample_user: User) -> None:
        """Test creating an expense transaction without a holding."""
        transaction = Transaction(
            user_id=sample_user.id,
            transaction_type="expense",
            category="food",
            description="Lunch",
            amount=Decimal("250.00"),
            transaction_date=date(2024, 6, 15),
        )
        test_session.add(transaction)
        test_session.commit()
        test_session.refresh(transaction)
        assert transaction.holding_id is None
        assert transaction.category == "food"
        assert transaction.quantity is None

    def test_transaction_relationship_to_user(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Test that transaction links back to its user."""
        transaction = Transaction(
            user_id=sample_user.id,
            transaction_type="income",
            amount=Decimal("80000.00"),
            transaction_date=date(2024, 7, 1),
        )
        test_session.add(transaction)
        test_session.commit()
        test_session.refresh(transaction)
        assert transaction.user.id == sample_user.id
        assert transaction in sample_user.transactions

    def test_transaction_relationship_to_holding(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Test that transaction links to its holding."""
        holding = InvestmentHolding(
            user_id=sample_user.id,
            asset_type="stock",
            symbol="AOT.BK",
            quantity=Decimal("50.0000"),
            average_cost_per_unit=Decimal("65.0000"),
            total_cost=Decimal("3250.00"),
            purchase_date=date(2024, 1, 1),
        )
        test_session.add(holding)
        test_session.commit()
        transaction = Transaction(
            user_id=sample_user.id,
            holding_id=holding.id,
            transaction_type="sell",
            amount=Decimal("3500.00"),
            quantity=Decimal("50.0000"),
            price_per_unit=Decimal("70.0000"),
            transaction_date=date(2024, 8, 1),
        )
        test_session.add(transaction)
        test_session.commit()
        test_session.refresh(transaction)
        assert transaction.holding is not None
        assert transaction.holding.symbol == "AOT.BK"
