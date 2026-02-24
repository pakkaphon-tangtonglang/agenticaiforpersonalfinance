"""Tests for TransactionCRUD operations."""

from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from finance_ai.database.crud.transaction_crud import TransactionCRUD
from finance_ai.database.models.investment_holding import InvestmentHolding
from finance_ai.database.models.user import User


class TestTransactionCRUD:
    """Tests for Transaction-specific CRUD operations."""

    def test_get_by_user_and_date_range(self, test_session: Session, sample_user: User) -> None:
        """Test filtering transactions by date range."""
        crud = TransactionCRUD()
        crud.create(
            test_session,
            user_id=sample_user.id,
            transaction_type="expense",
            amount=Decimal("500.00"),
            transaction_date=date(2024, 6, 15),
        )
        crud.create(
            test_session,
            user_id=sample_user.id,
            transaction_type="expense",
            amount=Decimal("300.00"),
            transaction_date=date(2024, 7, 1),
        )
        crud.create(
            test_session,
            user_id=sample_user.id,
            transaction_type="expense",
            amount=Decimal("200.00"),
            transaction_date=date(2024, 8, 1),
        )
        results = crud.get_by_user_and_date_range(
            test_session,
            sample_user.id,
            date(2024, 6, 1),
            date(2024, 7, 31),
        )
        assert len(results) == 2

    def test_get_by_holding(self, test_session: Session, sample_user: User) -> None:
        """Test filtering transactions by investment holding."""
        holding = InvestmentHolding(
            user_id=sample_user.id,
            asset_type="stock",
            symbol="PTT.BK",
            quantity=Decimal("100.0000"),
            average_cost_per_unit=Decimal("35.0000"),
            total_cost=Decimal("3500.00"),
            purchase_date=date(2024, 1, 1),
        )
        test_session.add(holding)
        test_session.commit()
        crud = TransactionCRUD()
        crud.create(
            test_session,
            user_id=sample_user.id,
            holding_id=holding.id,
            transaction_type="buy",
            amount=Decimal("3500.00"),
            transaction_date=date(2024, 1, 1),
        )
        results = crud.get_by_holding(test_session, holding.id)
        assert len(results) == 1
        assert results[0].transaction_type == "buy"

    def test_get_by_category(self, test_session: Session, sample_user: User) -> None:
        """Test filtering transactions by category."""
        crud = TransactionCRUD()
        crud.create(
            test_session,
            user_id=sample_user.id,
            transaction_type="expense",
            category="food",
            amount=Decimal("250.00"),
            transaction_date=date(2024, 6, 15),
        )
        crud.create(
            test_session,
            user_id=sample_user.id,
            transaction_type="expense",
            category="transport",
            amount=Decimal("100.00"),
            transaction_date=date(2024, 6, 15),
        )
        food_txns = crud.get_by_category(test_session, sample_user.id, "food")
        assert len(food_txns) == 1
        assert food_txns[0].amount == Decimal("250.00")

    def test_get_by_user_type_and_date_range(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Test filtering transactions by user, type, and date range."""
        crud = TransactionCRUD()
        crud.create(
            test_session,
            user_id=sample_user.id,
            transaction_type="expense",
            amount=Decimal("100.00"),
            transaction_date=date(2024, 6, 15),
        )
        crud.create(
            test_session,
            user_id=sample_user.id,
            transaction_type="income",
            amount=Decimal("50000.00"),
            transaction_date=date(2024, 6, 20),
        )
        crud.create(
            test_session,
            user_id=sample_user.id,
            transaction_type="expense",
            amount=Decimal("200.00"),
            transaction_date=date(2024, 8, 1),
        )
        results = crud.get_by_user_type_and_date_range(
            test_session,
            sample_user.id,
            "expense",
            date(2024, 6, 1),
            date(2024, 7, 31),
        )
        assert len(results) == 1
        assert results[0].amount == Decimal("100.00")

    def test_get_by_user_type_and_date_range_empty(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Test returns empty list when no transactions match."""
        crud = TransactionCRUD()
        results = crud.get_by_user_type_and_date_range(
            test_session,
            sample_user.id,
            "expense",
            date(2024, 1, 1),
            date(2024, 12, 31),
        )
        assert results == []

    def test_get_by_user_category_and_date_range(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Test filtering by user, category, and date range."""
        crud = TransactionCRUD()
        crud.create(
            test_session,
            user_id=sample_user.id,
            transaction_type="expense",
            category="food",
            amount=Decimal("80.00"),
            transaction_date=date(2024, 6, 10),
        )
        crud.create(
            test_session,
            user_id=sample_user.id,
            transaction_type="expense",
            category="transport",
            amount=Decimal("50.00"),
            transaction_date=date(2024, 6, 15),
        )
        crud.create(
            test_session,
            user_id=sample_user.id,
            transaction_type="expense",
            category="food",
            amount=Decimal("120.00"),
            transaction_date=date(2024, 8, 1),
        )
        results = crud.get_by_user_category_and_date_range(
            test_session,
            sample_user.id,
            "food",
            date(2024, 6, 1),
            date(2024, 7, 31),
        )
        assert len(results) == 1
        assert results[0].amount == Decimal("80.00")

    def test_get_by_user_category_and_date_range_empty(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Test returns empty list when no category transactions match."""
        crud = TransactionCRUD()
        results = crud.get_by_user_category_and_date_range(
            test_session,
            sample_user.id,
            "food",
            date(2024, 1, 1),
            date(2024, 12, 31),
        )
        assert results == []
