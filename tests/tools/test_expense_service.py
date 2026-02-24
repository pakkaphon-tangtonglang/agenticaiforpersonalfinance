"""Tests for expense service with database integration."""

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session

from finance_ai.database.crud.transaction_crud import TransactionCRUD
from finance_ai.database.models.transaction import Transaction
from finance_ai.database.models.user import User
from finance_ai.tools.expense_service import (
    convert_transaction_to_expense_record,
    create_expense_transaction,
    get_expenses_by_category_and_date_range,
    get_expenses_for_date_range,
    summarize_expenses_for_user,
)


class TestConvertTransactionToExpenseRecord:
    """Tests for converting Transaction to ExpenseRecord."""

    def test_converts_with_all_fields(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Converts transaction with all fields populated."""
        crud = TransactionCRUD()
        txn = crud.create(
            test_session,
            user_id=sample_user.id,
            transaction_type="expense",
            category="food",
            description="กาแฟ",
            amount=Decimal("80.00"),
            transaction_date=date(2026, 2, 1),
        )
        record = convert_transaction_to_expense_record(txn)
        assert record.amount == Decimal("80.00")
        assert record.category == "food"
        assert record.description == "กาแฟ"
        assert record.transaction_date == date(2026, 2, 1)

    def test_converts_with_null_category(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Defaults category to 'other' when null."""
        crud = TransactionCRUD()
        txn = crud.create(
            test_session,
            user_id=sample_user.id,
            transaction_type="expense",
            amount=Decimal("100.00"),
            transaction_date=date(2026, 2, 1),
        )
        record = convert_transaction_to_expense_record(txn)
        assert record.category == "other"

    def test_converts_with_null_description(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Defaults description to empty string when null."""
        crud = TransactionCRUD()
        txn = crud.create(
            test_session,
            user_id=sample_user.id,
            transaction_type="expense",
            category="food",
            amount=Decimal("50.00"),
            transaction_date=date(2026, 2, 1),
        )
        record = convert_transaction_to_expense_record(txn)
        assert record.description == ""


class TestCreateExpenseTransaction:
    """Tests for creating expense transactions."""

    def test_creates_valid_expense(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Creates expense transaction with valid inputs."""
        txn = create_expense_transaction(
            test_session,
            user_id=sample_user.id,
            amount=Decimal("80.00"),
            category="food",
            description="กาแฟ",
            transaction_date=date(2026, 2, 1),
        )
        assert isinstance(txn, Transaction)
        assert txn.transaction_type == "expense"
        assert txn.category == "food"
        assert txn.amount == Decimal("80.00")

    def test_normalizes_category(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Normalizes category to lowercase."""
        txn = create_expense_transaction(
            test_session,
            user_id=sample_user.id,
            amount=Decimal("50.00"),
            category="Food",
            description="ข้าว",
            transaction_date=date(2026, 2, 1),
        )
        assert txn.category == "food"

    def test_invalid_amount_raises(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Rejects negative expense amount."""
        with pytest.raises(ValueError, match="must be at least"):
            create_expense_transaction(
                test_session,
                user_id=sample_user.id,
                amount=Decimal("-10.00"),
                category="food",
                description="",
                transaction_date=date(2026, 2, 1),
            )

    def test_invalid_category_raises(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Rejects invalid category."""
        with pytest.raises(ValueError, match="Unknown expense category"):
            create_expense_transaction(
                test_session,
                user_id=sample_user.id,
                amount=Decimal("100.00"),
                category="gambling",
                description="",
                transaction_date=date(2026, 2, 1),
            )


class TestGetExpensesForDateRange:
    """Tests for querying expenses by date range."""

    def test_returns_expenses_in_range(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Returns only expense transactions within the date range."""
        create_expense_transaction(
            test_session,
            sample_user.id,
            Decimal("80.00"),
            "food",
            "กาแฟ",
            date(2026, 2, 5),
        )
        create_expense_transaction(
            test_session,
            sample_user.id,
            Decimal("120.00"),
            "transport",
            "BTS",
            date(2026, 2, 10),
        )
        # Outside range
        create_expense_transaction(
            test_session,
            sample_user.id,
            Decimal("200.00"),
            "food",
            "ข้าว",
            date(2026, 3, 1),
        )
        expenses = get_expenses_for_date_range(
            test_session,
            sample_user.id,
            date(2026, 2, 1),
            date(2026, 2, 28),
        )
        assert len(expenses) == 2

    def test_excludes_non_expense_transactions(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Excludes non-expense transaction types."""
        crud = TransactionCRUD()
        crud.create(
            test_session,
            user_id=sample_user.id,
            transaction_type="income",
            amount=Decimal("50000.00"),
            transaction_date=date(2026, 2, 15),
        )
        create_expense_transaction(
            test_session,
            sample_user.id,
            Decimal("100.00"),
            "food",
            "ข้าว",
            date(2026, 2, 15),
        )
        expenses = get_expenses_for_date_range(
            test_session,
            sample_user.id,
            date(2026, 2, 1),
            date(2026, 2, 28),
        )
        assert len(expenses) == 1
        assert expenses[0].category == "food"

    def test_empty_range(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Returns empty list when no expenses in range."""
        expenses = get_expenses_for_date_range(
            test_session,
            sample_user.id,
            date(2026, 2, 1),
            date(2026, 2, 28),
        )
        assert expenses == []


class TestGetExpensesByCategoryAndDateRange:
    """Tests for querying expenses by category and date range."""

    def test_filters_by_category(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Returns only expenses matching the category."""
        create_expense_transaction(
            test_session,
            sample_user.id,
            Decimal("80.00"),
            "food",
            "กาแฟ",
            date(2026, 2, 5),
        )
        create_expense_transaction(
            test_session,
            sample_user.id,
            Decimal("120.00"),
            "transport",
            "BTS",
            date(2026, 2, 10),
        )
        expenses = get_expenses_by_category_and_date_range(
            test_session,
            sample_user.id,
            "food",
            date(2026, 2, 1),
            date(2026, 2, 28),
        )
        assert len(expenses) == 1
        assert expenses[0].amount == Decimal("80.00")

    def test_invalid_category_raises(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Rejects invalid category."""
        with pytest.raises(ValueError, match="Unknown expense category"):
            get_expenses_by_category_and_date_range(
                test_session,
                sample_user.id,
                "gambling",
                date(2026, 2, 1),
                date(2026, 2, 28),
            )


class TestSummarizeExpensesForUser:
    """Integration tests for full expense summary flow."""

    def test_summarizes_multiple_categories(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Summarizes expenses across multiple categories."""
        create_expense_transaction(
            test_session,
            sample_user.id,
            Decimal("80.00"),
            "food",
            "กาแฟ",
            date(2026, 2, 5),
        )
        create_expense_transaction(
            test_session,
            sample_user.id,
            Decimal("200.00"),
            "food",
            "ข้าว",
            date(2026, 2, 10),
        )
        create_expense_transaction(
            test_session,
            sample_user.id,
            Decimal("120.00"),
            "transport",
            "BTS",
            date(2026, 2, 15),
        )
        summary = summarize_expenses_for_user(
            test_session,
            sample_user.id,
            date(2026, 2, 1),
            date(2026, 2, 28),
        )
        assert summary.total_amount == Decimal("400.00")
        assert summary.transaction_count == 3
        assert len(summary.category_breakdown) == 2

    def test_empty_summary(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Returns zero summary when no expenses exist."""
        summary = summarize_expenses_for_user(
            test_session,
            sample_user.id,
            date(2026, 2, 1),
            date(2026, 2, 28),
        )
        assert summary.total_amount == Decimal("0")
        assert summary.transaction_count == 0
        assert summary.category_breakdown == []
