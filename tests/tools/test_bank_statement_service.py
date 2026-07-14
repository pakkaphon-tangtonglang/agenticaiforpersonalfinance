"""Tests for bank statement import service."""

from collections.abc import Callable
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session

from finance_ai.database.models.user import User
from finance_ai.tools.bank_statement_parser import ParsedTransaction
from finance_ai.tools.bank_statement_service import bulk_insert_transactions


class TestBankStatementService:
    """Tests for bank statement service functions."""

    def test_bulk_insert_transactions_creates_records(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Verify transactions are created from parsed data."""
        parsed = [
            ParsedTransaction(
                transaction_date=date(2026, 1, 15),
                description="Food",
                amount=Decimal("150.00"),
                category="food",
                transaction_type="expense",
            ),
            ParsedTransaction(
                transaction_date=date(2026, 1, 16),
                description="Salary",
                amount=Decimal("30000.00"),
                category="salary",
                transaction_type="income",
            ),
        ]
        result = bulk_insert_transactions(parsed, sample_user.id, db_session_factory)
        assert result == 2

    def test_bulk_insert_empty_list(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Empty list returns zero counts."""
        result = bulk_insert_transactions([], sample_user.id, db_session_factory)
        assert result == 0

    def test_bulk_insert_skips_duplicates(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Duplicate transactions are skipped on second import."""
        parsed = [
            ParsedTransaction(
                transaction_date=date(2026, 1, 15),
                description="Food",
                amount=Decimal("150.00"),
                category="food",
                transaction_type="expense",
            ),
        ]
        first = bulk_insert_transactions(parsed, sample_user.id, db_session_factory)
        second = bulk_insert_transactions(parsed, sample_user.id, db_session_factory)
        assert first == 1
        assert second == 0
