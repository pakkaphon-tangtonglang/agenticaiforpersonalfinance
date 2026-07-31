"""Tests for finance_ai.tools.receipt_ocr_service (confirm + persist)."""

from datetime import date
from decimal import Decimal
from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.database.models.income import Income
from finance_ai.database.models.transaction import Transaction
from finance_ai.tools.receipt_ocr import ReceiptOcrResult
from finance_ai.tools.receipt_ocr_service import (
    ConfirmResult,
    confirm_receipt_transactions,
    to_parsed_transaction,
)


def _expense_draft(
    amount: str = "350.00",
    description: str = "ร้านอาหาร",
    txn_date: date = date(2026, 3, 1),
    category: str = "food",
) -> ReceiptOcrResult:
    """Build an expense draft for tests."""
    return ReceiptOcrResult(
        transaction_type="expense",
        amount=Decimal(amount),
        transaction_date=txn_date,
        description=description,
        category=category,
    )


def _income_draft(
    amount: str = "50000.00",
    description: str = "เงินเดือนมีนาคม",
    withholding_tax: str = "2500.00",
    employer_name: str = "ABC จำกัด",
) -> ReceiptOcrResult:
    """Build an income draft for tests."""
    return ReceiptOcrResult(
        transaction_type="income",
        amount=Decimal(amount),
        transaction_date=date(2026, 3, 31),
        description=description,
        category="other",
        income_type="salary",
        withholding_tax=Decimal(withholding_tax),
        employer_name=employer_name,
    )


class TestToParsedTransaction:
    """Tests for to_parsed_transaction converter."""

    def test_expense_draft_converted(self) -> None:
        """An expense draft maps to an expense ParsedTransaction."""
        parsed = to_parsed_transaction(_expense_draft())
        assert parsed.transaction_type == "expense"
        assert parsed.amount == Decimal("350.00")
        assert parsed.category == "food"
        assert parsed.transaction_date == date(2026, 3, 1)

    def test_income_draft_converted(self) -> None:
        """An income draft maps to an income ParsedTransaction."""
        parsed = to_parsed_transaction(_income_draft())
        assert parsed.transaction_type == "income"
        assert parsed.amount == Decimal("50000.00")


class TestConfirmReceiptTransactions:
    """Tests for confirm_receipt_transactions persistence."""

    def test_expense_persisted_to_transaction_table(
        self,
        sample_user: object,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Confirming an expense writes one row to the transactions table."""
        result = confirm_receipt_transactions(
            [_expense_draft()],
            str(sample_user.id),  # type: ignore[attr-defined]
            db_session_factory,
        )
        assert result.inserted == 1
        assert result.total == 1

        session = db_session_factory()
        rows = list(
            session.execute(
                select(Transaction).where(Transaction.user_id == str(sample_user.id))
            ).scalars()
        )
        assert len(rows) == 1
        assert rows[0].transaction_type == "expense"
        assert rows[0].amount == Decimal("350.00")
        session.close()

    def test_income_dual_written(
        self,
        sample_user: object,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Confirming income writes both Transaction and Income rows."""
        result = confirm_receipt_transactions(
            [_income_draft()],
            str(sample_user.id),  # type: ignore[attr-defined]
            db_session_factory,
        )
        assert result.inserted == 1

        session = db_session_factory()
        txns = list(
            session.execute(
                select(Transaction).where(Transaction.user_id == str(sample_user.id))
            ).scalars()
        )
        incomes = list(
            session.execute(select(Income).where(Income.user_id == str(sample_user.id))).scalars()
        )
        assert len(txns) == 1
        assert len(incomes) == 1
        assert txns[0].transaction_type == "income"
        assert incomes[0].amount == Decimal("50000.00")
        session.close()

    def test_duplicate_skipped(
        self,
        sample_user: object,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Confirming the same draft twice skips the duplicate."""
        uid = str(sample_user.id)  # type: ignore[attr-defined]
        confirm_receipt_transactions([_expense_draft()], uid, db_session_factory)
        result = confirm_receipt_transactions([_expense_draft()], uid, db_session_factory)
        assert result.inserted == 0
        assert result.total == 1
        assert result.skipped == 1

    def test_multiple_drafts_mixed(
        self,
        sample_user: object,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """A mix of expense and income drafts all persist."""
        uid = str(sample_user.id)  # type: ignore[attr-defined]
        result = confirm_receipt_transactions(
            [_expense_draft(), _income_draft()],
            uid,
            db_session_factory,
        )
        assert result.inserted == 2
        assert result.total == 2

    def test_empty_drafts_inserts_nothing(
        self,
        sample_user: object,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """An empty draft list inserts nothing and reports zero."""
        result = confirm_receipt_transactions(
            [], str(sample_user.id), db_session_factory  # type: ignore[attr-defined]
        )
        assert result.inserted == 0
        assert result.total == 0
        assert result.skipped == 0
