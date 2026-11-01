"""Tests for the dashboard income/expense history (list, add, edit, delete)."""

from datetime import date, timedelta
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from finance_ai.database.models.income import Income
from finance_ai.database.models.transaction import Transaction
from finance_ai.tools.transaction_history_service import (
    EntryInput,
    EntryNotFoundError,
    create_entry,
    delete_entry,
    list_entries,
    update_entry,
)

OCT = date(2026, 10, 15)


def _entry(**overrides: Any) -> EntryInput:
    """A valid expense input with optional overrides."""
    values: dict[str, Any] = {
        "transaction_type": "expense",
        "transaction_date": OCT,
        "amount": Decimal("120.00"),
        "category": "food",
        "description": "ข้าวกลางวัน",
    }
    return EntryInput(**{**values, **overrides})


def _income_copies(session: Session, user_id: str) -> list[Income]:
    """Tax copies of income for a user."""
    return list(session.query(Income).filter(Income.user_id == user_id).all())


class TestEntryInputValidation:
    """Inputs are checked before anything touches the database."""

    def test_future_date_rejected(self) -> None:
        """The history holds actual entries only."""
        with pytest.raises(ValidationError, match="future"):
            _entry(transaction_date=date.today() + timedelta(days=1))

    def test_amount_must_be_positive(self) -> None:
        """Zero or negative amounts are rejected."""
        with pytest.raises(ValidationError):
            _entry(amount=Decimal("0"))

    def test_expense_category_must_be_known(self) -> None:
        """Expense categories come from the expense list."""
        with pytest.raises(ValidationError, match="category"):
            _entry(category="spaceship")

    def test_income_uses_income_types(self) -> None:
        """Income entries use income types such as salary."""
        entry = _entry(transaction_type="income", category="salary")
        assert entry.category == "salary"
        with pytest.raises(ValidationError, match="category"):
            _entry(transaction_type="income", category="food")

    def test_amount_rounded_to_satang(self) -> None:
        """Amounts are stored with 2 decimal places."""
        assert _entry(amount=Decimal("99.999")).amount == Decimal("100.00")


class TestListEntries:
    """Listing shows the month's income and expenses, newest first."""

    def test_lists_month_newest_first_with_totals(
        self, test_session: Session, sample_user: Any
    ) -> None:
        """Entries outside the month and investment rows are excluded."""
        create_entry(test_session, sample_user.id, _entry(transaction_date=date(2026, 10, 1)))
        create_entry(
            test_session,
            sample_user.id,
            _entry(
                transaction_type="income",
                category="salary",
                amount=Decimal("30000"),
                transaction_date=date(2026, 10, 25),
            ),
        )
        create_entry(test_session, sample_user.id, _entry(transaction_date=date(2026, 9, 30)))
        test_session.add(
            Transaction(
                user_id=sample_user.id,
                transaction_type="buy",
                amount=Decimal("5000"),
                transaction_date=date(2026, 10, 5),
            )
        )
        test_session.commit()

        page = list_entries(test_session, sample_user.id, 2026, 10)

        assert [e.transaction_date for e in page.entries] == [date(2026, 10, 25), date(2026, 10, 1)]
        assert page.total_income == Decimal("30000.00")
        assert page.total_expense == Decimal("120.00")
        assert page.entries[0].category_label == "เงินเดือน"

    def test_filter_by_kind(self, test_session: Session, sample_user: Any) -> None:
        """kind='income' shows only income."""
        create_entry(test_session, sample_user.id, _entry())
        create_entry(
            test_session, sample_user.id, _entry(transaction_type="income", category="salary")
        )

        page = list_entries(test_session, sample_user.id, 2026, 10, kind="income")

        assert [e.transaction_type for e in page.entries] == ["income"]


class TestIncomeTaxCopyStaysInSync:
    """Income entries keep their tax copy (incomes table) in step."""

    def test_create_income_makes_linked_copy(self, test_session: Session, sample_user: Any) -> None:
        """A new income entry gets a tax copy linked to it."""
        entry = create_entry(
            test_session,
            sample_user.id,
            _entry(transaction_type="income", category="salary", amount=Decimal("30000")),
        )

        copies = _income_copies(test_session, sample_user.id)
        assert len(copies) == 1
        assert copies[0].transaction_id == entry.id
        assert copies[0].tax_year == 2026
        assert copies[0].amount == Decimal("30000.00")

    def test_edit_income_updates_copy(self, test_session: Session, sample_user: Any) -> None:
        """Changing amount or description changes the tax copy too."""
        entry = create_entry(
            test_session, sample_user.id, _entry(transaction_type="income", category="salary")
        )

        update_entry(
            test_session,
            sample_user.id,
            entry.id,
            _entry(
                transaction_type="income",
                category="bonus",
                amount=Decimal("5000"),
                description="โบนัส",
            ),
        )

        copy = _income_copies(test_session, sample_user.id)[0]
        assert (copy.amount, copy.income_type, copy.description) == (
            Decimal("5000.00"),
            "bonus",
            "โบนัส",
        )

    def test_switching_income_to_expense_removes_copy(
        self, test_session: Session, sample_user: Any
    ) -> None:
        """An entry that is no longer income has no tax copy."""
        entry = create_entry(
            test_session, sample_user.id, _entry(transaction_type="income", category="salary")
        )

        update_entry(test_session, sample_user.id, entry.id, _entry())

        assert not _income_copies(test_session, sample_user.id)

    def test_delete_income_removes_copy(self, test_session: Session, sample_user: Any) -> None:
        """Deleting an income entry deletes its tax copy."""
        entry = create_entry(
            test_session, sample_user.id, _entry(transaction_type="income", category="salary")
        )

        delete_entry(test_session, sample_user.id, entry.id)

        assert not _income_copies(test_session, sample_user.id)
        assert list_entries(test_session, sample_user.id, 2026, 10).entries == []

    def test_legacy_unlinked_copy_is_found_and_removed(
        self, test_session: Session, sample_user: Any
    ) -> None:
        """Copies made before the link existed are matched by year, description and amount."""
        txn = Transaction(
            user_id=sample_user.id,
            transaction_type="income",
            category="salary",
            description="เงินเดือน",
            amount=Decimal("30000"),
            transaction_date=OCT,
        )
        test_session.add(txn)
        test_session.add(
            Income(
                user_id=sample_user.id,
                income_type="salary",
                description="เงินเดือน",
                amount=Decimal("30000"),
                tax_year=2026,
                pay_period="monthly",
            )
        )
        test_session.commit()

        delete_entry(test_session, sample_user.id, txn.id)

        assert not _income_copies(test_session, sample_user.id)


class TestOwnership:
    """Users can only change their own entries."""

    def test_other_users_entry_is_not_found(self, test_session: Session, sample_user: Any) -> None:
        """Editing or deleting someone else's entry looks like a missing entry."""
        entry = create_entry(test_session, sample_user.id, _entry())

        with pytest.raises(EntryNotFoundError):
            delete_entry(test_session, "someone-else", entry.id)
        with pytest.raises(EntryNotFoundError):
            update_entry(test_session, "someone-else", entry.id, _entry())

    def test_investment_rows_cannot_be_edited_here(
        self, test_session: Session, sample_user: Any
    ) -> None:
        """Buy/sell rows belong to the assets page."""
        txn = Transaction(
            user_id=sample_user.id,
            transaction_type="buy",
            amount=Decimal("1"),
            transaction_date=OCT,
        )
        test_session.add(txn)
        test_session.commit()

        with pytest.raises(EntryNotFoundError):
            delete_entry(test_session, sample_user.id, txn.id)
