"""Tests for monthly recurring income and expenses."""

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session

from finance_ai.database.crud.income_crud import IncomeCRUD
from finance_ai.database.crud.transaction_crud import TransactionCRUD
from finance_ai.database.models.user import User
from finance_ai.tools.recurring_service import (
    cancel_recurring_entry,
    create_recurring_entry,
    due_dates,
    list_recurring_entries,
    post_due_entries,
)


def _rent(session: Session, user: User, start: date, day: int = 3) -> str:
    entry = create_recurring_entry(
        session, user.id, "income", "rental", Decimal("8000"), day, "ค่าเช่าบ้าน", start
    )
    return entry.id


class TestDueDates:
    """Which monthly dates fall due between the last posting and today."""

    def test_same_day_start_is_due_today(self) -> None:
        """Created on the 3rd for the 3rd: today counts."""
        assert due_dates(3, date(2026, 11, 3), None, date(2026, 11, 3)) == [date(2026, 11, 3)]

    def test_day_already_passed_waits_for_next_month(self) -> None:
        """Created on the 5th for the 3rd: first one is next month."""
        assert due_dates(3, date(2026, 11, 5), None, date(2026, 11, 30)) == []
        assert due_dates(3, date(2026, 11, 5), None, date(2026, 12, 3)) == [date(2026, 12, 3)]

    def test_catches_up_missed_months(self) -> None:
        """Months missed while the app slept are all posted."""
        result = due_dates(3, date(2026, 9, 1), date(2026, 9, 3), date(2026, 12, 10))
        assert result == [date(2026, 10, 3), date(2026, 11, 3), date(2026, 12, 3)]

    def test_day_31_uses_month_end(self) -> None:
        """Day 31 falls on the last day of shorter months."""
        result = due_dates(31, date(2027, 1, 31), date(2027, 1, 31), date(2027, 3, 31))
        assert result == [date(2027, 2, 28), date(2027, 3, 31)]


class TestCreateRecurringEntry:
    """Validation when creating an entry."""

    def test_rejects_bad_day(self, test_session: Session, sample_user: User) -> None:
        """Day must be 1-31."""
        with pytest.raises(ValueError, match="32"):
            create_recurring_entry(
                test_session, sample_user.id, "income", "rental",
                Decimal("8000"), 32, "", date(2026, 11, 3),
            )  # fmt: skip

    def test_rejects_unknown_type(self, test_session: Session, sample_user: User) -> None:
        """Type must be income or expense."""
        with pytest.raises(ValueError, match="transfer"):
            create_recurring_entry(
                test_session, sample_user.id, "transfer", "other",
                Decimal("1"), 1, "", date(2026, 11, 3),
            )  # fmt: skip

    def test_rejects_non_positive_amount(self, test_session: Session, sample_user: User) -> None:
        """Amount must be positive."""
        with pytest.raises(ValueError, match="0"):
            create_recurring_entry(
                test_session, sample_user.id, "expense", "debt",
                Decimal("0"), 1, "", date(2026, 11, 3),
            )  # fmt: skip

    def test_rejects_unknown_expense_category(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Expense categories are validated."""
        with pytest.raises(ValueError):
            create_recurring_entry(
                test_session, sample_user.id, "expense", "yacht",
                Decimal("10"), 1, "", date(2026, 11, 3),
            )  # fmt: skip

    def test_unknown_income_type_becomes_other(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Unknown income types are stored as other."""
        entry = create_recurring_entry(
            test_session, sample_user.id, "income", "lottery",
            Decimal("10"), 1, "", date(2026, 11, 3),
        )  # fmt: skip
        assert entry.category == "other"


class TestPostDueEntries:
    """Posting turns due months into transactions, once."""

    def test_income_creates_transaction_and_income(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Rental income shows on the dashboard and counts for tax."""
        _rent(test_session, sample_user, date(2026, 11, 3))
        posted = post_due_entries(test_session, date(2026, 11, 3), sample_user.id)
        assert posted == 1
        transactions = TransactionCRUD().get_all(test_session)
        assert [(t.transaction_type, t.category, t.amount) for t in transactions] == [
            ("income", "rental", Decimal("8000.00"))
        ]
        incomes = IncomeCRUD().get_by_user_and_year(test_session, sample_user.id, 2026)
        assert incomes[0].transaction_id == transactions[0].id

    def test_posting_twice_does_not_duplicate(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Running again the same day adds nothing."""
        _rent(test_session, sample_user, date(2026, 11, 3))
        post_due_entries(test_session, date(2026, 11, 3), sample_user.id)
        assert post_due_entries(test_session, date(2026, 11, 3), sample_user.id) == 0

    def test_expense_entry_posts_expense(self, test_session: Session, sample_user: User) -> None:
        """A monthly car loan is posted as a debt expense on its date."""
        create_recurring_entry(
            test_session, sample_user.id, "expense", "debt",
            Decimal("6500"), 10, "ผ่อนรถ", date(2026, 10, 1),
        )  # fmt: skip
        post_due_entries(test_session, date(2026, 11, 12))
        dates = sorted(t.transaction_date for t in TransactionCRUD().get_all(test_session))
        assert dates == [date(2026, 10, 10), date(2026, 11, 10)]

    def test_cancelled_entry_stops(self, test_session: Session, sample_user: User) -> None:
        """Cancelled entries post nothing more."""
        _rent(test_session, sample_user, date(2026, 11, 3))
        assert cancel_recurring_entry(test_session, sample_user.id, "เช่า") is not None
        assert post_due_entries(test_session, date(2026, 12, 3), sample_user.id) == 0


class TestListAndCancel:
    """Listing and cancelling by keyword."""

    def test_list_active_only(self, test_session: Session, sample_user: User) -> None:
        """Cancelled entries are hidden."""
        _rent(test_session, sample_user, date(2026, 11, 3))
        assert len(list_recurring_entries(test_session, sample_user.id)) == 1
        cancel_recurring_entry(test_session, sample_user.id, "")
        assert list_recurring_entries(test_session, sample_user.id) == []

    def test_cancel_without_entries_returns_none(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Nothing to cancel returns None."""
        assert cancel_recurring_entry(test_session, sample_user.id, "เช่า") is None


class TestInstallmentLinkedToDebt:
    """A monthly installment linked to a debt lowers it and stops once paid off."""

    def test_each_month_lowers_debt_then_stops(
        self, test_session: Session, sample_user: User
    ) -> None:
        """3,000 owed, 1,000 a month: three postings, then the entry ends."""
        from finance_ai.tools.debt_service import list_active_debts, record_debt

        debt = record_debt(test_session, sample_user.id, "ธนาคาร", Decimal("3000")).debt
        create_recurring_entry(
            test_session, sample_user.id, "expense", "debt", Decimal("1000"), 5, "ผ่อนธนาคาร",
            date(2026, 8, 1), debt_id=debt.id,
        )  # fmt: skip

        assert post_due_entries(test_session, date(2026, 9, 5), sample_user.id) == 2
        assert debt.remaining_balance == Decimal("1000")
        assert post_due_entries(test_session, date(2027, 1, 5), sample_user.id) == 1
        assert list_active_debts(test_session, sample_user.id) == []
        assert list_recurring_entries(test_session, sample_user.id) == []

    def test_debt_already_settled_posts_nothing(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Paid off by hand before the due date: no extra installment is recorded."""
        from finance_ai.tools.debt_service import record_debt, set_debt_balance

        debt = record_debt(test_session, sample_user.id, "ธนาคาร", Decimal("3000")).debt
        create_recurring_entry(
            test_session, sample_user.id, "expense", "debt", Decimal("1000"), 5, "ผ่อนธนาคาร",
            date(2026, 8, 1), debt_id=debt.id,
        )  # fmt: skip
        set_debt_balance(test_session, sample_user.id, "ธนาคาร", Decimal("0"))

        assert post_due_entries(test_session, date(2026, 9, 5), sample_user.id) == 0
        assert TransactionCRUD().get_all(test_session) == []
