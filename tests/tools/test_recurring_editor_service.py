"""Tests for adding, editing and stopping monthly entries from the dashboard."""

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from finance_ai.database.crud.transaction_crud import TransactionCRUD
from finance_ai.database.models.user import User
from finance_ai.tools.debt_service import record_debt
from finance_ai.tools.recurring_editor_service import (
    RecurringInput,
    RecurringNotFoundError,
    create_recurring_item,
    delete_recurring_item,
    list_recurring_items,
    update_recurring_item,
)

TODAY = date(2026, 11, 20)


def _rent(**changes: object) -> RecurringInput:
    values: dict[str, object] = {
        "transaction_type": "income", "category": "rental", "amount": "8000",
        "day_of_month": 3, "description": "ค่าเช่าบ้าน",
    }  # fmt: skip
    values.update(changes)
    return RecurringInput.model_validate(values)


class TestRecurringInput:
    """Form validation before any database work."""

    def test_day_out_of_range(self) -> None:
        """Day 32 is refused."""
        with pytest.raises(ValidationError):
            _rent(day_of_month=32)

    def test_category_must_match_type(self) -> None:
        """An income cannot use an expense category."""
        with pytest.raises(ValidationError, match="Unknown income category"):
            _rent(category="food")

    def test_amount_positive(self) -> None:
        """Zero is refused."""
        with pytest.raises(ValidationError):
            _rent(amount="0")


class TestRecurringEditor:
    """Create, list, edit and stop through the dashboard."""

    def test_create_posts_this_month_when_due(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Day 3, created on the 20th: this month's rent is recorded at once."""
        item = create_recurring_item(test_session, sample_user.id, _rent(), TODAY)
        assert item.posted_now == 1
        assert len(TransactionCRUD().get_all(test_session)) == 1
        [listed] = list_recurring_items(test_session, sample_user.id)
        assert (listed.category_label, listed.amount) == ("ค่าเช่า", Decimal("8000.00"))

    def test_update_changes_future_months(self, test_session: Session, sample_user: User) -> None:
        """New amount and day apply from now on; past postings stay."""
        item = create_recurring_item(test_session, sample_user.id, _rent(), TODAY)
        updated = update_recurring_item(
            test_session, sample_user.id, item.entry_id, _rent(amount="9000", day_of_month=5)
        )
        assert (updated.amount, updated.day_of_month) == (Decimal("9000.00"), 5)
        assert len(TransactionCRUD().get_all(test_session)) == 1

    def test_delete_stops_it(self, test_session: Session, sample_user: User) -> None:
        """A stopped entry is no longer listed; recorded months stay."""
        item = create_recurring_item(test_session, sample_user.id, _rent(), TODAY)
        delete_recurring_item(test_session, sample_user.id, item.entry_id)
        assert list_recurring_items(test_session, sample_user.id) == []
        assert len(TransactionCRUD().get_all(test_session)) == 1

    def test_installment_linked_to_debt(self, test_session: Session, sample_user: User) -> None:
        """Choosing a debt pays it down each month and shows its creditor."""
        debt = record_debt(test_session, sample_user.id, "ธนาคาร", Decimal("100000")).debt
        form = _rent(transaction_type="expense", category="debt", amount="5000", debt_id=debt.id)
        item = create_recurring_item(test_session, sample_user.id, form, TODAY)
        assert item.debt_creditor == "ธนาคาร"
        assert debt.remaining_balance == Decimal("95000")

    def test_someone_elses_debt_refused(self, test_session: Session, sample_user: User) -> None:
        """A debt id that is not the user's cannot be linked."""
        form = _rent(transaction_type="expense", category="debt", debt_id="not-mine")
        with pytest.raises(RecurringNotFoundError):
            create_recurring_item(test_session, sample_user.id, form, TODAY)

    def test_other_users_entry_not_found(self, test_session: Session, sample_user: User) -> None:
        """Someone else's id is treated as missing."""
        item = create_recurring_item(test_session, sample_user.id, _rent(), TODAY)
        with pytest.raises(RecurringNotFoundError):
            delete_recurring_item(test_session, "someone-else", item.entry_id)
        with pytest.raises(RecurringNotFoundError):
            update_recurring_item(test_session, "someone-else", item.entry_id, _rent())
