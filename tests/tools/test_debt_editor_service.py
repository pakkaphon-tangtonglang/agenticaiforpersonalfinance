"""Tests for adding, editing and deleting debts from the dashboard."""

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from finance_ai.database.models.user import User
from finance_ai.tools.debt_editor_service import (
    DebtInput,
    DebtNotFoundError,
    create_debt_entry,
    delete_debt_entry,
    list_debt_entries,
    update_debt_entry,
)
from finance_ai.tools.recurring_service import create_recurring_entry, list_recurring_entries


def _bank_loan(**changes: object) -> DebtInput:
    values: dict[str, object] = {
        "creditor": "ธนาคารกสิกร", "debt_type": "personal_loan",
        "original_amount": "500000", "remaining_balance": "420000",
        "interest_rate_percent": "18", "monthly_payment": "12000",
    }  # fmt: skip
    values.update(changes)
    return DebtInput.model_validate(values)


class TestDebtInput:
    """Form validation before any database work."""

    def test_rounds_money(self) -> None:
        """Money is kept to the satang."""
        assert _bank_loan(remaining_balance="1.005").remaining_balance == Decimal("1.01")

    def test_balance_above_original_rejected(self) -> None:
        """What is left cannot be more than what was borrowed."""
        with pytest.raises(ValidationError, match="cannot be more"):
            _bank_loan(remaining_balance="600000")

    def test_unknown_type_rejected(self) -> None:
        """The form only offers known types."""
        with pytest.raises(ValidationError, match="Unknown debt type"):
            _bank_loan(debt_type="mystery")

    def test_blank_creditor_rejected(self) -> None:
        """A creditor is required."""
        with pytest.raises(ValidationError):
            _bank_loan(creditor="  ")

    def test_optional_terms_blank(self) -> None:
        """Interest and installment may be left empty."""
        form = _bank_loan(interest_rate_percent=None, monthly_payment=None)
        assert form.interest_rate_percent is None


class TestDebtEditor:
    """Create, list, edit and delete through the dashboard."""

    def test_create_and_list(self, test_session: Session, sample_user: User) -> None:
        """A saved debt is listed with its labels and totals."""
        create_debt_entry(test_session, sample_user.id, _bank_loan())
        listing = list_debt_entries(test_session, sample_user.id)
        assert listing.total_remaining == Decimal("420000")
        assert listing.debts[0].debt_type_label == "สินเชื่อส่วนบุคคล"
        assert listing.debts[0].paid_percentage == Decimal("16.00")

    def test_update(self, test_session: Session, sample_user: User) -> None:
        """Editing replaces the values; balance 0 marks it paid off."""
        entry = create_debt_entry(test_session, sample_user.id, _bank_loan())
        updated = update_debt_entry(
            test_session, sample_user.id, entry.debt_id, _bank_loan(remaining_balance="0")
        )
        assert updated.is_paid_off is True
        assert list_debt_entries(test_session, sample_user.id).debts == []

    def test_delete_unlinks_installment(self, test_session: Session, sample_user: User) -> None:
        """Deleting a debt keeps the monthly installment but no longer pays it down."""
        entry = create_debt_entry(test_session, sample_user.id, _bank_loan())
        create_recurring_entry(
            test_session, sample_user.id, "expense", "debt", Decimal("12000"), 5, "ผ่อน",
            date(2026, 11, 1), debt_id=entry.debt_id,
        )  # fmt: skip
        delete_debt_entry(test_session, sample_user.id, entry.debt_id)
        assert list_debt_entries(test_session, sample_user.id).debts == []
        [installment] = list_recurring_entries(test_session, sample_user.id)
        assert installment.debt_id is None

    def test_other_users_debt_not_found(self, test_session: Session, sample_user: User) -> None:
        """Someone else's id is treated as missing."""
        entry = create_debt_entry(test_session, sample_user.id, _bank_loan())
        with pytest.raises(DebtNotFoundError):
            delete_debt_entry(test_session, "someone-else", entry.debt_id)
        with pytest.raises(DebtNotFoundError):
            update_debt_entry(test_session, "someone-else", entry.debt_id, _bank_loan())
