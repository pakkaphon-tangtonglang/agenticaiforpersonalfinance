"""Tests for debts the user still owes (friend, bank, credit card...)."""

from decimal import Decimal

import pytest
from sqlalchemy.orm import Session

from finance_ai.database.models.user import User
from finance_ai.tools.debt_service import (
    apply_debt_payment,
    find_active_debt,
    list_active_debts,
    record_debt,
    set_debt_balance,
    summarize_debts,
)


def _friend(session: Session, user: User, amount: str = "2000") -> str:
    return record_debt(session, user.id, "เพื่อน", Decimal(amount)).debt.id


class TestRecordDebt:
    """record_debt saves a new balance or adds to the same creditor's."""

    def test_new_debt(self, test_session: Session, sample_user: User) -> None:
        """A friend loan is saved with its full balance and the personal type."""
        result = record_debt(test_session, sample_user.id, "เพื่อน", Decimal("2000"))
        assert result.added_to_existing is False
        assert result.debt.remaining_balance == Decimal("2000")
        assert result.debt.original_amount == Decimal("2000")
        assert result.debt.debt_type == "personal"

    def test_same_creditor_adds_up(self, test_session: Session, sample_user: User) -> None:
        """Borrowing again from the same friend raises the one balance."""
        _friend(test_session, sample_user)
        result = record_debt(test_session, sample_user.id, " เพื่อน ", Decimal("500"))
        assert result.added_to_existing is True
        assert result.debt.remaining_balance == Decimal("2500")
        assert len(list_active_debts(test_session, sample_user.id)) == 1

    def test_bank_loan_with_terms(self, test_session: Session, sample_user: User) -> None:
        """Interest and the monthly installment are kept when given."""
        debt = record_debt(
            test_session, sample_user.id, "ธนาคารกสิกร", Decimal("500000"), "personal_loan",
            interest_rate_percent=Decimal("18"), monthly_payment=Decimal("12000"),
        ).debt  # fmt: skip
        assert (debt.debt_type, debt.interest_rate_percent) == ("personal_loan", Decimal("18"))
        assert debt.monthly_payment == Decimal("12000")

    def test_unknown_type_becomes_other(self, test_session: Session, sample_user: User) -> None:
        """A type the system does not know is filed under other."""
        debt = record_debt(test_session, sample_user.id, "แม่", Decimal("100"), "mystery").debt
        assert debt.debt_type == "other"

    @pytest.mark.parametrize("amount", ["0", "-5"])
    def test_amount_must_be_positive(
        self, test_session: Session, sample_user: User, amount: str
    ) -> None:
        """Zero or negative debts are rejected with the amount in the message."""
        with pytest.raises(ValueError, match="greater than 0"):
            record_debt(test_session, sample_user.id, "เพื่อน", Decimal(amount))

    def test_creditor_required(self, test_session: Session, sample_user: User) -> None:
        """A debt needs someone it is owed to."""
        with pytest.raises(ValueError, match="Creditor"):
            record_debt(test_session, sample_user.id, "  ", Decimal("100"))


class TestFindActiveDebt:
    """find_active_debt matches by creditor or type."""

    def test_matches_part_of_creditor(self, test_session: Session, sample_user: User) -> None:
        """'กสิกร' finds 'ธนาคารกสิกร'."""
        record_debt(test_session, sample_user.id, "ธนาคารกสิกร", Decimal("500000"))
        debt = find_active_debt(test_session, sample_user.id, "กสิกร")
        assert debt is not None and debt.creditor == "ธนาคารกสิกร"

    def test_matches_type_label(self, test_session: Session, sample_user: User) -> None:
        """'บัตรเครดิต' finds a credit card debt named KTC."""
        record_debt(test_session, sample_user.id, "KTC", Decimal("30000"), "credit_card")
        debt = find_active_debt(test_session, sample_user.id, "บัตรเครดิต")
        assert debt is not None and debt.creditor == "KTC"

    def test_no_match(self, test_session: Session, sample_user: User) -> None:
        """Nothing recorded for this creditor."""
        _friend(test_session, sample_user)
        assert find_active_debt(test_session, sample_user.id, "ธนาคาร") is None

    def test_blank_term_finds_nothing(self, test_session: Session, sample_user: User) -> None:
        """A blank search never guesses a debt."""
        _friend(test_session, sample_user)
        assert find_active_debt(test_session, sample_user.id, " ") is None


class TestApplyDebtPayment:
    """apply_debt_payment lowers the balance and never goes below zero."""

    def test_partial_payment(self, test_session: Session, sample_user: User) -> None:
        """2,000 owed, 500 paid: 1,500 left."""
        _friend(test_session, sample_user)
        debt = find_active_debt(test_session, sample_user.id, "เพื่อน")
        assert debt is not None
        result = apply_debt_payment(test_session, debt, Decimal("500"))
        assert (result.applied, result.remaining_balance) == (Decimal("500"), Decimal("1500"))
        assert result.paid_off is False

    def test_overpayment_closes_debt(self, test_session: Session, sample_user: User) -> None:
        """Paying more than owed closes the debt and reports the extra."""
        _friend(test_session, sample_user)
        debt = find_active_debt(test_session, sample_user.id, "เพื่อน")
        assert debt is not None
        result = apply_debt_payment(test_session, debt, Decimal("2500"))
        assert (result.applied, result.overpaid) == (Decimal("2000"), Decimal("500"))
        assert result.paid_off is True
        assert list_active_debts(test_session, sample_user.id) == []


class TestSetDebtBalance:
    """set_debt_balance corrects a balance; zero closes the debt."""

    def test_correct_balance(self, test_session: Session, sample_user: User) -> None:
        """The user says only 1,200 is left."""
        _friend(test_session, sample_user)
        debt = set_debt_balance(test_session, sample_user.id, "เพื่อน", Decimal("1200"))
        assert debt is not None and debt.remaining_balance == Decimal("1200")

    def test_zero_closes(self, test_session: Session, sample_user: User) -> None:
        """Balance 0 means the debt is settled."""
        _friend(test_session, sample_user)
        set_debt_balance(test_session, sample_user.id, "เพื่อน", Decimal("0"))
        assert list_active_debts(test_session, sample_user.id) == []

    def test_negative_rejected(self, test_session: Session, sample_user: User) -> None:
        """A balance cannot be negative."""
        _friend(test_session, sample_user)
        with pytest.raises(ValueError, match="negative"):
            set_debt_balance(test_session, sample_user.id, "เพื่อน", Decimal("-1"))

    def test_not_found(self, test_session: Session, sample_user: User) -> None:
        """No matching debt returns None."""
        assert set_debt_balance(test_session, sample_user.id, "เพื่อน", Decimal("1")) is None


def test_summarize_debts(test_session: Session, sample_user: User) -> None:
    """Total and monthly installments across open debts."""
    _friend(test_session, sample_user)
    record_debt(
        test_session, sample_user.id, "ธนาคาร", Decimal("500000"), "personal_loan",
        monthly_payment=Decimal("12000"),
    )  # fmt: skip
    summary = summarize_debts(test_session, sample_user.id)
    assert summary.total_remaining == Decimal("502000")
    assert summary.total_monthly_payment == Decimal("12000")
    assert summary.debt_count == 2
