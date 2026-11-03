"""Debts the user still owes and the payments that lower them.

Owing money is not spending it, so record_debt never creates a transaction.
Paying back is spending: the caller records a "debt" expense and then calls
apply_debt_payment, which lowers the balance (never below zero).
"""

from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.database.models.debt import Debt
from finance_ai.tools.debt_constants import DEBT_TYPES, DEFAULT_DEBT_TYPE, FALLBACK_DEBT_TYPE

_ZERO = Decimal("0")


class DebtRecordResult(BaseModel):
    """The saved debt and whether it was added to one already on file."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    debt: Debt
    added_to_existing: bool


class DebtPaymentResult(BaseModel):
    """What one payment did to a debt.

    Example:
        >>> DebtPaymentResult(applied=Decimal("500"), remaining_balance=Decimal("1500"),
        ...     overpaid=Decimal("0"), paid_off=False).paid_off
        False
    """

    applied: Decimal
    remaining_balance: Decimal
    overpaid: Decimal
    paid_off: bool


class DebtSummary(BaseModel):
    """Open debts with their totals."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    debts: list[Debt]
    debt_count: int
    total_remaining: Decimal
    total_monthly_payment: Decimal


def normalize_debt_type(debt_type: str) -> str:
    """Known debt type key, or "other".

    Example:
        >>> normalize_debt_type("Credit_Card"), normalize_debt_type("x")
        ('credit_card', 'other')
    """
    key = debt_type.strip().lower() or DEFAULT_DEBT_TYPE
    return key if key in DEBT_TYPES else FALLBACK_DEBT_TYPE


def list_active_debts(session: Session, user_id: str) -> list[Debt]:
    """The user's debts that are not paid off, oldest first.

    Example:
        >>> [debt.creditor for debt in list_active_debts(session, uid)]
        ['เพื่อน']
    """
    statement = (
        select(Debt)
        .where(Debt.user_id == user_id, Debt.is_paid_off.is_(False))
        .order_by(Debt.created_at)
    )
    return list(session.execute(statement).scalars().all())


def _same_creditor(session: Session, user_id: str, creditor: str) -> Optional[Debt]:
    """An open debt to exactly this creditor (case and spaces ignored)."""
    wanted = creditor.strip().lower()
    for debt in list_active_debts(session, user_id):
        if debt.creditor.strip().lower() == wanted:
            return debt
    return None


def _validate_new_debt(creditor: str, amount: Decimal) -> None:
    """Reject a blank creditor or a non-positive amount."""
    if not creditor.strip():
        raise ValueError("Creditor is required (who the money is owed to, e.g. 'เพื่อน').")
    if amount <= _ZERO:
        raise ValueError(f"Debt amount must be greater than 0. Received: {amount} THB")


def record_debt(  # pylint: disable=too-many-arguments
    session: Session,
    user_id: str,
    creditor: str,
    amount: Decimal,
    debt_type: str = DEFAULT_DEBT_TYPE,
    *,
    interest_rate_percent: Optional[Decimal] = None,
    monthly_payment: Optional[Decimal] = None,
) -> DebtRecordResult:
    """Save money owed (committed); the same creditor's open debt grows instead.

    Args:
        session: Database session.
        user_id: UUID of the user.
        creditor: Who is owed (e.g. "เพื่อน", "ธนาคารกสิกร").
        amount: Amount owed in THB.
        debt_type: Key from DEBT_TYPES (unknown → "other").
        interest_rate_percent: Yearly interest, if known.
        monthly_payment: Installment per month, if known.

    Returns:
        The debt and whether an existing one was increased.

    Raises:
        ValueError: If the creditor is blank or the amount is not positive.

    Example:
        >>> record_debt(session, uid, "เพื่อน", Decimal("2000")).debt.remaining_balance
        Decimal('2000')
    """
    _validate_new_debt(creditor, amount)
    existing = _same_creditor(session, user_id, creditor)
    if existing is not None:
        existing.original_amount += amount
        existing.remaining_balance += amount
        session.commit()
        return DebtRecordResult(debt=existing, added_to_existing=True)
    debt = Debt(
        user_id=user_id, creditor=creditor.strip(), debt_type=normalize_debt_type(debt_type),
        original_amount=amount, remaining_balance=amount, is_paid_off=False,
        interest_rate_percent=interest_rate_percent, monthly_payment=monthly_payment,
    )  # fmt: skip
    session.add(debt)
    session.commit()
    return DebtRecordResult(debt=debt, added_to_existing=False)


def _matches(debt: Debt, term: str) -> bool:
    """Whether the term names this debt's creditor or its type."""
    creditor = debt.creditor.lower()
    label = DEBT_TYPES.get(debt.debt_type, "").lower()
    return term in creditor or creditor in term or term == debt.debt_type or term in label


def find_active_debt(session: Session, user_id: str, search_term: str) -> Optional[Debt]:
    """The newest open debt whose creditor or type matches; None for a blank term.

    Example:
        >>> find_active_debt(session, uid, "กสิกร").creditor
        'ธนาคารกสิกร'
    """
    term = search_term.strip().lower()
    if not term:
        return None
    matches = [debt for debt in list_active_debts(session, user_id) if _matches(debt, term)]
    return matches[-1] if matches else None


def apply_debt_payment(session: Session, debt: Debt, amount: Decimal) -> DebtPaymentResult:
    """Lower the balance by a payment (committed); paying it all closes the debt.

    Args:
        session: Database session.
        debt: The debt being paid.
        amount: Payment in THB.

    Returns:
        How much went to the debt, what is left and any overpayment.

    Example:
        >>> apply_debt_payment(session, debt, Decimal("500")).remaining_balance
        Decimal('1500')
    """
    applied = min(amount, debt.remaining_balance)
    debt.remaining_balance -= applied
    debt.is_paid_off = debt.remaining_balance <= _ZERO
    session.commit()
    return DebtPaymentResult(
        applied=applied,
        remaining_balance=debt.remaining_balance,
        overpaid=amount - applied,
        paid_off=debt.is_paid_off,
    )


def set_debt_balance(
    session: Session, user_id: str, search_term: str, remaining_balance: Decimal
) -> Optional[Debt]:
    """Correct what is still owed (committed); zero marks the debt paid off.

    Raises:
        ValueError: If the balance is negative.

    Example:
        >>> set_debt_balance(session, uid, "เพื่อน", Decimal("1200")).remaining_balance
        Decimal('1200')
    """
    if remaining_balance < _ZERO:
        raise ValueError(f"Remaining balance cannot be negative. Received: {remaining_balance} THB")
    debt = find_active_debt(session, user_id, search_term)
    if debt is None:
        return None
    debt.remaining_balance = remaining_balance
    debt.is_paid_off = remaining_balance == _ZERO
    session.commit()
    return debt


def summarize_debts(session: Session, user_id: str) -> DebtSummary:
    """Open debts with the total owed and total monthly installments.

    Example:
        >>> summarize_debts(session, uid).total_remaining
        Decimal('502000.00')
    """
    debts = list_active_debts(session, user_id)
    return DebtSummary(
        debts=debts,
        debt_count=len(debts),
        total_remaining=sum((debt.remaining_balance for debt in debts), _ZERO),
        total_monthly_payment=sum((debt.monthly_payment or _ZERO for debt in debts), _ZERO),
    )
