"""Dashboard add / edit / delete for debts (the chat uses debt_service directly)."""

from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.database.models.debt import Debt
from finance_ai.database.models.recurring_entry import RecurringEntry
from finance_ai.tools.debt_constants import DEBT_TYPES
from finance_ai.tools.debt_service import list_active_debts

CENT = Decimal("0.01")
MAX_DEBT_AMOUNT = Decimal("999999999999.99")


class DebtNotFoundError(LookupError):
    """The debt does not exist or belongs to another user."""


class DebtInput(BaseModel):
    """A new or edited debt from the dashboard form."""

    creditor: str = Field(min_length=1, max_length=255)
    debt_type: str
    original_amount: Decimal = Field(gt=0, le=MAX_DEBT_AMOUNT)
    remaining_balance: Decimal = Field(ge=0, le=MAX_DEBT_AMOUNT)
    interest_rate_percent: Optional[Decimal] = Field(default=None, ge=0, le=100)
    monthly_payment: Optional[Decimal] = Field(default=None, ge=0, le=MAX_DEBT_AMOUNT)

    @field_validator("creditor")
    @classmethod
    def _trimmed(cls, value: str) -> str:
        """No blank creditor."""
        if not value.strip():
            raise ValueError("Creditor is required (who the money is owed to).")
        return value.strip()

    @field_validator("original_amount", "remaining_balance", "monthly_payment")
    @classmethod
    def _to_satang(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        """Store money with 2 decimal places."""
        return None if value is None else value.quantize(CENT, rounding=ROUND_HALF_UP)

    @field_validator("debt_type")
    @classmethod
    def _known_type(cls, value: str) -> str:
        """Only the types the form offers."""
        if value not in DEBT_TYPES:
            raise ValueError(f"Unknown debt type '{value}'. Allowed: {', '.join(DEBT_TYPES)}")
        return value

    @model_validator(mode="after")
    def _balance_within_original(self) -> "DebtInput":
        """What is left cannot exceed what was borrowed."""
        if self.remaining_balance > self.original_amount:
            raise ValueError(
                f"Remaining balance cannot be more than the original amount. "
                f"Original: {self.original_amount} THB, Remaining: {self.remaining_balance} THB"
            )
        return self


class DebtEntry(DebtInput):
    """A saved debt as the dashboard shows it."""

    debt_id: str
    debt_type_label: str
    paid_percentage: Decimal
    is_paid_off: bool


class DebtListing(BaseModel):
    """Open debts with totals."""

    debts: list[DebtEntry]
    total_remaining: Decimal
    total_monthly_payment: Decimal


def _to_entry(debt: Debt) -> DebtEntry:
    """ORM row → dashboard entry (paid percentage of the original amount)."""
    paid = debt.original_amount - debt.remaining_balance
    percentage = (paid / debt.original_amount * 100).quantize(CENT) if debt.original_amount else 0
    return DebtEntry(
        debt_id=debt.id, creditor=debt.creditor, debt_type=debt.debt_type,
        debt_type_label=DEBT_TYPES.get(debt.debt_type, debt.debt_type),
        original_amount=debt.original_amount, remaining_balance=debt.remaining_balance,
        interest_rate_percent=debt.interest_rate_percent, monthly_payment=debt.monthly_payment,
        paid_percentage=Decimal(percentage), is_paid_off=debt.is_paid_off,
    )  # fmt: skip


def list_debt_entries(session: Session, user_id: str) -> DebtListing:
    """Open debts for the dashboard card.

    Example:
        >>> list_debt_entries(session, uid).total_remaining
        Decimal('420000.00')
    """
    entries = [_to_entry(debt) for debt in list_active_debts(session, user_id)]
    return DebtListing(
        debts=entries,
        total_remaining=sum((entry.remaining_balance for entry in entries), Decimal("0")),
        total_monthly_payment=sum(
            (entry.monthly_payment or Decimal("0") for entry in entries), Decimal("0")
        ),
    )


def _owned_debt(session: Session, user_id: str, debt_id: str) -> Debt:
    """The user's debt, or DebtNotFoundError."""
    debt = session.get(Debt, debt_id)
    if debt is None or debt.user_id != user_id:
        raise DebtNotFoundError(f"Debt not found: {debt_id}")
    return debt


def _apply(debt: Debt, data: DebtInput) -> None:
    """Copy the form onto the row; a zero balance closes the debt."""
    for field, value in data.model_dump().items():
        setattr(debt, field, value)
    debt.is_paid_off = data.remaining_balance == 0


def create_debt_entry(session: Session, user_id: str, data: DebtInput) -> DebtEntry:
    """Save a debt typed into the dashboard (committed).

    Example:
        >>> create_debt_entry(session, uid, form).creditor
        'ธนาคารกสิกร'
    """
    debt = Debt(user_id=user_id)
    _apply(debt, data)
    session.add(debt)
    session.commit()
    return _to_entry(debt)


def update_debt_entry(session: Session, user_id: str, debt_id: str, data: DebtInput) -> DebtEntry:
    """Replace a debt's values (committed).

    Raises:
        DebtNotFoundError: If the debt is missing or not the user's.
    """
    debt = _owned_debt(session, user_id, debt_id)
    _apply(debt, data)
    session.commit()
    return _to_entry(debt)


def delete_debt_entry(session: Session, user_id: str, debt_id: str) -> None:
    """Delete a debt; linked installments stay but stop paying it down (committed).

    Raises:
        DebtNotFoundError: If the debt is missing or not the user's.
    """
    debt = _owned_debt(session, user_id, debt_id)
    linked = select(RecurringEntry).where(RecurringEntry.debt_id == debt.id)
    for installment in session.execute(linked).scalars():
        installment.debt_id = None
    session.flush()
    session.delete(debt)
    session.commit()
