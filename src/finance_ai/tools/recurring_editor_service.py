"""Dashboard add / edit / stop for monthly entries (the chat uses recurring_service)."""

from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.orm import Session

from finance_ai.database.models.debt import Debt
from finance_ai.database.models.recurring_entry import RecurringEntry
from finance_ai.tools.expense_constants import EXPENSE_CATEGORIES, INCOME_TYPES
from finance_ai.tools.recurring_service import (
    MAX_DAY_OF_MONTH,
    create_recurring_entry,
    list_recurring_entries,
    post_due_entries,
)

CENT = Decimal("0.01")
MAX_MONTHLY_AMOUNT = Decimal("999999999.99")


class RecurringNotFoundError(LookupError):
    """The entry (or the debt it links to) does not exist or is not the user's."""


class RecurringInput(BaseModel):
    """A new or edited monthly entry from the dashboard form."""

    transaction_type: Literal["income", "expense"]
    category: str
    amount: Decimal = Field(gt=0, le=MAX_MONTHLY_AMOUNT)
    day_of_month: int = Field(ge=1, le=MAX_DAY_OF_MONTH)
    description: str = Field(default="", max_length=255)
    debt_id: Optional[str] = None

    @field_validator("amount")
    @classmethod
    def _to_satang(cls, value: Decimal) -> Decimal:
        """Store money with 2 decimal places."""
        return value.quantize(CENT, rounding=ROUND_HALF_UP)

    @model_validator(mode="after")
    def _category_matches_type(self) -> "RecurringInput":
        """Income uses income types; expenses use expense categories."""
        allowed = INCOME_TYPES if self.transaction_type == "income" else EXPENSE_CATEGORIES
        if self.category not in allowed:
            raise ValueError(
                f"Unknown {self.transaction_type} category '{self.category}'. "
                f"Allowed: {', '.join(allowed)}"
            )
        return self


class RecurringItem(RecurringInput):
    """A saved monthly entry as the dashboard shows it."""

    entry_id: str
    category_label: str
    debt_creditor: Optional[str] = None
    posted_now: int = 0


def _label(entry: RecurringEntry) -> str:
    """Thai label of the entry's category."""
    labels = INCOME_TYPES if entry.transaction_type == "income" else EXPENSE_CATEGORIES
    return labels.get(entry.category, entry.category)


def _to_item(session: Session, entry: RecurringEntry, posted_now: int = 0) -> RecurringItem:
    """ORM row → dashboard item, with the creditor of a linked debt."""
    debt = session.get(Debt, entry.debt_id) if entry.debt_id else None
    return RecurringItem(
        entry_id=entry.id, transaction_type=entry.transaction_type,  # type: ignore[arg-type]
        category=entry.category, category_label=_label(entry), amount=entry.amount,
        day_of_month=entry.day_of_month, description=entry.description, debt_id=entry.debt_id,
        debt_creditor=debt.creditor if debt else None, posted_now=posted_now,
    )  # fmt: skip


def list_recurring_items(session: Session, user_id: str) -> list[RecurringItem]:
    """The user's active monthly entries for the dashboard card.

    Example:
        >>> [item.description for item in list_recurring_items(session, uid)]
        ['ค่าเช่าบ้าน']
    """
    return [_to_item(session, entry) for entry in list_recurring_entries(session, user_id)]


def _owned_debt_id(session: Session, user_id: str, debt_id: Optional[str]) -> Optional[str]:
    """The debt id when it is the user's; RecurringNotFoundError otherwise."""
    if not debt_id:
        return None
    debt = session.get(Debt, debt_id)
    if debt is None or debt.user_id != user_id:
        raise RecurringNotFoundError(f"Debt not found: {debt_id}")
    return debt.id


def create_recurring_item(
    session: Session, user_id: str, data: RecurringInput, today: date
) -> RecurringItem:
    """Save a monthly entry and record this month if its day has passed (committed).

    Example:
        >>> create_recurring_item(session, uid, form, date(2026, 11, 20)).posted_now
        1
    """
    debt_id = _owned_debt_id(session, user_id, data.debt_id)
    entry = create_recurring_entry(
        session, user_id, data.transaction_type, data.category, data.amount,
        data.day_of_month, data.description, today.replace(day=1), debt_id=debt_id,
    )  # fmt: skip
    posted_now = post_due_entries(session, today, user_id)
    return _to_item(session, entry, posted_now)


def _owned_entry(session: Session, user_id: str, entry_id: str) -> RecurringEntry:
    """The user's active entry, or RecurringNotFoundError."""
    entry = session.get(RecurringEntry, entry_id)
    if entry is None or entry.user_id != user_id or not entry.is_active:
        raise RecurringNotFoundError(f"Monthly entry not found: {entry_id}")
    return entry


def update_recurring_item(
    session: Session, user_id: str, entry_id: str, data: RecurringInput
) -> RecurringItem:
    """Change an entry from now on; months already recorded stay (committed).

    Raises:
        RecurringNotFoundError: If the entry or the chosen debt is not the user's.
    """
    entry = _owned_entry(session, user_id, entry_id)
    entry.debt_id = _owned_debt_id(session, user_id, data.debt_id)
    entry.transaction_type = data.transaction_type
    entry.category = data.category
    entry.amount = data.amount
    entry.day_of_month = data.day_of_month
    entry.description = data.description
    session.commit()
    return _to_item(session, entry)


def delete_recurring_item(session: Session, user_id: str, entry_id: str) -> None:
    """Stop an entry; months already recorded stay (committed).

    Raises:
        RecurringNotFoundError: If the entry is missing or not the user's.
    """
    _owned_entry(session, user_id, entry_id).is_active = False
    session.commit()
