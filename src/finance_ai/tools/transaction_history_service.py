"""Income/expense history for the dashboard: list, add, edit, delete.

Entries are rows of the ``transactions`` table with ``transaction_type``
``income`` or ``expense`` (investment buy/sell rows belong to the assets page
and are never listed or changed here). Every income entry also has a tax copy
in ``incomes`` (used by the tax agent); this service keeps that copy in step
on every create, edit and delete.
"""

import calendar
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.database.models.income import Income
from finance_ai.database.models.transaction import Transaction
from finance_ai.tools.expense_constants import (
    EXPENSE_CATEGORIES,
    INCOME_TYPES,
    MAX_SINGLE_EXPENSE_AMOUNT,
)

EntryType = Literal["income", "expense"]
HistoryKind = Literal["all", "income", "expense"]
ENTRY_TYPES: tuple[str, ...] = ("income", "expense")
CENT = Decimal("0.01")
INCOME_PAY_PERIOD = "monthly"


class EntryNotFoundError(LookupError):
    """The entry does not exist, is not the user's, or is not income/expense."""


class EntryInput(BaseModel):
    """A new or edited history entry, validated before any database work."""

    transaction_type: EntryType
    transaction_date: date
    amount: Decimal = Field(gt=0, le=MAX_SINGLE_EXPENSE_AMOUNT)
    category: str
    description: str = Field(default="", max_length=255)

    @field_validator("amount")
    @classmethod
    def _to_satang(cls, value: Decimal) -> Decimal:
        """Store money with 2 decimal places."""
        return value.quantize(CENT, rounding=ROUND_HALF_UP)

    @field_validator("transaction_date")
    @classmethod
    def _not_in_future(cls, value: date) -> date:
        """The history holds what actually happened, so no future dates."""
        if value > date.today():
            raise ValueError(
                f"Date cannot be in the future (history records actual entries). Received: {value}"
            )
        return value

    @model_validator(mode="after")
    def _category_matches_type(self) -> "EntryInput":
        """Expenses use expense categories; income uses income types."""
        allowed = INCOME_TYPES if self.transaction_type == "income" else EXPENSE_CATEGORIES
        if self.category not in allowed:
            raise ValueError(
                f"Unknown {self.transaction_type} category '{self.category}'. "
                f"Allowed: {', '.join(allowed)}"
            )
        return self


class HistoryEntry(BaseModel):
    """One income or expense row as shown on the dashboard."""

    id: str
    transaction_type: EntryType
    transaction_date: date
    category: str
    category_label: str
    description: str
    amount: Decimal


class HistoryPage(BaseModel):
    """A month of entries, newest first, with the month's totals."""

    entries: list[HistoryEntry]
    total_income: Decimal
    total_expense: Decimal


def _label(entry_type: str, category: Optional[str]) -> str:
    """Thai label for a category key (the key itself when unknown)."""
    labels = INCOME_TYPES if entry_type == "income" else EXPENSE_CATEGORIES
    return labels.get(category or "other", category or "")


def _to_entry(txn: Transaction) -> HistoryEntry:
    """Convert a transactions row into a dashboard entry."""
    return HistoryEntry(
        id=txn.id,
        transaction_type="income" if txn.transaction_type == "income" else "expense",
        transaction_date=txn.transaction_date,
        category=txn.category or "other",
        category_label=_label(txn.transaction_type, txn.category),
        description=txn.description or "",
        amount=Decimal(txn.amount).quantize(CENT),
    )


def _month_rows(session: Session, user_id: str, year: int, month: int) -> list[Transaction]:
    """Income/expense rows of one month, newest first (investment rows excluded)."""
    first = date(year, month, 1)
    last = date(year, month, calendar.monthrange(year, month)[1])
    statement = (
        select(Transaction)
        .where(Transaction.user_id == user_id)
        .where(Transaction.transaction_type.in_(ENTRY_TYPES))
        .where(Transaction.holding_id.is_(None))
        .where(Transaction.transaction_date.between(first, last))
        .order_by(Transaction.transaction_date.desc(), Transaction.created_at.desc())
    )
    return list(session.execute(statement).scalars())


def list_entries(
    session: Session, user_id: str, year: int, month: int, kind: HistoryKind = "all"
) -> HistoryPage:
    """List a month's income and expenses for the dashboard.

    Args:
        session: Database session.
        user_id: Owner of the entries.
        year: Year to show.
        month: Month to show (1-12).
        kind: "all", "income" or "expense".

    Returns:
        HistoryPage with entries (newest first) and the month's totals.

    Example:
        >>> list_entries(session, uid, 2026, 10).total_expense
        Decimal('120.00')
    """
    entries = [_to_entry(txn) for txn in _month_rows(session, user_id, year, month)]
    total = {
        kind_: sum((e.amount for e in entries if e.transaction_type == kind_), Decimal("0"))
        for kind_ in ENTRY_TYPES
    }
    shown = entries if kind == "all" else [e for e in entries if e.transaction_type == kind]
    return HistoryPage(
        entries=shown,
        total_income=total["income"].quantize(CENT),
        total_expense=total["expense"].quantize(CENT),
    )


def _owned_entry(session: Session, user_id: str, entry_id: str) -> Transaction:
    """Load an income/expense row owned by the user, or raise EntryNotFoundError."""
    txn = session.get(Transaction, entry_id)
    if txn is None or txn.user_id != user_id or txn.transaction_type not in ENTRY_TYPES:
        raise EntryNotFoundError(f"Entry not found: {entry_id}")
    if txn.holding_id is not None:
        raise EntryNotFoundError(f"Entry not found: {entry_id}")
    return txn


def _tax_copy(session: Session, txn: Transaction) -> Optional[Income]:
    """The income tax copy for a transaction: linked one, else a legacy match."""
    linked = (
        session.execute(select(Income).where(Income.transaction_id == txn.id)).scalars().first()
    )
    if linked is not None:
        return linked
    legacy = (
        select(Income)
        .where(Income.user_id == txn.user_id, Income.transaction_id.is_(None))
        .where(Income.tax_year == txn.transaction_date.year)
        .where(Income.description == txn.description, Income.amount == txn.amount)
    )
    return session.execute(legacy).scalars().first()


def _write_tax_copy(session: Session, txn: Transaction, copy: Optional[Income]) -> None:
    """Create or update the tax copy so it mirrors an income transaction."""
    if copy is None:
        copy = Income(user_id=txn.user_id, pay_period=INCOME_PAY_PERIOD)
        session.add(copy)
    copy.transaction_id = txn.id
    copy.income_type = txn.category or "other"
    copy.description = txn.description
    copy.amount = txn.amount
    copy.tax_year = txn.transaction_date.year


def _apply(txn: Transaction, data: EntryInput) -> None:
    """Copy validated input onto a transactions row."""
    txn.transaction_type = data.transaction_type
    txn.transaction_date = data.transaction_date
    txn.amount = data.amount
    txn.category = data.category
    txn.description = data.description or _label(data.transaction_type, data.category)


def create_entry(session: Session, user_id: str, data: EntryInput) -> HistoryEntry:
    """Add an income or expense entry (and its tax copy for income).

    Args:
        session: Database session (committed here).
        user_id: Owner of the entry.
        data: Validated entry input.

    Returns:
        The created entry.

    Example:
        >>> create_entry(session, uid, EntryInput(...)).id
    """
    txn = Transaction(user_id=user_id)
    _apply(txn, data)
    session.add(txn)
    session.flush()
    if data.transaction_type == "income":
        _write_tax_copy(session, txn, None)
    session.commit()
    return _to_entry(txn)


def update_entry(session: Session, user_id: str, entry_id: str, data: EntryInput) -> HistoryEntry:
    """Edit an entry and keep its income tax copy in step.

    Args:
        session: Database session (committed here).
        user_id: Owner of the entry.
        entry_id: Transaction id.
        data: Validated replacement values.

    Returns:
        The updated entry.

    Raises:
        EntryNotFoundError: If the entry is missing or not the user's.
    """
    txn = _owned_entry(session, user_id, entry_id)
    copy = _tax_copy(session, txn) if txn.transaction_type == "income" else None
    _apply(txn, data)
    if data.transaction_type == "income":
        _write_tax_copy(session, txn, copy)
    elif copy is not None:
        session.delete(copy)
    session.commit()
    return _to_entry(txn)


def delete_entry(session: Session, user_id: str, entry_id: str) -> None:
    """Delete an entry and, for income, its tax copy.

    Args:
        session: Database session (committed here).
        user_id: Owner of the entry.
        entry_id: Transaction id.

    Raises:
        EntryNotFoundError: If the entry is missing or not the user's.
    """
    txn = _owned_entry(session, user_id, entry_id)
    copy = _tax_copy(session, txn) if txn.transaction_type == "income" else None
    if copy is not None:
        session.delete(copy)
    session.delete(txn)
    session.commit()
