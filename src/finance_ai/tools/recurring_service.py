"""Monthly recurring income and expenses (rent received, loan installments...).

Each due month becomes an ordinary transaction (income also gets an Income
row for tax). Posting runs when the user uses the app and once a day, and
catches up months missed while the server slept; last_posted_on makes it
safe to run any number of times.
"""

import calendar
from datetime import date, datetime
from decimal import Decimal
from typing import Optional
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.database.crud.income_crud import IncomeCRUD
from finance_ai.database.crud.transaction_crud import TransactionCRUD
from finance_ai.database.models.recurring_entry import RecurringEntry
from finance_ai.database.models.debt import Debt
from finance_ai.tools.debt_service import apply_debt_payment
from finance_ai.tools.expense_calculator import validate_expense_category
from finance_ai.tools.expense_constants import INCOME_TYPES

ENTRY_TYPES: frozenset[str] = frozenset({"income", "expense"})
MAX_DAY_OF_MONTH: int = 31
BANGKOK = ZoneInfo("Asia/Bangkok")


def bangkok_today() -> date:
    """Today's date in Thailand (the server clock runs in UTC).

    Example:
        >>> isinstance(bangkok_today(), date)
        True
    """
    return datetime.now(BANGKOK).date()


def due_dates(
    day_of_month: int, start_date: date, last_posted_on: Optional[date], today: date
) -> list[date]:
    """Monthly dates on or after start (and after the last posting) up to today.

    Days past a month's end fall on its last day (31 → 28 Feb).

    Example:
        >>> due_dates(3, date(2026, 11, 3), None, date(2026, 11, 3))
        [datetime.date(2026, 11, 3)]
    """
    dates: list[date] = []
    year, month = start_date.year, start_date.month
    while (year, month) <= (today.year, today.month):
        occurrence = date(year, month, min(day_of_month, calendar.monthrange(year, month)[1]))
        after_last = last_posted_on is None or occurrence > last_posted_on
        if start_date <= occurrence <= today and after_last:
            dates.append(occurrence)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return dates


def _normalize_category(entry_type: str, category: str) -> str:
    """Validated expense category, or a known income type (unknown → other)."""
    if entry_type == "expense":
        return validate_expense_category(category)
    normalized = category.strip().lower()
    return normalized if normalized in INCOME_TYPES else "other"


def _validate_entry(entry_type: str, amount: Decimal, day_of_month: int) -> None:
    """Reject unknown types, non-positive amounts and impossible days."""
    if entry_type not in ENTRY_TYPES:
        raise ValueError(f"Entry type must be income or expense. Received: '{entry_type}'")
    if amount <= 0:
        raise ValueError(f"Amount must be greater than 0. Received: {amount} THB")
    if not 1 <= day_of_month <= MAX_DAY_OF_MONTH:
        raise ValueError(f"Day of month must be 1-31. Received: {day_of_month}")


def create_recurring_entry(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    session: Session,
    user_id: str,
    entry_type: str,
    category: str,
    amount: Decimal,
    day_of_month: int,
    description: str,
    start_date: date,
    *,
    debt_id: Optional[str] = None,
) -> RecurringEntry:
    """Save a monthly income or expense (committed).

    Args:
        session: Database session.
        user_id: UUID of the user.
        entry_type: "income" or "expense".
        category: Income type (e.g. "rental") or expense category (e.g. "debt").
        amount: Amount per month in THB.
        day_of_month: Day it happens each month (1-31).
        description: Note shown on each transaction.
        start_date: First day it may happen.
        debt_id: Debt each posting pays down (installments only).

    Returns:
        The saved entry.

    Raises:
        ValueError: If the type, amount, day or expense category is invalid.

    Example:
        >>> create_recurring_entry(session, uid, "income", "rental", Decimal("8000"),
        ...     3, "ค่าเช่าบ้าน", date(2026, 11, 3)).day_of_month
        3
    """
    normalized_type = entry_type.strip().lower()
    _validate_entry(normalized_type, amount, day_of_month)
    entry = RecurringEntry(
        user_id=user_id,
        transaction_type=normalized_type,
        category=_normalize_category(normalized_type, category),
        description=description,
        amount=amount,
        day_of_month=day_of_month,
        start_date=start_date,
        is_active=True,
        debt_id=debt_id,
    )
    session.add(entry)
    session.commit()
    return entry


def _post_occurrence(session: Session, entry: RecurringEntry, occurrence: date) -> None:
    """One transaction for the month (plus an Income row for income)."""
    transaction = TransactionCRUD().create(
        session,
        user_id=entry.user_id,
        transaction_type=entry.transaction_type,
        category=entry.category,
        description=entry.description or entry.category,
        amount=entry.amount,
        transaction_date=occurrence,
    )
    if entry.transaction_type == "income":
        IncomeCRUD().create(
            session,
            user_id=entry.user_id,
            income_type=entry.category,
            description=entry.description,
            amount=entry.amount,
            tax_year=occurrence.year,
            pay_period="monthly",
            withholding_tax=Decimal("0"),
            transaction_id=transaction.id,
        )


def _active_entries(session: Session, user_id: Optional[str]) -> list[RecurringEntry]:
    """Active entries for one user, or for everyone when user_id is None."""
    statement = select(RecurringEntry).where(RecurringEntry.is_active.is_(True))
    if user_id is not None:
        statement = statement.where(RecurringEntry.user_id == user_id)
    return list(session.execute(statement).scalars().all())


def post_due_entries(session: Session, today: date, user_id: Optional[str] = None) -> int:
    """Turn every due month into a transaction (committed); safe to repeat.

    Args:
        session: Database session.
        today: Post occurrences up to and including this date.
        user_id: Only this user's entries; None means all users.

    Returns:
        Number of transactions created.

    Example:
        >>> post_due_entries(session, date(2026, 11, 3), uid)
        1
    """
    posted = sum(_post_entry(session, entry, today) for entry in _active_entries(session, user_id))
    session.commit()
    return posted


def _post_entry(session: Session, entry: RecurringEntry, today: date) -> int:
    """Post one entry's due months; an installment stops when its debt is gone."""
    posted = 0
    for occurrence in due_dates(entry.day_of_month, entry.start_date, entry.last_posted_on, today):
        debt = _linked_open_debt(session, entry)
        if entry.debt_id is not None and debt is None:
            entry.is_active = False
            break
        _post_occurrence(session, entry, occurrence)
        entry.last_posted_on = occurrence
        posted += 1
        if debt is not None and apply_debt_payment(session, debt, entry.amount).paid_off:
            entry.is_active = False
            break
    return posted


def _linked_open_debt(session: Session, entry: RecurringEntry) -> Optional[Debt]:
    """The debt this installment pays down, if it is still open."""
    if entry.debt_id is None:
        return None
    debt = session.get(Debt, entry.debt_id)
    return None if debt is None or debt.is_paid_off else debt


def list_recurring_entries(session: Session, user_id: str) -> list[RecurringEntry]:
    """The user's active recurring entries, oldest first.

    Example:
        >>> [entry.description for entry in list_recurring_entries(session, uid)]
        ['ค่าเช่าบ้าน']
    """
    entries = _active_entries(session, user_id)
    return sorted(entries, key=lambda entry: entry.created_at)


def cancel_recurring_entry(
    session: Session, user_id: str, search_term: str
) -> Optional[RecurringEntry]:
    """Stop the entry whose description or category matches (the latest if blank).

    Args:
        session: Database session (committed here).
        user_id: UUID of the user.
        search_term: Word from the description or category, e.g. "เช่า".

    Returns:
        The cancelled entry, or None when nothing matches.

    Example:
        >>> cancel_recurring_entry(session, uid, "เช่า").is_active
        False
    """
    entries = list_recurring_entries(session, user_id)
    term = search_term.strip().lower()
    matches = [e for e in entries if term in e.description.lower() or term in e.category]
    if not matches:
        return None
    entry = matches[-1]
    entry.is_active = False
    session.commit()
    return entry
