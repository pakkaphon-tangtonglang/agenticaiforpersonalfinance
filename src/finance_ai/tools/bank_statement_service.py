"""Bank statement import service.

Handles bulk insertion of parsed transactions into the database
with duplicate detection. Income transactions are dual-written to
both Transaction and Income tables.
"""

from collections.abc import Callable
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.tools.bank_statement_parser import ParsedTransaction


def bulk_insert_transactions(
    transactions: list[ParsedTransaction],
    user_id: str,
    db_session_factory: Callable[[], Session],
) -> int:
    """Bulk insert parsed transactions into the database.

    Args:
        transactions: List of transactions to insert.
        user_id: UUID of the user.
        db_session_factory: Session factory callable.

    Returns:
        Number of transactions inserted (duplicates skipped).

    Example:
        >>> count = bulk_insert_transactions(txns, "user-123", factory)
        >>> print(f"Inserted {count} transactions")
    """
    from finance_ai.database.models.income import Income  # noqa: PLC0415
    from finance_ai.database.models.transaction import Transaction  # noqa: PLC0415

    session = db_session_factory()
    count = 0

    try:
        for txn in transactions:
            inserted = _insert_single_transaction(
                session,
                txn,
                user_id,
                Transaction,
                Income,
            )
            if inserted:
                count += 1
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return count


def _insert_single_transaction(
    session: Session,
    txn: ParsedTransaction,
    user_id: str,
    transaction_cls: Any,
    income_cls: Any,
) -> bool:
    """Insert a single transaction if not a duplicate.

    Skips records that already exist (same user, date, description, amount).
    Income transactions are also written to the incomes table.

    Args:
        session: Database session.
        txn: Parsed transaction to insert.
        user_id: UUID of the user.
        transaction_cls: Transaction model class.
        income_cls: Income model class.

    Returns:
        True if inserted, False if skipped as duplicate.
    """
    existing: Any = session.execute(
        select(transaction_cls).where(
            transaction_cls.user_id == user_id,
            transaction_cls.transaction_date == txn.transaction_date,
            transaction_cls.description == txn.description,
            transaction_cls.amount == txn.amount,
        )
    ).first()
    if existing:
        return False

    record = transaction_cls(
        user_id=user_id,
        transaction_type=txn.transaction_type,
        category=txn.category,
        amount=txn.amount,
        description=txn.description,
        transaction_date=txn.transaction_date,
    )
    session.add(record)

    if txn.transaction_type == "income":
        _insert_income_record(session, txn, user_id, income_cls)

    return True


def _insert_income_record(
    session: Session,
    txn: ParsedTransaction,
    user_id: str,
    income_cls: Any,
) -> None:
    """Insert an income record if not a duplicate.

    Args:
        session: Database session.
        txn: Parsed transaction (must be income type).
        user_id: UUID of the user.
        income_cls: Income model class.
    """
    duplicate_income: Any = session.execute(
        select(income_cls).where(
            income_cls.user_id == user_id,
            income_cls.tax_year == txn.transaction_date.year,
            income_cls.description == txn.description,
            income_cls.amount == txn.amount,
        )
    ).first()
    if not duplicate_income:
        session.add(
            income_cls(
                user_id=user_id,
                income_type="other",
                description=txn.description,
                amount=txn.amount,
                tax_year=txn.transaction_date.year,
                pay_period="monthly",
            )
        )
