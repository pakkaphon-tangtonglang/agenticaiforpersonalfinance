"""Database-integrated expense tracking service.

Orchestrates expense operations by querying transaction records
from the database and delegating computation to pure calculator functions.
"""

from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from finance_ai.database.crud.transaction_crud import TransactionCRUD
from finance_ai.database.models.transaction import Transaction
from finance_ai.tools.expense_calculator import (
    ExpenseRecord,
    ExpenseSummaryResult,
    summarize_expenses,
    validate_expense_amount,
    validate_expense_category,
)
from finance_ai.tools.expense_constants import EXPENSE_TRANSACTION_TYPE


def convert_transaction_to_expense_record(
    transaction: Transaction,
) -> ExpenseRecord:
    """Convert a Transaction ORM model to an ExpenseRecord.

    Args:
        transaction: Transaction model instance.

    Returns:
        ExpenseRecord suitable for pure calculator functions.

    Example:
        >>> record = convert_transaction_to_expense_record(transaction)
        >>> record.category
        'food'
    """
    return ExpenseRecord(
        amount=transaction.amount,
        category=transaction.category or "other",
        description=transaction.description or "",
        transaction_date=transaction.transaction_date,
    )


def create_expense_transaction(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    session: Session,
    user_id: str,
    amount: Decimal,
    category: str,
    description: str,
    transaction_date: date,
) -> Transaction:
    """Validate inputs and create an expense transaction record.

    Args:
        session: Database session.
        user_id: UUID of the user.
        amount: Expense amount in THB.
        category: Expense category key.
        description: Description of the expense.
        transaction_date: Date of the expense.

    Returns:
        Created Transaction instance.

    Raises:
        ValueError: If amount or category is invalid.

    Example:
        >>> txn = create_expense_transaction(
        ...     session, user_id, Decimal("80"), "food", "กาแฟ", date.today()
        ... )
    """
    validate_expense_amount(amount)
    normalized_category = validate_expense_category(category)
    crud = TransactionCRUD()
    transaction = crud.create(
        session,
        user_id=user_id,
        transaction_type=EXPENSE_TRANSACTION_TYPE,
        category=normalized_category,
        description=description,
        amount=amount,
        transaction_date=transaction_date,
    )
    session.commit()
    session.refresh(transaction)
    return transaction


def get_expenses_for_date_range(
    session: Session,
    user_id: str,
    start_date: date,
    end_date: date,
) -> list[ExpenseRecord]:
    """Query expense transactions for a user in a date range.

    Args:
        session: Database session.
        user_id: UUID of the user.
        start_date: Start of range (inclusive).
        end_date: End of range (inclusive).

    Returns:
        List of ExpenseRecord instances.

    Example:
        >>> expenses = get_expenses_for_date_range(
        ...     session, user_id, date(2026, 2, 1), date(2026, 2, 28)
        ... )
    """
    crud = TransactionCRUD()
    transactions = crud.get_by_user_type_and_date_range(
        session,
        user_id,
        EXPENSE_TRANSACTION_TYPE,
        start_date,
        end_date,
    )
    return [convert_transaction_to_expense_record(txn) for txn in transactions]


def get_expenses_by_category_and_date_range(
    session: Session,
    user_id: str,
    category: str,
    start_date: date,
    end_date: date,
) -> list[ExpenseRecord]:
    """Query expenses filtered by category and date range.

    Args:
        session: Database session.
        user_id: UUID of the user.
        category: Expense category key.
        start_date: Start of range (inclusive).
        end_date: End of range (inclusive).

    Returns:
        List of ExpenseRecord instances matching the category.

    Raises:
        ValueError: If category is invalid.

    Example:
        >>> expenses = get_expenses_by_category_and_date_range(
        ...     session, user_id, "food", date(2026, 2, 1), date(2026, 2, 28)
        ... )
    """
    normalized_category = validate_expense_category(category)
    crud = TransactionCRUD()
    transactions = crud.get_by_user_category_and_date_range(
        session,
        user_id,
        normalized_category,
        start_date,
        end_date,
    )
    return [convert_transaction_to_expense_record(txn) for txn in transactions]


def summarize_expenses_for_user(
    session: Session,
    user_id: str,
    start_date: date,
    end_date: date,
) -> ExpenseSummaryResult:
    """Summarize expenses for a user in a date range.

    Queries DB, converts to ExpenseRecord list, then delegates
    to the pure summarize_expenses function.

    Args:
        session: Database session.
        user_id: UUID of the user.
        start_date: Start of summary period (inclusive).
        end_date: End of summary period (inclusive).

    Returns:
        ExpenseSummaryResult with category breakdown.

    Example:
        >>> summary = summarize_expenses_for_user(
        ...     session, user_id, date(2026, 2, 1), date(2026, 2, 28)
        ... )
        >>> summary.total_amount
        Decimal('1500.00')
    """
    expenses = get_expenses_for_date_range(session, user_id, start_date, end_date)
    return summarize_expenses(expenses, start_date, end_date)
