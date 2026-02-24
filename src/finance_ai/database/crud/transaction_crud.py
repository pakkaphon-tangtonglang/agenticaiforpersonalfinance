"""CRUD operations for the Transaction model."""

from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.database.crud.base_crud import BaseCRUD
from finance_ai.database.models.transaction import Transaction


class TransactionCRUD(BaseCRUD[Transaction]):
    """
    CRUD operations specific to Transaction model.

    Example:
        >>> transaction_crud = TransactionCRUD()
        >>> txns = transaction_crud.get_by_holding(session, holding_id)
    """

    def __init__(self) -> None:
        """Initialize TransactionCRUD with Transaction model."""
        super().__init__(Transaction)

    def get_by_user_and_date_range(
        self,
        session: Session,
        user_id: str,
        start_date: date,
        end_date: date,
    ) -> list[Transaction]:
        """
        Get transactions for a user within a date range.

        Args:
            session: Database session.
            user_id: UUID of the user.
            start_date: Start of date range (inclusive).
            end_date: End of date range (inclusive).

        Returns:
            List of Transaction instances.
        """
        statement = select(Transaction).where(
            Transaction.user_id == user_id,
            Transaction.transaction_date >= start_date,
            Transaction.transaction_date <= end_date,
        )
        return list(session.execute(statement).scalars().all())

    def get_by_holding(self, session: Session, holding_id: str) -> list[Transaction]:
        """
        Get all transactions for a specific investment holding.

        Args:
            session: Database session.
            holding_id: UUID of the investment holding.

        Returns:
            List of Transaction instances.
        """
        statement = select(Transaction).where(Transaction.holding_id == holding_id)
        return list(session.execute(statement).scalars().all())

    def get_by_category(self, session: Session, user_id: str, category: str) -> list[Transaction]:
        """
        Get transactions filtered by category for a user.

        Args:
            session: Database session.
            user_id: UUID of the user.
            category: Transaction category (e.g., "food", "transport").

        Returns:
            List of matching Transaction instances.
        """
        statement = select(Transaction).where(
            Transaction.user_id == user_id,
            Transaction.category == category,
        )
        return list(session.execute(statement).scalars().all())

    def get_by_user_type_and_date_range(
        self,
        session: Session,
        user_id: str,
        transaction_type: str,
        start_date: date,
        end_date: date,
    ) -> list[Transaction]:
        """
        Get transactions filtered by user, type, and date range.

        Args:
            session: Database session.
            user_id: UUID of the user.
            transaction_type: Transaction type (e.g., "expense").
            start_date: Start of date range (inclusive).
            end_date: End of date range (inclusive).

        Returns:
            List of matching Transaction instances.
        """
        statement = select(Transaction).where(
            Transaction.user_id == user_id,
            Transaction.transaction_type == transaction_type,
            Transaction.transaction_date >= start_date,
            Transaction.transaction_date <= end_date,
        )
        return list(session.execute(statement).scalars().all())

    def get_by_user_category_and_date_range(
        self,
        session: Session,
        user_id: str,
        category: str,
        start_date: date,
        end_date: date,
    ) -> list[Transaction]:
        """
        Get transactions filtered by user, category, and date range.

        Args:
            session: Database session.
            user_id: UUID of the user.
            category: Transaction category (e.g., "food").
            start_date: Start of date range (inclusive).
            end_date: End of date range (inclusive).

        Returns:
            List of matching Transaction instances.
        """
        statement = select(Transaction).where(
            Transaction.user_id == user_id,
            Transaction.category == category,
            Transaction.transaction_date >= start_date,
            Transaction.transaction_date <= end_date,
        )
        return list(session.execute(statement).scalars().all())
