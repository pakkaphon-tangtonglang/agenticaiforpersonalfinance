"""CRUD operations for the Income model."""

from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from finance_ai.database.crud.base_crud import BaseCRUD
from finance_ai.database.models.income import Income


class IncomeCRUD(BaseCRUD[Income]):
    """
    CRUD operations specific to Income model.

    Example:
        >>> income_crud = IncomeCRUD()
        >>> incomes = income_crud.get_by_user_and_year(session, user_id, 2024)
    """

    def __init__(self) -> None:
        """Initialize IncomeCRUD with Income model."""
        super().__init__(Income)

    def get_by_user_and_year(self, session: Session, user_id: str, tax_year: int) -> list[Income]:
        """
        Get all income records for a user in a specific tax year.

        Args:
            session: Database session.
            user_id: UUID of the user.
            tax_year: Tax year to filter by.

        Returns:
            List of Income instances.
        """
        statement = select(Income).where(Income.user_id == user_id, Income.tax_year == tax_year)
        return list(session.execute(statement).scalars().all())

    def get_total_income_for_year(self, session: Session, user_id: str, tax_year: int) -> Decimal:
        """
        Calculate total income for a user in a specific tax year.

        Args:
            session: Database session.
            user_id: UUID of the user.
            tax_year: Tax year to sum.

        Returns:
            Total income as Decimal. Returns Decimal("0") if no records.
        """
        statement = select(func.sum(Income.amount)).where(
            Income.user_id == user_id, Income.tax_year == tax_year
        )
        result = session.execute(statement).scalar()
        return Decimal(str(result)) if result is not None else Decimal("0")
