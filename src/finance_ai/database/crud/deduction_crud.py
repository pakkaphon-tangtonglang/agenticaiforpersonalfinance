"""CRUD operations for the Deduction model."""

from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from finance_ai.database.crud.base_crud import BaseCRUD
from finance_ai.database.models.deduction import Deduction


class DeductionCRUD(BaseCRUD[Deduction]):
    """
    CRUD operations specific to Deduction model.

    Example:
        >>> deduction_crud = DeductionCRUD()
        >>> deductions = deduction_crud.get_by_user_and_year(session, user_id, 2024)
    """

    def __init__(self) -> None:
        """Initialize DeductionCRUD with Deduction model."""
        super().__init__(Deduction)

    def get_by_user_and_year(
        self, session: Session, user_id: str, tax_year: int
    ) -> list[Deduction]:
        """
        Get all deduction records for a user in a specific tax year.

        Args:
            session: Database session.
            user_id: UUID of the user.
            tax_year: Tax year to filter by.

        Returns:
            List of Deduction instances.
        """
        statement = select(Deduction).where(
            Deduction.user_id == user_id, Deduction.tax_year == tax_year
        )
        return list(session.execute(statement).scalars().all())

    def get_total_deductions_for_year(
        self, session: Session, user_id: str, tax_year: int
    ) -> Decimal:
        """
        Calculate total deductions for a user in a specific tax year.

        Args:
            session: Database session.
            user_id: UUID of the user.
            tax_year: Tax year to sum.

        Returns:
            Total deductions as Decimal. Returns Decimal("0") if no records.
        """
        statement = select(func.sum(Deduction.amount)).where(
            Deduction.user_id == user_id, Deduction.tax_year == tax_year
        )
        result = session.execute(statement).scalar()
        return Decimal(str(result)) if result is not None else Decimal("0")

    def get_by_type(
        self,
        session: Session,
        user_id: str,
        tax_year: int,
        deduction_type: str,
    ) -> list[Deduction]:
        """
        Get deductions filtered by type for a user and year.

        Args:
            session: Database session.
            user_id: UUID of the user.
            tax_year: Tax year to filter by.
            deduction_type: Type of deduction to filter.

        Returns:
            List of matching Deduction instances.
        """
        statement = select(Deduction).where(
            Deduction.user_id == user_id,
            Deduction.tax_year == tax_year,
            Deduction.deduction_type == deduction_type,
        )
        return list(session.execute(statement).scalars().all())
