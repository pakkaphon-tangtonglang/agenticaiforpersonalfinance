"""CRUD operations for the TaxFiling model."""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.database.crud.base_crud import BaseCRUD
from finance_ai.database.models.tax_filing import TaxFiling


class TaxFilingCRUD(BaseCRUD[TaxFiling]):
    """
    CRUD operations specific to TaxFiling model.

    Example:
        >>> tax_filing_crud = TaxFilingCRUD()
        >>> filing = tax_filing_crud.get_by_user_and_year(session, user_id, 2024)
    """

    def __init__(self) -> None:
        """Initialize TaxFilingCRUD with TaxFiling model."""
        super().__init__(TaxFiling)

    def get_by_user_and_year(
        self, session: Session, user_id: str, tax_year: int
    ) -> Optional[TaxFiling]:
        """
        Get a tax filing for a specific user and tax year.

        Args:
            session: Database session.
            user_id: UUID of the user.
            tax_year: Tax year to look up.

        Returns:
            TaxFiling instance or None if not found.
        """
        statement = select(TaxFiling).where(
            TaxFiling.user_id == user_id, TaxFiling.tax_year == tax_year
        )
        return session.execute(statement).scalar_one_or_none()

    def get_all_by_user(self, session: Session, user_id: str) -> list[TaxFiling]:
        """
        Get all tax filings for a user across all years.

        Args:
            session: Database session.
            user_id: UUID of the user.

        Returns:
            List of TaxFiling instances.
        """
        statement = select(TaxFiling).where(TaxFiling.user_id == user_id)
        return list(session.execute(statement).scalars().all())
