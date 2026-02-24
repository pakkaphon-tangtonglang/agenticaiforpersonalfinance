"""Tests for TaxFilingCRUD operations."""

from decimal import Decimal

from sqlalchemy.orm import Session

from finance_ai.database.crud.tax_filing_crud import TaxFilingCRUD
from finance_ai.database.models.user import User


class TestTaxFilingCRUD:
    """Tests for TaxFiling-specific CRUD operations."""

    def _create_filing(
        self,
        crud: TaxFilingCRUD,
        session: Session,
        user: User,
        tax_year: int = 2024,
    ) -> None:
        """Helper to create a test tax filing."""
        crud.create(
            session,
            user_id=user.id,
            tax_year=tax_year,
            gross_income=Decimal("960000.00"),
            total_deductions=Decimal("160000.00"),
            net_income=Decimal("800000.00"),
            total_tax=Decimal("75000.00"),
            effective_tax_rate=Decimal("0.0781"),
            tax_due_or_refund=Decimal("15000.00"),
        )

    def test_get_by_user_and_year(self, test_session: Session, sample_user: User) -> None:
        """Test finding a filing by user and year."""
        crud = TaxFilingCRUD()
        self._create_filing(crud, test_session, sample_user, 2024)
        found = crud.get_by_user_and_year(test_session, sample_user.id, 2024)
        assert found is not None
        assert found.tax_year == 2024

    def test_get_by_user_and_year_not_found(self, test_session: Session, sample_user: User) -> None:
        """Test that nonexistent year returns None."""
        crud = TaxFilingCRUD()
        found = crud.get_by_user_and_year(test_session, sample_user.id, 2099)
        assert found is None

    def test_get_all_by_user(self, test_session: Session, sample_user: User) -> None:
        """Test getting all filings for a user across years."""
        crud = TaxFilingCRUD()
        self._create_filing(crud, test_session, sample_user, 2023)
        self._create_filing(crud, test_session, sample_user, 2024)
        filings = crud.get_all_by_user(test_session, sample_user.id)
        assert len(filings) == 2
