"""Tests for DeductionCRUD operations."""

from decimal import Decimal

from sqlalchemy.orm import Session

from finance_ai.database.crud.deduction_crud import DeductionCRUD
from finance_ai.database.models.user import User


class TestDeductionCRUD:
    """Tests for Deduction-specific CRUD operations."""

    def test_get_by_user_and_year(self, test_session: Session, sample_user: User) -> None:
        """Test filtering deductions by user and year."""
        crud = DeductionCRUD()
        crud.create(
            test_session,
            user_id=sample_user.id,
            deduction_type="personal_allowance",
            amount=Decimal("60000.00"),
            maximum_allowed=Decimal("60000.00"),
            tax_year=2024,
        )
        crud.create(
            test_session,
            user_id=sample_user.id,
            deduction_type="social_security",
            amount=Decimal("9000.00"),
            maximum_allowed=Decimal("9000.00"),
            tax_year=2023,
        )
        results = crud.get_by_user_and_year(test_session, sample_user.id, 2024)
        assert len(results) == 1

    def test_get_total_deductions_for_year(self, test_session: Session, sample_user: User) -> None:
        """Test summing deductions for a specific year."""
        crud = DeductionCRUD()
        crud.create(
            test_session,
            user_id=sample_user.id,
            deduction_type="personal_allowance",
            amount=Decimal("60000.00"),
            maximum_allowed=Decimal("60000.00"),
            tax_year=2024,
        )
        crud.create(
            test_session,
            user_id=sample_user.id,
            deduction_type="social_security",
            amount=Decimal("9000.00"),
            maximum_allowed=Decimal("9000.00"),
            tax_year=2024,
        )
        total = crud.get_total_deductions_for_year(test_session, sample_user.id, 2024)
        assert total == Decimal("69000.00")

    def test_get_total_deductions_no_records(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Test that total deductions returns zero when no records exist."""
        crud = DeductionCRUD()
        total = crud.get_total_deductions_for_year(test_session, sample_user.id, 2024)
        assert total == Decimal("0")

    def test_get_by_type(self, test_session: Session, sample_user: User) -> None:
        """Test filtering deductions by type."""
        crud = DeductionCRUD()
        crud.create(
            test_session,
            user_id=sample_user.id,
            deduction_type="rmf",
            amount=Decimal("200000.00"),
            maximum_allowed=Decimal("500000.00"),
            tax_year=2024,
        )
        crud.create(
            test_session,
            user_id=sample_user.id,
            deduction_type="ssf",
            amount=Decimal("100000.00"),
            maximum_allowed=Decimal("200000.00"),
            tax_year=2024,
        )
        rmf_results = crud.get_by_type(test_session, sample_user.id, 2024, "rmf")
        assert len(rmf_results) == 1
        assert rmf_results[0].deduction_type == "rmf"
