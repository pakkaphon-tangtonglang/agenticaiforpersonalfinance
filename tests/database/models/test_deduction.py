"""Tests for the Deduction database model."""

from decimal import Decimal

from sqlalchemy.orm import Session

from finance_ai.database.models.deduction import Deduction
from finance_ai.database.models.user import User


class TestDeductionModel:
    """Tests for Deduction model creation and relationships."""

    def test_create_deduction_record(self, test_session: Session, sample_user: User) -> None:
        """Test creating a deduction with all fields."""
        deduction = Deduction(
            user_id=sample_user.id,
            deduction_type="personal_allowance",
            description="Personal tax allowance",
            amount=Decimal("60000.00"),
            maximum_allowed=Decimal("60000.00"),
            tax_year=2024,
            document_reference="REF-001",
        )
        test_session.add(deduction)
        test_session.commit()
        test_session.refresh(deduction)
        assert deduction.id is not None
        assert deduction.amount == Decimal("60000.00")
        assert deduction.deduction_type == "personal_allowance"

    def test_deduction_relationship_to_user(self, test_session: Session, sample_user: User) -> None:
        """Test that deduction links back to its user."""
        deduction = Deduction(
            user_id=sample_user.id,
            deduction_type="social_security",
            amount=Decimal("9000.00"),
            maximum_allowed=Decimal("9000.00"),
            tax_year=2024,
        )
        test_session.add(deduction)
        test_session.commit()
        test_session.refresh(deduction)
        assert deduction.user.id == sample_user.id
        assert deduction in sample_user.deductions

    def test_multiple_deduction_types(self, test_session: Session, sample_user: User) -> None:
        """Test creating deductions of different types."""
        types_and_amounts = [
            ("rmf", Decimal("200000.00"), Decimal("500000.00")),
            ("ssf", Decimal("100000.00"), Decimal("200000.00")),
            ("life_insurance", Decimal("50000.00"), Decimal("100000.00")),
        ]
        for deduction_type, amount, max_allowed in types_and_amounts:
            deduction = Deduction(
                user_id=sample_user.id,
                deduction_type=deduction_type,
                amount=amount,
                maximum_allowed=max_allowed,
                tax_year=2024,
            )
            test_session.add(deduction)
        test_session.commit()
        assert len(sample_user.deductions) == 3

    def test_deduction_optional_fields(self, test_session: Session, sample_user: User) -> None:
        """Test that optional fields can be None."""
        deduction = Deduction(
            user_id=sample_user.id,
            deduction_type="donations",
            amount=Decimal("10000.00"),
            maximum_allowed=Decimal("100000.00"),
            tax_year=2024,
        )
        test_session.add(deduction)
        test_session.commit()
        test_session.refresh(deduction)
        assert deduction.description is None
        assert deduction.document_reference is None
