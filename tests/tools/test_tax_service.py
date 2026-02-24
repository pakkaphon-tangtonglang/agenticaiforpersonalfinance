"""Tests for tax service with database integration."""

from decimal import Decimal

import pytest
from sqlalchemy.orm import Session

from finance_ai.database.crud.income_crud import IncomeCRUD
from finance_ai.database.crud.deduction_crud import DeductionCRUD
from finance_ai.database.crud.tax_filing_crud import TaxFilingCRUD
from finance_ai.database.models.user import User
from finance_ai.tools.tax_service import (
    aggregate_deductions_by_type,
    build_auto_allowances,
    calculate_tax_for_user,
    merge_deductions,
)


class TestBuildAutoAllowances:
    """Tests for automatic allowance generation."""

    def test_single_user_personal_only(self, sample_user: User) -> None:
        """Test that single user gets only personal allowance."""
        sample_user.marital_status = "single"
        sample_user.number_of_children = 0
        sample_user.number_of_parents = 0
        allowances = build_auto_allowances(sample_user)
        assert allowances == {"personal_allowance": Decimal("60000")}

    def test_married_user_with_family(self, sample_user: User) -> None:
        """Test married user gets spouse, child, and parent allowances."""
        sample_user.marital_status = "married"
        sample_user.number_of_children = 2
        sample_user.number_of_parents = 2
        allowances = build_auto_allowances(sample_user)
        assert allowances["personal_allowance"] == Decimal("60000")
        assert allowances["spouse_allowance"] == Decimal("60000")
        assert allowances["child_allowance"] == Decimal("60000")
        assert allowances["parent_allowance"] == Decimal("60000")


class TestAggregateDeductions:
    """Tests for deduction aggregation."""

    def test_empty_deductions(self) -> None:
        """Test aggregating empty list."""
        result = aggregate_deductions_by_type([])
        assert not result

    def test_aggregate_same_type(self, test_session: Session, sample_user: User) -> None:
        """Test aggregating multiple deductions of same type."""
        crud = DeductionCRUD()
        deduction_one = crud.create(
            test_session,
            user_id=sample_user.id,
            deduction_type="life_insurance",
            amount=Decimal("50000.00"),
            maximum_allowed=Decimal("100000.00"),
            tax_year=2024,
        )
        deduction_two = crud.create(
            test_session,
            user_id=sample_user.id,
            deduction_type="life_insurance",
            amount=Decimal("30000.00"),
            maximum_allowed=Decimal("100000.00"),
            tax_year=2024,
        )
        result = aggregate_deductions_by_type([deduction_one, deduction_two])
        assert result["life_insurance"] == Decimal("80000.00")


class TestMergeDeductions:
    """Tests for merging auto and user deductions."""

    def test_auto_overrides_user(self) -> None:
        """Test that auto allowances override user-entered ones."""
        auto = {"personal_allowance": Decimal("60000")}
        user = {"personal_allowance": Decimal("90000"), "rmf": Decimal("200000")}
        merged = merge_deductions(auto, user)
        assert merged["personal_allowance"] == Decimal("60000")
        assert merged["rmf"] == Decimal("200000")


class TestCalculateTaxForUser:
    """Integration tests for full tax calculation flow."""

    def test_user_with_salary_income(self, test_session: Session, sample_user: User) -> None:
        """Test full calculation with salary income and standard deductions."""
        income_crud = IncomeCRUD()
        income_crud.create(
            test_session,
            user_id=sample_user.id,
            income_type="salary",
            amount=Decimal("960000.00"),
            tax_year=2024,
            pay_period="annual",
            withholding_tax=Decimal("60000.00"),
        )
        deduction_crud = DeductionCRUD()
        deduction_crud.create(
            test_session,
            user_id=sample_user.id,
            deduction_type="social_security",
            amount=Decimal("9000.00"),
            maximum_allowed=Decimal("9000.00"),
            tax_year=2024,
        )
        result = calculate_tax_for_user(test_session, sample_user.id, 2024)
        assert result.gross_income == Decimal("960000.00")
        assert result.total_tax > Decimal("0")
        assert result.withholding_tax_paid == Decimal("60000.00")
        filing = TaxFilingCRUD().get_by_user_and_year(test_session, sample_user.id, 2024)
        assert filing is not None
        assert filing.gross_income == Decimal("960000.00")

    def test_user_with_no_income(self, test_session: Session, sample_user: User) -> None:
        """Test that user with no income gets zero tax."""
        result = calculate_tax_for_user(test_session, sample_user.id, 2024)
        assert result.gross_income == Decimal("0")
        assert result.total_tax == Decimal("0")
        assert result.net_income == Decimal("0")

    def test_user_not_found_raises_error(self, test_session: Session) -> None:
        """Test that nonexistent user raises ValueError."""
        with pytest.raises(ValueError, match="User not found"):
            calculate_tax_for_user(test_session, "nonexistent-id", 2024)

    def test_married_user_gets_spouse_allowance(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Test that married user auto-gets spouse allowance."""
        sample_user.marital_status = "married"
        test_session.commit()
        income_crud = IncomeCRUD()
        income_crud.create(
            test_session,
            user_id=sample_user.id,
            income_type="salary",
            amount=Decimal("500000.00"),
            tax_year=2024,
            pay_period="annual",
        )
        result = calculate_tax_for_user(test_session, sample_user.id, 2024)
        assert result.total_deductions >= Decimal("120000")

    def test_filing_is_updated_on_recalculation(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Test that recalculating updates existing filing."""
        income_crud = IncomeCRUD()
        income_crud.create(
            test_session,
            user_id=sample_user.id,
            income_type="salary",
            amount=Decimal("500000.00"),
            tax_year=2024,
            pay_period="annual",
        )
        first_result = calculate_tax_for_user(test_session, sample_user.id, 2024)
        income_crud.create(
            test_session,
            user_id=sample_user.id,
            income_type="freelance",
            amount=Decimal("200000.00"),
            tax_year=2024,
            pay_period="one_time",
        )
        second_result = calculate_tax_for_user(test_session, sample_user.id, 2024)
        assert second_result.gross_income > first_result.gross_income
        filings = TaxFilingCRUD().get_all_by_user(test_session, sample_user.id)
        assert len(filings) == 1
