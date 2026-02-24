"""Tests for the Income database model."""

from decimal import Decimal

from sqlalchemy.orm import Session

from finance_ai.database.models.income import Income
from finance_ai.database.models.user import User


class TestIncomeModel:
    """Tests for Income model creation, defaults, and relationships."""

    def test_create_income_record(self, test_session: Session, sample_user: User) -> None:
        """Test creating an income record with all fields."""
        income = Income(
            user_id=sample_user.id,
            income_type="salary",
            description="Monthly salary",
            amount=Decimal("80000.00"),
            tax_year=2024,
            pay_period="monthly",
            employer_name="Test Company",
            withholding_tax=Decimal("5000.00"),
        )
        test_session.add(income)
        test_session.commit()
        test_session.refresh(income)
        assert income.id is not None
        assert income.amount == Decimal("80000.00")
        assert income.income_type == "salary"
        assert income.withholding_tax == Decimal("5000.00")

    def test_income_default_withholding_tax(self, test_session: Session, sample_user: User) -> None:
        """Test that withholding_tax defaults to zero."""
        income = Income(
            user_id=sample_user.id,
            income_type="freelance",
            amount=Decimal("30000.00"),
            tax_year=2024,
            pay_period="one_time",
        )
        test_session.add(income)
        test_session.commit()
        test_session.refresh(income)
        assert income.withholding_tax == Decimal("0")

    def test_income_relationship_to_user(self, test_session: Session, sample_user: User) -> None:
        """Test that income links back to its user."""
        income = Income(
            user_id=sample_user.id,
            income_type="salary",
            amount=Decimal("80000.00"),
            tax_year=2024,
            pay_period="monthly",
        )
        test_session.add(income)
        test_session.commit()
        test_session.refresh(income)
        assert income.user.id == sample_user.id
        assert income in sample_user.incomes

    def test_income_has_timestamps(self, test_session: Session, sample_user: User) -> None:
        """Test that income gets timestamps."""
        income = Income(
            user_id=sample_user.id,
            income_type="dividend",
            amount=Decimal("5000.00"),
            tax_year=2024,
            pay_period="annual",
        )
        test_session.add(income)
        test_session.commit()
        test_session.refresh(income)
        assert income.created_at is not None
        assert income.updated_at is not None

    def test_income_decimal_precision(self, test_session: Session, sample_user: User) -> None:
        """Test that Decimal amounts are stored with correct precision."""
        income = Income(
            user_id=sample_user.id,
            income_type="interest",
            amount=Decimal("12345.67"),
            tax_year=2024,
            pay_period="annual",
        )
        test_session.add(income)
        test_session.commit()
        test_session.refresh(income)
        assert income.amount == Decimal("12345.67")
