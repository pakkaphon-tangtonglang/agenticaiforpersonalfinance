"""Tests for IncomeCRUD operations."""

from decimal import Decimal

from sqlalchemy.orm import Session

from finance_ai.database.crud.income_crud import IncomeCRUD
from finance_ai.database.models.user import User


class TestIncomeCRUD:
    """Tests for Income-specific CRUD operations."""

    def test_get_by_user_and_year(self, test_session: Session, sample_user: User) -> None:
        """Test filtering income records by user and year."""
        crud = IncomeCRUD()
        crud.create(
            test_session,
            user_id=sample_user.id,
            income_type="salary",
            amount=Decimal("80000.00"),
            tax_year=2024,
            pay_period="monthly",
        )
        crud.create(
            test_session,
            user_id=sample_user.id,
            income_type="freelance",
            amount=Decimal("20000.00"),
            tax_year=2023,
            pay_period="one_time",
        )
        results = crud.get_by_user_and_year(test_session, sample_user.id, 2024)
        assert len(results) == 1
        assert results[0].income_type == "salary"

    def test_get_total_income_for_year(self, test_session: Session, sample_user: User) -> None:
        """Test summing income for a specific year."""
        crud = IncomeCRUD()
        crud.create(
            test_session,
            user_id=sample_user.id,
            income_type="salary",
            amount=Decimal("80000.00"),
            tax_year=2024,
            pay_period="monthly",
        )
        crud.create(
            test_session,
            user_id=sample_user.id,
            income_type="freelance",
            amount=Decimal("30000.00"),
            tax_year=2024,
            pay_period="one_time",
        )
        total = crud.get_total_income_for_year(test_session, sample_user.id, 2024)
        assert total == Decimal("110000.00")

    def test_get_total_income_no_records(self, test_session: Session, sample_user: User) -> None:
        """Test that total income returns zero when no records exist."""
        crud = IncomeCRUD()
        total = crud.get_total_income_for_year(test_session, sample_user.id, 2024)
        assert total == Decimal("0")
