"""Tests for the User database model."""

from datetime import date

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from finance_ai.database.models.user import User


class TestUserModel:
    """Tests for User model creation, defaults, and constraints."""

    def test_create_user_with_all_fields(self, sample_user: User) -> None:
        """Test creating a user with all fields populated."""
        assert sample_user.email == "test@example.com"
        assert sample_user.full_name == "Test User"
        assert sample_user.tax_id == "1234567890123"
        assert sample_user.date_of_birth == date(1990, 1, 15)
        assert sample_user.marital_status == "single"
        assert sample_user.number_of_children == 0
        assert sample_user.number_of_parents == 2

    def test_create_user_with_defaults(self, sample_user_minimal: User) -> None:
        """Test that default values are correctly applied."""
        assert sample_user_minimal.marital_status == "single"
        assert sample_user_minimal.number_of_children == 0
        assert sample_user_minimal.number_of_parents == 0
        assert sample_user_minimal.is_active is True
        assert sample_user_minimal.tax_id is None
        assert sample_user_minimal.date_of_birth is None

    def test_user_has_uuid_primary_key(self, sample_user: User) -> None:
        """Test that user gets a UUID primary key."""
        assert sample_user.id is not None
        assert len(sample_user.id) == 36

    def test_user_has_timestamps(self, sample_user: User) -> None:
        """Test that user gets created_at and updated_at timestamps."""
        assert sample_user.created_at is not None
        assert sample_user.updated_at is not None

    def test_duplicate_email_raises_error(self, test_session: Session, sample_user: User) -> None:
        """Test that duplicate email addresses raise IntegrityError."""
        duplicate_user = User(
            email="test@example.com",
            hashed_password="another_hash",
            full_name="Duplicate User",
        )
        test_session.add(duplicate_user)
        with pytest.raises(IntegrityError):
            test_session.commit()

    def test_duplicate_tax_id_raises_error(self, test_session: Session, sample_user: User) -> None:
        """Test that duplicate tax IDs raise IntegrityError."""
        duplicate_user = User(
            email="other@example.com",
            hashed_password="another_hash",
            full_name="Other User",
            tax_id="1234567890123",
        )
        test_session.add(duplicate_user)
        with pytest.raises(IntegrityError):
            test_session.commit()

    def test_user_is_active_by_default(self, sample_user: User) -> None:
        """Test that new users are active by default."""
        assert sample_user.is_active is True

    def test_user_relationships_initialized_as_empty(self, sample_user: User) -> None:
        """Test that relationship lists start empty."""
        assert sample_user.incomes == []
        assert sample_user.deductions == []
        assert sample_user.investment_holdings == []
        assert sample_user.transactions == []
        assert sample_user.tax_filings == []
        assert sample_user.financial_goals == []
