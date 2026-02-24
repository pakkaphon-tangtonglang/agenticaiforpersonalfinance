"""Tests for UserCRUD operations."""

from sqlalchemy.orm import Session

from finance_ai.database.crud.user_crud import UserCRUD
from finance_ai.database.models.user import User


class TestUserCRUD:
    """Tests for User-specific CRUD operations."""

    def test_get_by_email(self, test_session: Session, sample_user: User) -> None:
        """Test finding a user by email."""
        crud = UserCRUD()
        found = crud.get_by_email(test_session, "test@example.com")
        assert found is not None
        assert found.id == sample_user.id

    def test_get_by_email_not_found(self, test_session: Session) -> None:
        """Test that nonexistent email returns None."""
        crud = UserCRUD()
        found = crud.get_by_email(test_session, "nobody@example.com")
        assert found is None

    def test_get_by_tax_id(self, test_session: Session, sample_user: User) -> None:
        """Test finding a user by Thai tax ID."""
        crud = UserCRUD()
        found = crud.get_by_tax_id(test_session, "1234567890123")
        assert found is not None
        assert found.id == sample_user.id

    def test_get_by_tax_id_not_found(self, test_session: Session) -> None:
        """Test that nonexistent tax ID returns None."""
        crud = UserCRUD()
        found = crud.get_by_tax_id(test_session, "0000000000000")
        assert found is None

    def test_get_active_users(self, test_session: Session, sample_user: User) -> None:
        """Test retrieving only active users."""
        crud = UserCRUD()
        inactive_user = crud.create(
            test_session,
            email="inactive@example.com",
            hashed_password="hash",
            full_name="Inactive",
            is_active=False,
        )
        active = crud.get_active_users(test_session)
        active_ids = [u.id for u in active]
        assert sample_user.id in active_ids
        assert inactive_user.id not in active_ids

    def test_deactivate_user(self, test_session: Session, sample_user: User) -> None:
        """Test soft-deleting a user."""
        crud = UserCRUD()
        result = crud.deactivate(test_session, sample_user.id)
        assert result is True
        test_session.refresh(sample_user)
        assert sample_user.is_active is False

    def test_deactivate_nonexistent_user(self, test_session: Session) -> None:
        """Test that deactivating a nonexistent user returns False."""
        crud = UserCRUD()
        result = crud.deactivate(test_session, "fake-id")
        assert result is False
