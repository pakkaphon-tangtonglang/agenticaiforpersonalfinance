"""Tests for the BaseCRUD generic operations."""

from sqlalchemy.orm import Session

from finance_ai.database.crud.base_crud import BaseCRUD
from finance_ai.database.models.user import User


class TestBaseCRUD:
    """Tests for generic CRUD operations using User model."""

    def test_create_record(self, test_session: Session) -> None:
        """Test creating a record via BaseCRUD."""
        crud = BaseCRUD(User)
        user = crud.create(
            test_session,
            email="crud@example.com",
            hashed_password="hash123",
            full_name="CRUD User",
        )
        assert user.id is not None
        assert user.email == "crud@example.com"

    def test_get_by_id(self, test_session: Session, sample_user: User) -> None:
        """Test retrieving a record by ID."""
        crud = BaseCRUD(User)
        found = crud.get_by_id(test_session, sample_user.id)
        assert found is not None
        assert found.email == sample_user.email

    def test_get_by_id_not_found(self, test_session: Session) -> None:
        """Test that get_by_id returns None for nonexistent ID."""
        crud = BaseCRUD(User)
        found = crud.get_by_id(test_session, "nonexistent-uuid")
        assert found is None

    def test_get_all(self, test_session: Session) -> None:
        """Test retrieving all records with pagination."""
        crud = BaseCRUD(User)
        for i in range(5):
            crud.create(
                test_session,
                email=f"user{i}@example.com",
                hashed_password="hash",
                full_name=f"User {i}",
            )
        all_users = crud.get_all(test_session)
        assert len(all_users) == 5

    def test_get_all_with_pagination(self, test_session: Session) -> None:
        """Test skip and limit in get_all."""
        crud = BaseCRUD(User)
        for i in range(5):
            crud.create(
                test_session,
                email=f"page{i}@example.com",
                hashed_password="hash",
                full_name=f"Page User {i}",
            )
        page = crud.get_all(test_session, skip=2, limit=2)
        assert len(page) == 2

    def test_update_record(self, test_session: Session, sample_user: User) -> None:
        """Test updating an existing record."""
        crud = BaseCRUD(User)
        updated = crud.update(
            test_session,
            sample_user.id,
            full_name="Updated Name",
        )
        assert updated is not None
        assert updated.full_name == "Updated Name"

    def test_update_nonexistent_record(self, test_session: Session) -> None:
        """Test that updating a nonexistent record returns None."""
        crud = BaseCRUD(User)
        result = crud.update(test_session, "fake-id", full_name="Nope")
        assert result is None

    def test_delete_record(self, test_session: Session, sample_user: User) -> None:
        """Test deleting an existing record."""
        crud = BaseCRUD(User)
        user_id = sample_user.id
        deleted = crud.delete(test_session, user_id)
        assert deleted is True
        assert crud.get_by_id(test_session, user_id) is None

    def test_delete_nonexistent_record(self, test_session: Session) -> None:
        """Test that deleting a nonexistent record returns False."""
        crud = BaseCRUD(User)
        deleted = crud.delete(test_session, "fake-id")
        assert deleted is False
