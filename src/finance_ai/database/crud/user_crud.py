"""CRUD operations for the User model."""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.database.crud.base_crud import BaseCRUD
from finance_ai.database.models.user import User


class UserCRUD(BaseCRUD[User]):
    """
    CRUD operations specific to User model.

    Example:
        >>> user_crud = UserCRUD()
        >>> user = user_crud.get_by_email(session, "test@example.com")
    """

    def __init__(self) -> None:
        """Initialize UserCRUD with User model."""
        super().__init__(User)

    def get_by_email(self, session: Session, email: str) -> Optional[User]:
        """
        Get a user by email address.

        Args:
            session: Database session.
            email: User email address.

        Returns:
            User instance or None if not found.
        """
        statement = select(User).where(User.email == email)
        return session.execute(statement).scalar_one_or_none()

    def get_by_tax_id(self, session: Session, tax_id: str) -> Optional[User]:
        """
        Get a user by Thai tax ID.

        Args:
            session: Database session.
            tax_id: 13-digit Thai tax ID.

        Returns:
            User instance or None if not found.
        """
        statement = select(User).where(User.tax_id == tax_id)
        return session.execute(statement).scalar_one_or_none()

    def get_active_users(self, session: Session, skip: int = 0, limit: int = 100) -> list[User]:
        """
        Get all active (non-deactivated) users.

        Args:
            session: Database session.
            skip: Number of records to skip.
            limit: Maximum number of records to return.

        Returns:
            List of active User instances.
        """
        statement = select(User).where(User.is_active.is_(True)).offset(skip).limit(limit)
        return list(session.execute(statement).scalars().all())

    def get_or_create_demo_user(self, session: Session, user_id: str) -> User:
        """Get an existing user by ID, or create a demo user if not found.

        Used by the FastAPI backend to ensure a valid user record exists
        before expense/investment agents try to write transactions.

        Args:
            session: Database session.
            user_id: UUID string for the user.

        Returns:
            Existing or newly created User instance.

        Example:
            >>> user = user_crud.get_or_create_demo_user(session, "abc123")
            >>> user.email
            'demo-abc123@finance-ai.local'
        """
        existing = self.get_by_id(session, user_id)
        if existing is not None:
            return existing
        return self.create(
            session,
            id=user_id,
            email=f"demo-{user_id[:8]}@finance-ai.local",
            hashed_password="demo-not-for-production",
            full_name="Demo User",
        )

    def deactivate(self, session: Session, user_id: str) -> bool:
        """
        Soft-delete a user by setting is_active to False.

        Args:
            session: Database session.
            user_id: UUID of the user to deactivate.

        Returns:
            True if deactivated, False if user not found.
        """
        user = self.get_by_id(session, user_id)
        if user is None:
            return False
        user.is_active = False
        session.commit()
        return True
