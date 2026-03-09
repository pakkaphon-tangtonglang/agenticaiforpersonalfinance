"""CRUD operations for the Conversation model."""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.database.crud.base_crud import BaseCRUD
from finance_ai.database.models.conversation import Conversation


class ConversationCRUD(BaseCRUD[Conversation]):
    """CRUD operations specific to Conversation model.

    Example:
        >>> crud = ConversationCRUD()
        >>> conversations = crud.get_by_user(session, user_id)
    """

    def __init__(self) -> None:
        """Initialize ConversationCRUD with Conversation model."""
        super().__init__(Conversation)

    def get_by_user(self, session: Session, user_id: str, limit: int = 20) -> list[Conversation]:
        """Get conversations for a user, ordered by most recent first.

        Args:
            session: Database session.
            user_id: UUID of the user.
            limit: Maximum number of conversations to return.

        Returns:
            List of Conversation instances.
        """
        statement = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
        )
        return list(session.execute(statement).scalars().all())

    def get_active_by_user(self, session: Session, user_id: str) -> list[Conversation]:
        """Get active conversations for a user.

        Args:
            session: Database session.
            user_id: UUID of the user.

        Returns:
            List of active Conversation instances.
        """
        statement = (
            select(Conversation)
            .where(
                Conversation.user_id == user_id,
                Conversation.is_active.is_(True),
            )
            .order_by(Conversation.updated_at.desc())
        )
        return list(session.execute(statement).scalars().all())

    def get_latest_by_user(self, session: Session, user_id: str) -> Optional[Conversation]:
        """Get the most recently updated active conversation.

        Args:
            session: Database session.
            user_id: UUID of the user.

        Returns:
            Latest active Conversation or None.
        """
        statement = (
            select(Conversation)
            .where(
                Conversation.user_id == user_id,
                Conversation.is_active.is_(True),
            )
            .order_by(Conversation.updated_at.desc())
            .limit(1)
        )
        return session.execute(statement).scalar_one_or_none()
