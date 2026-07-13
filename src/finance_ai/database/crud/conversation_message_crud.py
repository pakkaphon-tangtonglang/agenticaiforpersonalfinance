"""CRUD operations for the ConversationMessage model."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.database.crud.base_crud import BaseCRUD
from finance_ai.database.models.conversation_message import ConversationMessage


class ConversationMessageCRUD(BaseCRUD[ConversationMessage]):
    """CRUD operations specific to ConversationMessage model.

    Example:
        >>> crud = ConversationMessageCRUD()
        >>> messages = crud.get_by_conversation(session, conv_id)
    """

    def __init__(self) -> None:
        """Initialize with ConversationMessage model."""
        super().__init__(ConversationMessage)

    def get_by_conversation(
        self,
        session: Session,
        conversation_id: str,
        limit: int = 100,
    ) -> list[ConversationMessage]:
        """Get messages for a conversation, ordered by created_at.

        Args:
            session: Database session.
            conversation_id: UUID of the conversation.
            limit: Maximum number of messages to return.

        Returns:
            List of ConversationMessage ordered ascending.
        """
        statement = (
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.created_at.asc())
            .limit(limit)
        )
        return list(session.execute(statement).scalars().all())

    def get_recent_by_conversation(
        self,
        session: Session,
        conversation_id: str,
        limit: int = 10,
    ) -> list[ConversationMessage]:
        """Get the N most recent messages for LLM context.

        Returns messages in chronological order (oldest first)
        by using a subquery to select the latest N, then sorting.

        Args:
            session: Database session.
            conversation_id: UUID of the conversation.
            limit: Maximum number of recent messages.

        Returns:
            List of recent ConversationMessage in ascending order.
        """
        subquery = (
            select(ConversationMessage.id)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.created_at.desc())
            .limit(limit)
        )
        statement = (
            select(ConversationMessage)
            .where(ConversationMessage.id.in_(subquery))
            .order_by(ConversationMessage.created_at.asc())
        )
        return list(session.execute(statement).scalars().all())
