"""Conversation message database model for Personal Finance AI."""

from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finance_ai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from finance_ai.database.models.conversation import Conversation


class ConversationMessage(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Individual message within a conversation."""

    __tablename__ = "conversation_messages"

    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
