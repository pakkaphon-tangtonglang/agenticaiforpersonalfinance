"""LineUserMapping database model linking LINE accounts to app users."""

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finance_ai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from finance_ai.database.models.conversation import Conversation
    from finance_ai.database.models.user import User


class LineUserMapping(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Mapping between a LINE userId and an application user + conversation.

    Lets the router keep its chat-history context when the user chats
    inside the LINE app.
    """

    __tablename__ = "line_user_mappings"

    line_user_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False, index=True
    )
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id"), nullable=False
    )

    user: Mapped["User"] = relationship()
    conversation: Mapped["Conversation"] = relationship()
