"""Conversation history persistence service.

Provides functions for saving, loading, and managing conversations.
Service layer sits between the Streamlit UI and the CRUD layer.
"""

from typing import Any, Optional

from sqlalchemy.orm import Session

from finance_ai.database.crud.conversation_crud import ConversationCRUD
from finance_ai.database.crud.conversation_message_crud import ConversationMessageCRUD
from finance_ai.database.models.conversation import Conversation
from finance_ai.database.models.conversation_message import ConversationMessage
from finance_ai.tools.conversation_constants import (
    CONVERSATION_TITLE_MAX_LENGTH,
    DEFAULT_CONVERSATION_TITLE,
    MAX_CONVERSATIONS_PER_USER,
    MAX_HISTORY_MESSAGES,
)

_conversation_crud = ConversationCRUD()
_message_crud = ConversationMessageCRUD()


def create_conversation(
    session: Session,
    user_id: str,
    title: str = DEFAULT_CONVERSATION_TITLE,
) -> Conversation:
    """Create a new conversation for a user.

    Args:
        session: Database session.
        user_id: UUID of the user.
        title: Conversation title.

    Returns:
        Created Conversation instance.

    Example:
        >>> conv = create_conversation(session, "user-123")
    """
    return _conversation_crud.create(session, user_id=user_id, title=title)


def save_user_message(
    session: Session,
    conversation_id: str,
    content: str,
) -> ConversationMessage:
    """Save a user message to a conversation.

    Args:
        session: Database session.
        conversation_id: UUID of the conversation.
        content: Message text.

    Returns:
        Created ConversationMessage instance.

    Example:
        >>> msg = save_user_message(session, conv_id, "คำนวณภาษี")
    """
    return _message_crud.create(
        session,
        conversation_id=conversation_id,
        role="user",
        content=content,
    )


def save_assistant_message(
    session: Session,
    conversation_id: str,
    content: str,
    intent: Optional[str] = None,
) -> ConversationMessage:
    """Save an assistant message to a conversation.

    Args:
        session: Database session.
        conversation_id: UUID of the conversation.
        content: Message text.
        intent: Optional intent label (e.g., "tax", "expense").

    Returns:
        Created ConversationMessage instance.

    Example:
        >>> msg = save_assistant_message(session, conv_id, "ผล...", "tax")
    """
    return _message_crud.create(
        session,
        conversation_id=conversation_id,
        role="assistant",
        content=content,
        intent=intent,
    )


def get_recent_history_as_tuples(
    session: Session,
    conversation_id: str,
    limit: int = MAX_HISTORY_MESSAGES,
) -> list[tuple[str, str]]:
    """Load recent messages as (role, content) tuples for LangGraph.

    LangGraph's add_messages annotator converts these tuples
    to HumanMessage/AIMessage automatically.

    Args:
        session: Database session.
        conversation_id: UUID of the conversation.
        limit: Maximum number of messages to return.

    Returns:
        List of (role, content) tuples in chronological order.

    Example:
        >>> history = get_recent_history_as_tuples(session, conv_id)
        >>> # [("user", "คำนวณภาษี"), ("assistant", "ผล...")]
    """
    messages = _message_crud.get_recent_by_conversation(session, conversation_id, limit=limit)
    return [(msg.role, msg.content) for msg in messages if msg.content.strip()]


def load_conversation_messages(
    session: Session,
    conversation_id: str,
) -> list[dict[str, Any]]:
    """Load all messages for Streamlit display.

    Returns format matching st.session_state.messages structure.

    Args:
        session: Database session.
        conversation_id: UUID of the conversation.

    Returns:
        List of message dicts with role, content, and optional intent.

    Example:
        >>> msgs = load_conversation_messages(session, conv_id)
    """
    messages = _message_crud.get_by_conversation(session, conversation_id)
    return [_message_to_dict(msg) for msg in messages]


def _message_to_dict(message: ConversationMessage) -> dict[str, Any]:
    """Convert a ConversationMessage to a Streamlit-compatible dict.

    Args:
        message: ConversationMessage instance.

    Returns:
        Dict with role, content, and optional intent.
    """
    result: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
    }
    if message.intent:
        result["intent"] = message.intent
    return result


def get_or_create_active_conversation(
    session: Session,
    user_id: str,
) -> Conversation:
    """Get the latest active conversation or create a new one.

    Args:
        session: Database session.
        user_id: UUID of the user.

    Returns:
        Active Conversation instance.

    Example:
        >>> conv = get_or_create_active_conversation(session, uid)
    """
    existing = _conversation_crud.get_latest_by_user(session, user_id)
    if existing is not None:
        return existing
    return create_conversation(session, user_id)


def update_conversation_title(
    session: Session,
    conversation_id: str,
    title: str,
) -> Optional[Conversation]:
    """Update the title of a conversation.

    Truncates title to CONVERSATION_TITLE_MAX_LENGTH characters.

    Args:
        session: Database session.
        conversation_id: UUID of the conversation.
        title: New title text.

    Returns:
        Updated Conversation or None if not found.

    Example:
        >>> update_conversation_title(session, conv_id, "ภาษี 2026")
    """
    truncated = title[:CONVERSATION_TITLE_MAX_LENGTH]
    return _conversation_crud.update(session, conversation_id, title=truncated)


def list_user_conversations(
    session: Session,
    user_id: str,
    limit: int = MAX_CONVERSATIONS_PER_USER,
) -> list[Conversation]:
    """List active conversations for sidebar display.

    Args:
        session: Database session.
        user_id: UUID of the user.
        limit: Maximum conversations to return.

    Returns:
        List of active Conversations, most recent first.

    Example:
        >>> convs = list_user_conversations(session, uid)
    """
    conversations = _conversation_crud.get_active_by_user(session, user_id)
    return conversations[:limit]
