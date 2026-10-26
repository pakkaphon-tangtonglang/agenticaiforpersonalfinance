"""Conversation history persistence service.

Provides functions for saving, loading, and managing conversations.
Service layer sits between the API layer and the CRUD layer.
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
    kind: str = "chat",
    guru_id: str | None = None,
) -> Conversation:
    """Create a new conversation for a user.

    Args:
        session: Database session.
        user_id: UUID of the user.
        title: Conversation title.
        kind: "chat" for the main tab or "guru" for the Guru tab.
        guru_id: Persona id for guru conversations.

    Returns:
        Created Conversation instance.

    Example:
        >>> conv = create_conversation(session, "user-123")
    """
    from finance_ai.tools.user_provisioning_service import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        ensure_user_exists,
    )

    ensure_user_exists(session, user_id)
    conversation = _conversation_crud.create(
        session, user_id=user_id, title=title, kind=kind, guru_id=guru_id
    )
    session.commit()
    session.refresh(conversation)
    return conversation


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
    message = _message_crud.create(
        session,
        conversation_id=conversation_id,
        role="user",
        content=content,
    )
    session.commit()
    session.refresh(message)
    return message


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
    message = _message_crud.create(
        session,
        conversation_id=conversation_id,
        role="assistant",
        content=content,
        intent=intent,
    )
    session.commit()
    session.refresh(message)
    return message


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
    """Load all messages for a conversation.

    Returns format matching the API message dict structure.

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
    """Convert a ConversationMessage to an API-friendly dict.

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
    conversation = _conversation_crud.update(session, conversation_id, title=truncated)
    if conversation is not None:
        session.commit()
        session.refresh(conversation)
    return conversation


def list_user_conversations(
    session: Session,
    user_id: str,
    limit: int = MAX_CONVERSATIONS_PER_USER,
    kind: str = "chat",
) -> list[Conversation]:
    """List active conversations of one kind for sidebar display.

    Args:
        session: Database session.
        user_id: UUID of the user.
        limit: Maximum conversations to return.
        kind: "chat" for the main tab or "guru" for the Guru tab.

    Returns:
        List of active Conversations, most recent first.

    Example:
        >>> convs = list_user_conversations(session, uid)
    """
    conversations = _conversation_crud.get_active_by_user(session, user_id, kind=kind)
    return conversations[:limit]


def get_owned_guru_conversation(
    session: Session,
    conversation_id: str,
    user_id: str,
    guru_id: str,
) -> Optional[Conversation]:
    """Return a guru conversation only if the user owns it and it is for this guru.

    Args:
        session: Database session.
        conversation_id: Conversation UUID from the client.
        user_id: User UUID that must own the conversation.
        guru_id: Persona id (or "panel") the conversation must belong to.

    Returns:
        The Conversation, or None when missing, not owned, not a guru chat,
        or created for a different guru.

    Example:
        >>> get_owned_guru_conversation(session, conv_id, uid, "dalio")
    """
    conversation = _conversation_crud.get_by_id(session, conversation_id)
    if conversation is None or conversation.user_id != user_id:
        return None
    if conversation.kind != "guru" or conversation.guru_id != guru_id:
        return None
    return conversation


ATTACHMENT_INTENT = "upload"


def save_attachment_note(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    session: Session,
    conversation_id: str,
    user_id: str,
    file_name: str,
    summary: str,
) -> bool:
    """Record an in-chat file upload as a user/assistant message pair.

    Keeps the chat history complete ("📎 slip.jpg" -> "บันทึกแล้ว 2 รายการ"),
    so reopened chats show the upload and later answers know what was saved.

    Args:
        session: Database session.
        conversation_id: Main-chat conversation UUID.
        user_id: User UUID that must own the conversation.
        file_name: Name of the attached file.
        summary: What happened (saved items, cancelled, failed).

    Returns:
        True when saved; False if the conversation is missing, not owned,
        or not a main chat.

    Example:
        >>> save_attachment_note(session, conv_id, uid, "slip.jpg", "บันทึกแล้ว 1 รายการ")
        True
    """
    conversation = _conversation_crud.get_by_id(session, conversation_id)
    if conversation is None or conversation.user_id != user_id or conversation.kind != "chat":
        return False
    save_user_message(session, conversation_id, f"📎 {file_name}")
    save_assistant_message(session, conversation_id, summary, ATTACHMENT_INTENT)
    return True
