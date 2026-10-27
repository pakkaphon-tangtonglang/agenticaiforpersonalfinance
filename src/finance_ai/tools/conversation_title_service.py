"""Conversation title and lifecycle service.

Lets the agent name a conversation after its first exchange, and lets
users rename or delete their own conversations from the sidebar.
"""

import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy.orm import Session

from finance_ai.core.logging import get_logger
from finance_ai.database.crud.conversation_crud import ConversationCRUD
from finance_ai.database.models.conversation import Conversation
from finance_ai.tools.conversation_constants import (
    CONVERSATION_TITLE_MAX_LENGTH,
    CONVERSATION_TITLE_PROMPT,
    DEFAULT_CONVERSATION_TITLE,
)
from finance_ai.tools.conversation_service import (
    load_conversation_messages,
    update_conversation_title,
)

logger = get_logger(__name__)

_conversation_crud = ConversationCRUD()

# Characters stripped from the edges of an LLM-generated title
_TITLE_EDGE_CHARACTERS = " \t\"'`“”‘’「」#*_-:."

# Maximum characters of the assistant reply sent as title context
_ASSISTANT_CONTEXT_LENGTH = 300


def clean_generated_title(raw_title: str) -> str:
    """Normalize an LLM-generated title into a short sidebar label.

    Args:
        raw_title: Raw text returned by the LLM.

    Returns:
        First non-empty line without quotes/markdown, truncated to
        CONVERSATION_TITLE_MAX_LENGTH. Empty string if nothing is left.

    Example:
        >>> clean_generated_title('## "วางแผนภาษี"')
        'วางแผนภาษี'
    """
    lines = [line for line in str(raw_title).splitlines() if line.strip()]
    if not lines:
        return ""
    without_bold = re.sub(r"\*\*|__", "", lines[0])
    return without_bold.strip(_TITLE_EDGE_CHARACTERS)[:CONVERSATION_TITLE_MAX_LENGTH].strip()


def _fallback_title(first_user_message: str) -> str:
    """Build a title from the first user message when the LLM can't.

    Args:
        first_user_message: The first message the user sent.

    Returns:
        Cleaned, truncated message, or the default title if empty.
    """
    return clean_generated_title(first_user_message) or DEFAULT_CONVERSATION_TITLE


def generate_conversation_title(
    chat_model: Any,
    user_message: str,
    assistant_message: str,
) -> str:
    """Ask the LLM for a short conversation title.

    Args:
        chat_model: LangChain chat model with an ``invoke`` method.
        user_message: First user message in the conversation.
        assistant_message: First assistant reply (truncated as context).

    Returns:
        Cleaned title, falling back to the user message on error or empty reply.

    Example:
        >>> generate_conversation_title(model, "คำนวณภาษี", "ภาษีของคุณ...")
        'คำนวณภาษีเงินเดือน'
    """
    messages = [
        SystemMessage(content=CONVERSATION_TITLE_PROMPT),
        HumanMessage(
            content=f"ผู้ใช้: {user_message}\n"
            f"ผู้ช่วย: {assistant_message[:_ASSISTANT_CONTEXT_LENGTH]}"
        ),
    ]
    try:
        title = clean_generated_title(chat_model.invoke(messages).text)
    except Exception:  # noqa: BLE001 — any LLM failure must not break the chat
        logger.warning("Conversation title generation failed", exc_info=True)
        title = ""
    return title or _fallback_title(user_message)


def _get_owned_conversation(
    session: Session,
    conversation_id: str,
    user_id: str,
) -> Optional[Conversation]:
    """Return the conversation only if it belongs to the user.

    Args:
        session: Database session.
        conversation_id: Conversation UUID.
        user_id: User UUID that must own the conversation.

    Returns:
        The Conversation, or None if missing or owned by someone else.
    """
    conversation = _conversation_crud.get_by_id(session, conversation_id)
    if conversation is None or conversation.user_id != user_id:
        return None
    return conversation


def _first_message(messages: list[dict[str, Any]], role: str) -> str:
    """Return the content of the first message with the given role.

    Args:
        messages: Message dicts from load_conversation_messages.
        role: 'user' or 'assistant'.

    Returns:
        Message content, or empty string if none.
    """
    return next((str(msg["content"]) for msg in messages if msg["role"] == role), "")


def auto_title_conversation(
    session: Session,
    conversation_id: str,
    user_id: str,
    chat_model: Any,
) -> Optional[str]:
    """Name a default-titled conversation from its first exchange.

    Idempotent: a conversation that already has a title (auto or manual)
    is returned unchanged without calling the LLM.

    Args:
        session: Database session.
        conversation_id: Conversation UUID.
        user_id: User UUID that must own the conversation.
        chat_model: LangChain chat model used for the title.

    Returns:
        The conversation's title after the call, or None if not found/owned.

    Example:
        >>> auto_title_conversation(session, conv_id, user_id, model)
        'คำนวณภาษีเงินเดือน'
    """
    conversation = _get_owned_conversation(session, conversation_id, user_id)
    if conversation is None:
        return None
    if conversation.title != DEFAULT_CONVERSATION_TITLE:
        return conversation.title
    messages = load_conversation_messages(session, conversation_id)
    user_message = _first_message(messages, "user")
    if not user_message:
        return conversation.title
    title = generate_conversation_title(
        chat_model, user_message, _first_message(messages, "assistant")
    )
    update_conversation_title(session, conversation_id, title)
    return title


def rename_conversation(
    session: Session,
    conversation_id: str,
    user_id: str,
    title: str,
) -> Optional[Conversation]:
    """Rename a conversation owned by the user.

    Args:
        session: Database session.
        conversation_id: Conversation UUID.
        user_id: User UUID that must own the conversation.
        title: New title (stripped and truncated).

    Returns:
        Updated Conversation, or None if not found/owned.

    Example:
        >>> rename_conversation(session, conv_id, user_id, "ภาษี 2026")
    """
    if _get_owned_conversation(session, conversation_id, user_id) is None:
        return None
    return update_conversation_title(session, conversation_id, title.strip())


def delete_user_conversation(
    session: Session,
    conversation_id: str,
    user_id: str,
) -> bool:
    """Delete a conversation (and its messages) owned by the user.

    Args:
        session: Database session.
        conversation_id: Conversation UUID.
        user_id: User UUID that must own the conversation.

    Returns:
        True if deleted, False if not found/owned.

    Example:
        >>> delete_user_conversation(session, conv_id, user_id)
        True
    """
    if _get_owned_conversation(session, conversation_id, user_id) is None:
        return False
    _conversation_crud.delete(session, conversation_id)
    session.commit()
    return True
