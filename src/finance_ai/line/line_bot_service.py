"""Bridge between LINE messages and the multi-agent orchestration graph."""

from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from finance_ai.agents.router_agent import orchestrate_query
from finance_ai.database.models.line_user_mapping import LineUserMapping
from finance_ai.line.link_command import (
    link_line_user,
    parse_link_command,
    parse_unlink_command,
    unlink_line_user,
)
from finance_ai.line.mapping_service import get_or_create_line_mapping
from finance_ai.line.messaging_client import send_line_push
from finance_ai.tools.conversation_service import (
    get_recent_history_as_tuples,
    save_assistant_message,
    save_user_message,
)


def process_line_message(
    session_factory: sessionmaker[Session],
    chat_model_provider: Callable[[], Any] | None,
    line_user_id: str,
    text: str,
) -> str:
    """Run one LINE user message through the agent graph and return the answer.

    Resolves (or lazily creates) the LINE user mapping, replays the mapped
    conversation history for routing context, and persists both the user
    and assistant messages.

    Args:
        session_factory: Factory creating database sessions.
        chat_model_provider: Callable returning the chat model (lazy so the
            model is only built when an agent actually runs).
        line_user_id: LINE platform userId of the sender.
        text: The user's Thai text message.

    Returns:
        str: The agent's response text.

    Example:
        >>> reply = process_line_message(factory, get_chat_model, uid, "สวัสดี")
        >>> reply
        'สวัสดีครับ...'
    """
    with session_factory() as session:
        mapping = get_or_create_line_mapping(session, line_user_id)
        link_reply = _maybe_handle_link_command(session, mapping, text)
        if link_reply is not None:
            return link_reply
        history = get_recent_history_as_tuples(session, mapping.conversation_id)
        save_user_message(session, mapping.conversation_id, text)
        chat_model = chat_model_provider() if chat_model_provider is not None else None
        result = orchestrate_query(
            query=text,
            chat_model=chat_model,
            user_id=mapping.user_id,
            db_session_factory=session_factory,
            chat_history=history,
        )
        save_assistant_message(
            session, mapping.conversation_id, result["response"], result["intent"]
        )
        return str(result["response"])


def _maybe_handle_link_command(session: Session, mapping: LineUserMapping, text: str) -> str | None:
    """Handle link/unlink commands, or return None to run the agent."""
    if parse_unlink_command(text):
        success, reply = unlink_line_user(session, mapping.line_user_id)
    else:
        target_user_id = parse_link_command(text)
        if target_user_id is None:
            return None
        success, reply = link_line_user(session, mapping.line_user_id, target_user_id)
    intent = "link_success" if success else "link_failed"
    save_assistant_message(session, mapping.conversation_id, reply, intent)
    return reply


def handle_line_event(
    line_user_id: str,
    text: str,
    session_factory: sessionmaker[Session],
    chat_model_provider: Callable[[], Any] | None,
    access_token: str,
) -> None:
    """Process one LINE text event end-to-end and push the answer back.

    Designed to run as a FastAPI background task so the webhook returns
    200 within LINE's ~1s window while the agent (10-30s) runs after.

    Args:
        line_user_id: LINE platform userId of the sender.
        text: The user's Thai text message.
        session_factory: Factory creating database sessions.
        chat_model_provider: Callable returning the chat model.
        access_token: LINE channel access token for the Push API.

    Example:
        >>> handle_line_event(uid, "ภาษีของฉัน", factory, get_chat_model, token)
    """
    try:
        reply = process_line_message(session_factory, chat_model_provider, line_user_id, text)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        reply = f"ขออภัยครับ เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้ง ({exc})"
    send_line_push(access_token, line_user_id, reply)
