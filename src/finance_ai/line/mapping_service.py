"""Service mapping LINE userIds to application users and conversations."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.database.models.line_user_mapping import LineUserMapping
from finance_ai.database.models.user import User
from finance_ai.tools.conversation_service import create_conversation

_LINE_EMAIL_DOMAIN = "line.users.finance-ai.local"


def get_or_create_line_mapping(session: Session, line_user_id: str) -> LineUserMapping:
    """Return the mapping for a LINE user, creating it on first message.

    Args:
        session: Database session used for the lookup and creation.
        line_user_id: LINE platform userId (e.g. 'U4af4980629...').

    Returns:
        LineUserMapping linking the LINE account to an app user and a
        conversation (so chat history persists across LINE messages).

    Example:
        >>> mapping = get_or_create_line_mapping(session, line_user_id)
        >>> mapping.user_id
        '...'
    """
    existing = session.scalar(
        select(LineUserMapping).where(LineUserMapping.line_user_id == line_user_id)
    )
    if existing is not None:
        return existing
    return _create_line_mapping(session, line_user_id)


def _create_line_mapping(session: Session, line_user_id: str) -> LineUserMapping:
    """Create the backing user, default conversation, and mapping row."""
    user = User(
        email=f"line-{line_user_id}@{_LINE_EMAIL_DOMAIN}",
        hashed_password="line-login-not-supported",
        full_name="ผู้ใช้ LINE",
    )
    session.add(user)
    session.flush()
    conversation = create_conversation(session, user.id)
    mapping = LineUserMapping(
        line_user_id=line_user_id,
        user_id=user.id,
        conversation_id=conversation.id,
    )
    session.add(mapping)
    session.commit()
    session.refresh(mapping)
    return mapping
