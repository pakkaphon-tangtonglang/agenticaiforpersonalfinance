"""User provisioning service.

Auto-creates placeholder users for clients that self-generate their IDs
(web UI uses a per-browser UUID). Postgres enforces foreign keys (SQLite
did not), so any write for an unknown user_id fails — this service
guarantees the user row exists first.
"""

from sqlalchemy.orm import Session

from finance_ai.core.logging import get_logger
from finance_ai.database.crud.user_crud import UserCRUD
from finance_ai.database.models.user import User

WEB_EMAIL_DOMAIN = "users.finance-ai.local"

logger = get_logger(__name__)


def ensure_user_exists(session: Session, user_id: str) -> None:
    """Create a placeholder user when the given user_id has no row yet.

    Safe to call before every write: existing users (including
    LINE-mapped ones) are returned untouched. The caller owns the
    transaction — this function flushes but does not commit.

    Args:
        session: Database session.
        user_id: Client-supplied user identifier (UUID string).

    Example:
        >>> ensure_user_exists(session, "abc-123")
    """
    if UserCRUD().get_by_id(session, user_id) is not None:
        return
    user = User(
        id=user_id,
        email=f"{user_id}@{WEB_EMAIL_DOMAIN}",
        hashed_password="web-client-not-supported",
        full_name="ผู้ใช้เว็บไซต์",
    )
    session.add(user)
    session.flush()
    logger.info("Auto-created user %s (web client)", user_id)
