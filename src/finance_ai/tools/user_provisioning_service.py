"""User provisioning service.

Auto-creates placeholder users for clients that self-generate their IDs
(web UI uses a per-browser UUID). Postgres enforces foreign keys (SQLite
did not), so any write for an unknown user_id fails — this service
guarantees the user row exists first. The web API calls it once per
request from an app-wide dependency (``main.provision_request_user``).
"""

import re
from typing import Any, cast

from sqlalchemy import insert as generic_insert
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from finance_ai.core.logging import get_logger
from finance_ai.database.crud.user_crud import UserCRUD
from finance_ai.database.models.user import User

WEB_EMAIL_DOMAIN = "users.finance-ai.local"
# users.id is VARCHAR(36); web ids are UUIDs, demo/test ids are short slugs.
_USER_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,36}")

logger = get_logger(__name__)


def is_valid_user_id(user_id: str) -> bool:
    """Check a client-supplied user id fits the users table safely.

    Args:
        user_id: Identifier from the request.

    Returns:
        True for 1-36 letters, digits, '-' or '_'.

    Example:
        >>> is_valid_user_id("11111111-2222-3333-4444-555555555555")
        True
    """
    return bool(_USER_ID_PATTERN.fullmatch(user_id))


def _placeholder_user(user_id: str) -> dict[str, Any]:
    """Column values for a web user created on first contact."""
    return {
        "id": user_id,
        "email": f"{user_id}@{WEB_EMAIL_DOMAIN}",
        "hashed_password": "web-client-not-supported",
        "full_name": "ผู้ใช้เว็บไซต์",
    }


def insert_user_if_missing(session: Session, user_id: str) -> bool:
    """Insert the placeholder user, ignoring a row created concurrently.

    Uses INSERT ... ON CONFLICT DO NOTHING on Postgres and SQLite, so two
    first requests for the same new user cannot fail each other.

    Args:
        session: Database session (caller owns the transaction).
        user_id: Client-supplied user identifier.

    Returns:
        True when this call created the row.

    Example:
        >>> insert_user_if_missing(session, "abc-123")
        True
    """
    dialect = session.get_bind().dialect.name
    values = _placeholder_user(user_id)
    if dialect in ("postgresql", "sqlite"):
        insert = postgresql_insert if dialect == "postgresql" else sqlite_insert
        statement: Any = insert(User).values(**values).on_conflict_do_nothing()
    else:
        statement = generic_insert(User).values(**values)
    result = cast(CursorResult[Any], session.execute(statement))
    return bool(result.rowcount)


def ensure_user_exists(session: Session, user_id: str) -> None:
    """Create a placeholder user when the given user_id has no row yet.

    Safe to call before every write: existing users (including
    LINE-mapped ones) are returned untouched. The caller owns the
    transaction — this function does not commit.

    Args:
        session: Database session.
        user_id: Client-supplied user identifier (UUID string).

    Example:
        >>> ensure_user_exists(session, "abc-123")
    """
    if UserCRUD().get_by_id(session, user_id) is not None:
        return
    if insert_user_if_missing(session, user_id):
        logger.info("Auto-created user %s (web client)", user_id)
