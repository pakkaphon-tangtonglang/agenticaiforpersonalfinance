"""Deliver a fired watchlist alert outside the web app (LINE push).

The web app shows alerts itself (in-app list + browser notification). Users
who linked LINE (`เชื่อมต่อ <user-id>`) also get the alert as a LINE push.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.core.config import get_settings
from finance_ai.core.logging import get_logger
from finance_ai.database.models.line_user_mapping import LineUserMapping
from finance_ai.line.markdown_to_text import markdown_to_line_text
from finance_ai.line.messaging_client import send_line_push

logger = get_logger(__name__)

ALERT_TITLE = "🔔 ข่าวประจำวัน {symbol}"
LINE_TEXT_LIMIT = 5000  # LINE rejects text messages longer than this
_TRUNCATION_MARK = "…"


def _fit_line_limit(text: str) -> str:
    """Cut text to LINE's message limit, marking the cut."""
    if len(text) <= LINE_TEXT_LIMIT:
        return text
    return text[: LINE_TEXT_LIMIT - len(_TRUNCATION_MARK)] + _TRUNCATION_MARK


def find_line_user_ids(session: Session, user_id: str) -> list[str]:
    """LINE userIds linked to an app user (usually zero or one).

    Example:
        >>> find_line_user_ids(session, uid)
        ['U4af4980629...']
    """
    stmt = select(LineUserMapping.line_user_id).where(LineUserMapping.user_id == user_id)
    return list(session.execute(stmt).scalars().all())


def push_alert_to_line(session: Session, user_id: str, symbol: str, content: str) -> int:
    """Push a fired alert to every LINE account linked to the user.

    Failures are logged, never raised: the in-app notification is already saved.

    Args:
        session: Database session.
        user_id: App user who owns the alert.
        symbol: Asset symbol (used in the title).
        content: Alert summary (Markdown).

    Returns:
        Number of LINE messages sent.

    Example:
        >>> push_alert_to_line(session, uid, "PTT.BK", "ราคา 35.50 บาท")
        1
    """
    access_token = get_settings().line_channel_access_token
    line_user_ids = find_line_user_ids(session, user_id) if access_token else []
    title = ALERT_TITLE.format(symbol=symbol)
    text = _fit_line_limit(f"{title}\n\n{markdown_to_line_text(content)}")
    sent = 0
    for line_user_id in line_user_ids:
        try:
            send_line_push(str(access_token), line_user_id, text, quick_reply_labels=())
            sent += 1
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            logger.warning("LINE alert push failed for %s: %s", symbol, exc)
    return sent
