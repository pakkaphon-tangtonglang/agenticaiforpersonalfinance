"""LINE command linking a LINE account to an existing web-app user.

The web frontend identifies users by a random UUID stored in the
browser (localStorage 'pfai_uid'). Sending `เชื่อมต่อ <user-id>` in
LINE re-points this LINE account's mapping at that user so both
surfaces share the same finances.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.database.models.line_user_mapping import LineUserMapping
from finance_ai.database.models.user import User

LINK_COMMAND_PREFIXES = ("เชื่อมต่อ", "link")


def parse_link_command(text: str) -> str | None:
    """Extract the target user id from a link command, if the text is one.

    Args:
        text: Raw message text from the LINE user.

    Returns:
        The target user id when the text matches 'เชื่อมต่อ <id>' or
        'link <id>' (case-insensitive), otherwise None.

    Example:
        >>> parse_link_command("เชื่อมต่อ 00000000-de20-...")
        '00000000-de20-...'
    """
    tokens = text.strip().split()
    if len(tokens) != 2 or tokens[0].lower() not in LINK_COMMAND_PREFIXES:
        return None
    return tokens[1]


def link_line_user(session: Session, line_user_id: str, target_user_id: str) -> tuple[bool, str]:
    """Re-point a LINE mapping at an existing web-app user.

    Args:
        session: Database session.
        line_user_id: LINE platform userId of the sender.
        target_user_id: Existing application user id to link to.

    Returns:
        (success, thai_message) — success is True when the mapping now
        points at the target user; the message is the bot's reply text.

    Example:
        >>> link_line_user(session, "U4af...", user_id)
        (True, 'เชื่อมต่อบัญชีเรียบร้อยแล้วครับ ...')
    """
    mapping = session.scalar(
        select(LineUserMapping).where(LineUserMapping.line_user_id == line_user_id)
    )
    target_user = session.get(User, target_user_id)
    if mapping is None or target_user is None:
        return False, "ไม่พบผู้ใช้รหัสนี้ครับ กรุณาคัดลอก User ID จากหน้าเว็บแล้วส่ง 'เชื่อมต่อ <User ID>' อีกครั้ง"
    mapping.user_id = target_user_id
    session.commit()
    display_name = target_user.full_name or target_user.email
    return True, f"เชื่อมต่อบัญชีเรียบร้อยแล้วครับ ({display_name}) ข้อมูลการเงินชุดนี้จะใช้ทั้งในแชท LINE และหน้าเว็บ"
