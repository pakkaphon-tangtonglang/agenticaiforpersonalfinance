"""LINE commands linking a LINE account to an existing web-app user.

The web frontend identifies users by a random UUID stored in the
browser (localStorage 'pfai_uid'). Sending `เชื่อมต่อ <user-id>` in
LINE re-points this LINE account's mapping at that user so both
surfaces share the same finances; `ยกเลิกเชื่อมต่อ` restores the
auto-created LINE-only user.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.database.models.line_user_mapping import LineUserMapping
from finance_ai.database.models.user import User
from finance_ai.line.mapping_service import LINE_EMAIL_DOMAIN

LINK_COMMAND_PREFIXES = ("เชื่อมต่อ", "link")
UNLINK_COMMANDS = ("ยกเลิกเชื่อมต่อ", "unlink")


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


def parse_unlink_command(text: str) -> bool:
    """Return True when the text is an unlink command.

    Args:
        text: Raw message text from the LINE user.

    Returns:
        bool: True for 'ยกเลิกเชื่อมต่อ' or 'unlink' (case-insensitive).

    Example:
        >>> parse_unlink_command("UNLINK")
        True
    """
    return text.strip().lower() in UNLINK_COMMANDS


def unlink_line_user(session: Session, line_user_id: str) -> tuple[bool, str]:
    """Restore a LINE mapping to its auto-created LINE-only user.

    Args:
        session: Database session.
        line_user_id: LINE platform userId of the sender.

    Returns:
        (success, thai_message) — success is True when the mapping again
        points at the LINE-only account; the message is the bot's reply.

    Example:
        >>> unlink_line_user(session, "U4af...")
        (True, 'ยกเลิกการเชื่อมต่อเรียบร้อยครับ ...')
    """
    mapping = session.scalar(
        select(LineUserMapping).where(LineUserMapping.line_user_id == line_user_id)
    )
    if mapping is None:
        return False, "ยังไม่มีการเชื่อมต่อบัญชีอยู่ครับ"
    mapping.user_id = _get_or_create_line_only_user(session, line_user_id).id
    session.commit()
    return True, "ยกเลิกการเชื่อมต่อเรียบร้อยครับ กลับไปใช้บัญชีเดิมของแชท LINE นี้แล้ว"


def unlink_web_user(session: Session, user_id: str) -> list[str]:
    """Disconnect every LINE chat linked to a web-app user.

    Args:
        session: Database session.
        user_id: Web-app user id whose LINE links should be removed.

    Returns:
        The line_user_ids that were unlinked (possibly empty).

    Example:
        >>> unlink_web_user(session, user_id)
        ['U4af...']
    """
    mappings = session.scalars(
        select(LineUserMapping).where(LineUserMapping.user_id == user_id)
    ).all()
    unlinked = []
    for mapping in mappings:
        line_only_user = _get_or_create_line_only_user(session, mapping.line_user_id)
        mapping.user_id = line_only_user.id
        unlinked.append(mapping.line_user_id)
    if unlinked:
        session.commit()
    return unlinked


def _get_or_create_line_only_user(session: Session, line_user_id: str) -> User:
    """Return the auto-created user for a LINE account, recreating if needed."""
    line_email = f"line-{line_user_id}@{LINE_EMAIL_DOMAIN}"
    user = session.scalar(select(User).where(User.email == line_email))
    if user is None:
        user = User(
            email=line_email,
            hashed_password="line-login-not-supported",
            full_name="ผู้ใช้ LINE",
        )
        session.add(user)
        session.flush()
    return user
