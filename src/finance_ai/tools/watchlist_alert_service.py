"""Daily news alerts for watchlist assets.

Each watched asset can have one daily time ("HH:MM", Bangkok time). At that
time the background scheduler fetches news + price for the asset and stores
a notification (see scheduler_service.execute_scheduled_fetch).

Alerts are AssetSchedule rows marked with WATCHLIST_ALERT_DESCRIPTION. They are
updated in place and deactivated instead of deleted, because notifications
keep a foreign key to their schedule.
"""

import re
from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from finance_ai.database.crud.schedule_crud import AssetScheduleCRUD
from finance_ai.database.models.asset_schedule import AssetSchedule
from finance_ai.tools.input_normalizer import twelve_hour_to_24
from finance_ai.tools.background_scheduler import (
    get_scheduler,
    register_schedule,
    unregister_schedule,
)

WATCHLIST_ALERT_DESCRIPTION = "watchlist_daily_news"
_ALERT_TIME_PATTERN = re.compile(r"^\s*(\d{1,2})\s*[:.]\s*(\d{2})\s*(?:น\.?)?\s*$")
_DAILY_CRON_PATTERN = re.compile(r"^(\d{1,2}) (\d{1,2}) \* \* \*$")
_LAST_HOUR = 23
_LAST_MINUTE = 59


class AlertTimeError(ValueError):
    """Raised when an alert time is not a valid 24-hour HH:MM time."""


def parse_alert_time(text: str) -> str:
    """Normalize a 24-hour time like "9:05", "21.30" or "20.00 น." to "HH:MM".

    Args:
        text: Time typed by the user (":" or "." between hour and minute).

    Returns:
        Time as zero-padded "HH:MM".

    Raises:
        AlertTimeError: When the text is not a valid 24-hour time.

    Example:
        >>> parse_alert_time("21.30")
        '21:30'
    """
    match = _ALERT_TIME_PATTERN.match(twelve_hour_to_24(text or ""))
    hour, minute = (int(match.group(1)), int(match.group(2))) if match else (-1, -1)
    if not (0 <= hour <= _LAST_HOUR and 0 <= minute <= _LAST_MINUTE):
        raise AlertTimeError(
            f"Alert time must be a 24-hour time like 08:30 or 21.00. Received: {text!r}"
        )
    return f"{hour:02d}:{minute:02d}"


def alert_time_to_cron(alert_time: str) -> str:
    """Turn "HH:MM" into a daily cron expression ("MM HH * * *").

    Example:
        >>> alert_time_to_cron("21:30")
        '30 21 * * *'
    """
    hour, minute = parse_alert_time(alert_time).split(":")
    return f"{int(minute)} {int(hour)} * * *"


def cron_to_alert_time(cron_expression: str) -> str | None:
    """Read "HH:MM" back from a daily cron expression, or None if not daily.

    Example:
        >>> cron_to_alert_time("30 21 * * *")
        '21:30'
    """
    match = _DAILY_CRON_PATTERN.match(cron_expression.strip())
    if match is None:
        return None
    return f"{int(match.group(2)):02d}:{int(match.group(1)):02d}"


def get_watchlist_alert_times(session: Session, user_id: str) -> dict[str, str]:
    """Map each symbol with an active daily alert to its "HH:MM" time.

    Example:
        >>> get_watchlist_alert_times(session, uid)
        {'PTT.BK': '08:30'}
    """
    schedules = AssetScheduleCRUD().get_active_by_description(
        session, user_id, WATCHLIST_ALERT_DESCRIPTION
    )
    times = {item.symbol: cron_to_alert_time(item.cron_expression) for item in schedules}
    return {symbol: time for symbol, time in times.items() if time}


def set_watchlist_alert(
    session: Session,
    user_id: str,
    symbol: str,
    alert_time: str,
    session_factory: Callable[[], Session] | None = None,
) -> str:
    """Create or move the daily news alert for one watched asset.

    Args:
        session: Database session.
        user_id: UUID of the user.
        symbol: Canonical symbol (e.g. "PTT.BK").
        alert_time: Daily time, e.g. "08:30" or "21.00".
        session_factory: Factory used by the background job; when given and
            the scheduler is running, the job is (re)registered right away.

    Returns:
        The saved time as "HH:MM".

    Raises:
        AlertTimeError: When alert_time is not a valid 24-hour time.

    Example:
        >>> set_watchlist_alert(session, uid, "PTT.BK", "8.30")
        '08:30'
    """
    normalized_time = parse_alert_time(alert_time)
    schedule = _upsert_alert_schedule(session, user_id, symbol, alert_time_to_cron(normalized_time))
    _register_if_running(schedule, session_factory)
    return normalized_time


def clear_watchlist_alert(session: Session, user_id: str, symbol: str) -> bool:
    """Turn off the daily news alert for one asset.

    Returns:
        True if an active alert was turned off, False if there was none.

    Example:
        >>> clear_watchlist_alert(session, uid, "PTT.BK")
        True
    """
    schedule = AssetScheduleCRUD().get_for_symbol(
        session, user_id, symbol, WATCHLIST_ALERT_DESCRIPTION
    )
    if schedule is None or not schedule.is_active:
        return False
    schedule.is_active = False
    session.commit()
    unregister_schedule(schedule.id)
    return True


def _upsert_alert_schedule(
    session: Session, user_id: str, symbol: str, cron_expression: str
) -> AssetSchedule:
    """Update the asset's alert schedule in place, or create it."""
    crud = AssetScheduleCRUD()
    schedule = crud.get_for_symbol(session, user_id, symbol, WATCHLIST_ALERT_DESCRIPTION)
    if schedule is None:
        return crud.create(session, user_id, symbol, WATCHLIST_ALERT_DESCRIPTION, cron_expression)
    schedule.cron_expression = cron_expression
    schedule.is_active = True
    session.commit()
    return schedule


def _register_if_running(schedule: AssetSchedule, session_factory: Any) -> None:
    """Register the job now if the scheduler runs (else it loads at startup)."""
    if session_factory is not None and get_scheduler().running:
        register_schedule(schedule.id, schedule.cron_expression, session_factory)
