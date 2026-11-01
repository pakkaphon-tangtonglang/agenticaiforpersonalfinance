"""Tests for starting the alert scheduler with the app."""

from unittest.mock import patch

from sqlalchemy.exc import OperationalError

from finance_ai.main import _start_alert_scheduler


def test_starts_scheduler() -> None:
    """The scheduler starts with the app's session factory."""
    with patch("finance_ai.main.start_scheduler", return_value=2) as start:
        _start_alert_scheduler()
    start.assert_called_once()


def test_database_error_does_not_stop_app() -> None:
    """A database error at startup is logged, not raised."""
    error = OperationalError("SELECT 1", {}, Exception("down"))
    with patch("finance_ai.main.start_scheduler", side_effect=error):
        _start_alert_scheduler()


def test_thai_datetime_converts_utc_to_bangkok() -> None:
    """Aware UTC times are shown in Bangkok time, rolling the date over."""
    from datetime import datetime, timezone

    from finance_ai.main import _format_thai_datetime

    value = datetime(2026, 12, 31, 20, 30, tzinfo=timezone.utc)
    assert _format_thai_datetime(value) == "1 ม.ค. 2027, 03:30"
