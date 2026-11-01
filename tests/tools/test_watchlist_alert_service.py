"""Tests for watchlist_alert_service (daily news alerts)."""

# pylint: disable=redefined-outer-name

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from finance_ai.database.models.asset_schedule import AssetSchedule
from finance_ai.database.models.user import User
from finance_ai.tools.watchlist_alert_service import (
    WATCHLIST_ALERT_DESCRIPTION,
    AlertTimeError,
    alert_time_to_cron,
    clear_watchlist_alert,
    cron_to_alert_time,
    get_watchlist_alert_times,
    parse_alert_time,
    set_watchlist_alert,
)

_MODULE = "finance_ai.tools.watchlist_alert_service"


@pytest.fixture
def session(test_session: Session) -> Session:
    """Shared in-memory session (tests/conftest.py)."""
    return test_session


@pytest.fixture
def user_id(sample_user: User) -> str:
    """A stored user's id."""
    return str(sample_user.id)


class TestParseAlertTime:
    """Tests for parse_alert_time."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("21:30", "21:30"),
            ("21.30", "21:30"),
            ("8:05", "08:05"),
            (" 00.00 ", "00:00"),
            ("20.00 น.", "20:00"),
            ("8:30น", "08:30"),
            ("8:30 pm", "20:30"),
            ("12 am", "00:00"),
            ("๒๐:๐๐", "20:00"),
        ],
    )
    def test_valid_times(self, text: str, expected: str) -> None:
        """Colon or dot separators, padded output."""
        assert parse_alert_time(text) == expected

    @pytest.mark.parametrize("text", ["24:00", "12:60", "noon", "", "1230", "9:5", "8", "13 pm"])
    def test_invalid_times(self, text: str) -> None:
        """Out-of-range or malformed times are rejected with the input shown."""
        with pytest.raises(AlertTimeError, match="24-hour"):
            parse_alert_time(text)


class TestCronConversion:
    """Tests for alert_time_to_cron and cron_to_alert_time."""

    def test_round_trip(self) -> None:
        """HH:MM → cron → HH:MM."""
        assert alert_time_to_cron("08:05") == "5 8 * * *"
        assert cron_to_alert_time("5 8 * * *") == "08:05"

    def test_non_daily_cron(self) -> None:
        """A weekly cron is not a daily alert time."""
        assert cron_to_alert_time("0 9 * * 1") is None


class TestSetAndClearAlert:
    """Tests for set/clear/get of watchlist alerts."""

    def test_set_creates_schedule(self, session: Session, user_id: str) -> None:
        """First set creates an active daily schedule."""
        assert set_watchlist_alert(session, user_id, "PTT.BK", "8.30") == "08:30"
        assert get_watchlist_alert_times(session, user_id) == {"PTT.BK": "08:30"}

    def test_set_again_moves_time(self, session: Session, user_id: str) -> None:
        """Setting again updates the same row instead of adding one."""
        set_watchlist_alert(session, user_id, "PTT.BK", "08:30")
        set_watchlist_alert(session, user_id, "PTT.BK", "21:00")
        rows = session.query(AssetSchedule).filter_by(description=WATCHLIST_ALERT_DESCRIPTION)
        assert rows.count() == 1
        assert get_watchlist_alert_times(session, user_id) == {"PTT.BK": "21:00"}

    def test_invalid_time_saves_nothing(self, session: Session, user_id: str) -> None:
        """A bad time raises and stores no schedule."""
        with pytest.raises(AlertTimeError):
            set_watchlist_alert(session, user_id, "PTT.BK", "25:00")
        assert not get_watchlist_alert_times(session, user_id)

    def test_clear_then_reenable(self, session: Session, user_id: str) -> None:
        """Clearing deactivates; setting again reactivates the same row."""
        set_watchlist_alert(session, user_id, "PTT.BK", "08:30")
        with patch(f"{_MODULE}.unregister_schedule") as unregister:
            assert clear_watchlist_alert(session, user_id, "PTT.BK") is True
        unregister.assert_called_once()
        assert not get_watchlist_alert_times(session, user_id)
        set_watchlist_alert(session, user_id, "PTT.BK", "09:00")
        assert get_watchlist_alert_times(session, user_id) == {"PTT.BK": "09:00"}

    def test_clear_without_alert(self, session: Session, user_id: str) -> None:
        """Clearing an asset with no alert returns False."""
        assert clear_watchlist_alert(session, user_id, "PTT.BK") is False

    def test_registers_when_scheduler_running(self, session: Session, user_id: str) -> None:
        """A running scheduler gets the job immediately."""
        factory = MagicMock()
        with (
            patch(f"{_MODULE}.get_scheduler") as get_scheduler,
            patch(f"{_MODULE}.register_schedule") as register,
        ):
            get_scheduler.return_value.running = True
            set_watchlist_alert(session, user_id, "PTT.BK", "07:15", factory)
        assert register.call_args.args[1:] == ("15 7 * * *", factory)

    def test_skips_register_when_stopped(self, session: Session, user_id: str) -> None:
        """A stopped scheduler picks the alert up at startup instead."""
        with (
            patch(f"{_MODULE}.get_scheduler") as get_scheduler,
            patch(f"{_MODULE}.register_schedule") as register,
        ):
            get_scheduler.return_value.running = False
            set_watchlist_alert(session, user_id, "PTT.BK", "07:15", MagicMock())
        register.assert_not_called()
