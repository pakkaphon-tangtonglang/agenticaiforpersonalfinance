"""Tests for background scheduler module."""

from unittest.mock import MagicMock, patch

import pytest

from finance_ai.tools.background_scheduler import (
    _parse_cron_expression,
    get_scheduler,
    load_all_schedules,
    register_schedule,
    start_scheduler,
    stop_scheduler,
    unregister_schedule,
)

MODULE_PATH = "finance_ai.tools.background_scheduler"


class TestParseCronExpression:
    """Tests for _parse_cron_expression."""

    def test_standard_cron(self) -> None:
        """Parses a standard 5-field cron expression."""
        result = _parse_cron_expression("42 11 * * *")
        assert result == {
            "minute": "42",
            "hour": "11",
            "day": "*",
            "month": "*",
            "day_of_week": "*",
        }

    def test_weekday_cron(self) -> None:
        """Parses cron with day-of-week restriction."""
        result = _parse_cron_expression("0 9 * * 1-5")
        assert result["day_of_week"] == "1-5"
        assert result["hour"] == "9"

    def test_invalid_field_count(self) -> None:
        """Raises ValueError for invalid cron expression."""
        with pytest.raises(ValueError, match="Invalid cron expression"):
            _parse_cron_expression("* *")

    def test_six_fields_rejected(self) -> None:
        """Rejects 6-field cron expressions."""
        with pytest.raises(ValueError, match="Expected 5 fields"):
            _parse_cron_expression("0 0 0 * * *")


class TestGetScheduler:
    """Tests for get_scheduler."""

    def test_returns_scheduler(self) -> None:
        """Returns a BackgroundScheduler instance."""
        stop_scheduler()  # reset singleton
        scheduler = get_scheduler()
        assert scheduler is not None
        stop_scheduler()

    def test_returns_same_instance(self) -> None:
        """Returns the same singleton instance."""
        stop_scheduler()
        s1 = get_scheduler()
        s2 = get_scheduler()
        assert s1 is s2
        stop_scheduler()


class TestRegisterSchedule:
    """Tests for register_schedule."""

    def test_registers_job(self) -> None:
        """Registers a cron job in the scheduler."""
        stop_scheduler()
        factory = MagicMock()
        register_schedule("sched-1", "42 11 * * *", factory)

        scheduler = get_scheduler()
        job = scheduler.get_job("asset_schedule_sched-1")
        assert job is not None
        stop_scheduler()

    def test_replaces_existing_job(self) -> None:
        """Re-registering replaces the existing job."""
        stop_scheduler()
        factory = MagicMock()
        register_schedule("sched-1", "42 11 * * *", factory)
        register_schedule("sched-1", "0 9 * * *", factory)

        scheduler = get_scheduler()
        jobs = scheduler.get_jobs()
        matching = [j for j in jobs if j.id == "asset_schedule_sched-1"]
        assert len(matching) == 1
        stop_scheduler()


class TestUnregisterSchedule:
    """Tests for unregister_schedule."""

    def test_removes_existing_job(self) -> None:
        """Removes a registered job."""
        stop_scheduler()
        factory = MagicMock()
        register_schedule("sched-2", "0 21 * * *", factory)
        unregister_schedule("sched-2")

        scheduler = get_scheduler()
        assert scheduler.get_job("asset_schedule_sched-2") is None
        stop_scheduler()

    def test_noop_for_nonexistent(self) -> None:
        """Does nothing for non-existent job."""
        stop_scheduler()
        unregister_schedule("nonexistent")  # should not raise
        stop_scheduler()


class TestLoadAllSchedules:
    """Tests for load_all_schedules."""

    @patch(f"{MODULE_PATH}.register_schedule")
    def test_loads_active_schedules(
        self,
        mock_register: MagicMock,
    ) -> None:
        """Loads active schedules from DB and registers them."""
        stop_scheduler()
        mock_schedule_1 = MagicMock()
        mock_schedule_1.id = "s1"
        mock_schedule_1.cron_expression = "42 11 * * *"
        mock_schedule_2 = MagicMock()
        mock_schedule_2.id = "s2"
        mock_schedule_2.cron_expression = "0 9 * * 1-5"

        mock_session = MagicMock()
        mock_crud_instance = MagicMock()
        mock_crud_instance.get_all_active.return_value = [
            mock_schedule_1,
            mock_schedule_2,
        ]

        factory = MagicMock(return_value=mock_session)

        with patch(
            "finance_ai.database.crud.schedule_crud.AssetScheduleCRUD",
            return_value=mock_crud_instance,
        ):
            count = load_all_schedules(factory)

        assert count == 2
        assert mock_register.call_count == 2
        stop_scheduler()

    @patch(f"{MODULE_PATH}.register_schedule")
    def test_returns_zero_when_no_schedules(
        self,
        mock_register: MagicMock,
    ) -> None:
        """Returns 0 when no active schedules exist."""
        stop_scheduler()
        mock_session = MagicMock()
        mock_crud_instance = MagicMock()
        mock_crud_instance.get_all_active.return_value = []

        factory = MagicMock(return_value=mock_session)

        with patch(
            "finance_ai.database.crud.schedule_crud.AssetScheduleCRUD",
            return_value=mock_crud_instance,
        ):
            count = load_all_schedules(factory)

        assert count == 0
        mock_register.assert_not_called()
        stop_scheduler()


class TestStartStopScheduler:
    """Tests for start_scheduler and stop_scheduler."""

    @patch(f"{MODULE_PATH}.load_all_schedules", return_value=3)
    def test_start_scheduler(self, mock_load: MagicMock) -> None:
        """Starts scheduler and returns schedule count."""
        stop_scheduler()
        factory = MagicMock()
        count = start_scheduler(factory)

        assert count == 3
        scheduler = get_scheduler()
        assert scheduler.running is True
        stop_scheduler()

    @patch(f"{MODULE_PATH}.load_all_schedules", return_value=0)
    def test_start_twice_is_safe(self, mock_load: MagicMock) -> None:
        """Calling start_scheduler twice doesn't crash."""
        stop_scheduler()
        factory = MagicMock()
        start_scheduler(factory)
        start_scheduler(factory)  # should not raise

        scheduler = get_scheduler()
        assert scheduler.running is True
        stop_scheduler()

    def test_stop_scheduler(self) -> None:
        """Stops a running scheduler."""
        stop_scheduler()
        scheduler = get_scheduler()
        scheduler.start()
        assert scheduler.running is True

        stop_scheduler()
        # After stop, singleton is reset
        new_scheduler = get_scheduler()
        assert new_scheduler.running is False
        stop_scheduler()
