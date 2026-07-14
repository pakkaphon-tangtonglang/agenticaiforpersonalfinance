"""Tests for AssetSchedule model."""

from sqlalchemy.orm import Session

from finance_ai.database.models.asset_schedule import AssetSchedule
from finance_ai.database.models.user import User


class TestAssetSchedule:
    """Tests for AssetSchedule model."""

    def test_create_schedule(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Create a schedule with required fields."""
        schedule = AssetSchedule(
            user_id=sample_user.id,
            symbol="GC=F",
            description="Gold price check",
            cron_expression="0 21 * * *",
        )
        test_session.add(schedule)
        test_session.commit()
        test_session.refresh(schedule)
        assert schedule.id is not None
        assert schedule.is_active is True
        assert schedule.run_count == 0
        assert schedule.max_runs is None

    def test_defaults(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Verify default field values."""
        schedule = AssetSchedule(
            user_id=sample_user.id,
            symbol="PTT.BK",
            description="PTT",
            cron_expression="0 9 * * 1-5",
        )
        test_session.add(schedule)
        test_session.commit()
        test_session.refresh(schedule)
        assert schedule.is_active is True
        assert schedule.run_count == 0
