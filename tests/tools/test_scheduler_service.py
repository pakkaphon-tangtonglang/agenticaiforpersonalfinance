"""Tests for scheduler_service module."""

from unittest.mock import MagicMock, patch
from decimal import Decimal

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from finance_ai.database.base import Base
from finance_ai.database.models.user import User
from finance_ai.tools.scheduler_service import (
    create_schedule,
    deactivate_schedule,
    delete_schedule,
    execute_scheduled_fetch,
    get_unread_notifications,
    get_user_schedules,
    mark_all_notifications_read,
    mark_notification_read,
)


@pytest.fixture
def engine() -> Engine:
    """Create in-memory SQLite engine."""
    eng = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=eng)
    return eng


@pytest.fixture
def session(engine: Engine) -> Session:  # type: ignore[misc]
    """Create test session."""
    factory = sessionmaker(bind=engine)
    sess = factory()
    try:
        yield sess
    finally:
        sess.rollback()
        sess.close()


@pytest.fixture
def user(session: Session) -> User:
    """Create a test user."""
    u = User(
        email="scheduler-test@example.com",
        hashed_password="hashed",
        full_name="Scheduler Test User",
    )
    session.add(u)
    session.commit()
    session.refresh(u)
    return u


class TestCreateSchedule:
    """Tests for create_schedule."""

    def test_creates_schedule(self, session: Session, user: User) -> None:
        """Create a schedule via the service layer."""
        schedule = create_schedule(
            session,
            user.id,
            "GC=F",
            "ราคาทอง",
            "0 21 * * *",
        )
        assert schedule.symbol == "GC=F"
        assert schedule.is_active is True


class TestGetUserSchedules:
    """Tests for get_user_schedules."""

    def test_returns_all(self, session: Session, user: User) -> None:
        """Return all schedules when active_only=False."""
        create_schedule(session, user.id, "GC=F", "ทอง", "0 21 * * *")
        s2 = create_schedule(session, user.id, "PTT.BK", "PTT", "0 9 * * *")
        deactivate_schedule(session, s2.id, user.id)

        result = get_user_schedules(session, user.id, active_only=False)
        assert len(result) == 2

    def test_returns_active_only(self, session: Session, user: User) -> None:
        """Return only active schedules."""
        create_schedule(session, user.id, "GC=F", "ทอง", "0 21 * * *")
        s2 = create_schedule(session, user.id, "PTT.BK", "PTT", "0 9 * * *")
        deactivate_schedule(session, s2.id, user.id)

        result = get_user_schedules(session, user.id, active_only=True)
        assert len(result) == 1
        assert result[0].symbol == "GC=F"


class TestDeleteSchedule:
    """Tests for delete_schedule."""

    def test_delete_existing(self, session: Session, user: User) -> None:
        """Delete an existing schedule."""
        schedule = create_schedule(
            session,
            user.id,
            "GC=F",
            "ทอง",
            "0 21 * * *",
        )
        assert delete_schedule(session, schedule.id, user.id) is True

    def test_delete_nonexistent(self, session: Session, user: User) -> None:
        """Return False for nonexistent schedule."""
        assert delete_schedule(session, "fake-id", user.id) is False


class TestDeactivateSchedule:
    """Tests for deactivate_schedule."""

    def test_deactivate(self, session: Session, user: User) -> None:
        """Deactivate an existing schedule."""
        schedule = create_schedule(
            session,
            user.id,
            "GC=F",
            "ทอง",
            "0 21 * * *",
        )
        assert deactivate_schedule(session, schedule.id, user.id) is True


class TestExecuteScheduledFetch:
    """Tests for execute_scheduled_fetch."""

    @patch("finance_ai.tools.market_data_service.fetch_finance_news")
    @patch("finance_ai.tools.price_client.fetch_current_price")
    def test_fetch_success(
        self,
        mock_price: MagicMock,
        mock_news: MagicMock,
        session: Session,
        user: User,
    ) -> None:
        """Successful fetch creates notification with price and news."""
        mock_price.return_value = Decimal("2350.00")
        mock_news_result = MagicMock()
        mock_news_result.has_news = True
        mock_news_result.news_content = "Gold prices rise"
        mock_news.return_value = mock_news_result
        schedule = create_schedule(
            session,
            user.id,
            "GC=F",
            "ทอง",
            "0 21 * * *",
        )

        notification = execute_scheduled_fetch(session, schedule)
        assert notification.symbol == "GC=F"
        assert "2,350.00" in notification.content
        assert "Gold prices rise" in notification.content
        assert notification.schedule_id == schedule.id
        assert notification.is_read is False

    @patch("finance_ai.tools.market_data_service.fetch_finance_news")
    @patch("finance_ai.tools.price_client.fetch_current_price")
    def test_fetch_failure(
        self,
        mock_price: MagicMock,
        mock_news: MagicMock,
        session: Session,
        user: User,
    ) -> None:
        """Failed fetch creates notification with error message."""
        mock_price.return_value = None
        mock_news_result = MagicMock()
        mock_news_result.has_news = False
        mock_news.return_value = mock_news_result
        schedule = create_schedule(
            session,
            user.id,
            "GC=F",
            "ทอง",
            "0 21 * * *",
        )

        notification = execute_scheduled_fetch(session, schedule)
        assert "ไม่สามารถดึงราคา" in notification.content


class TestNotificationService:
    """Tests for notification service functions."""

    def test_get_unread(self, session: Session, user: User) -> None:
        """Get unread notifications."""
        from finance_ai.database.crud.schedule_crud import (
            AssetNotificationCRUD,
        )

        crud = AssetNotificationCRUD()
        crud.create(session, user.id, "GC=F", "ราคา 2,350")

        result = get_unread_notifications(session, user.id)
        assert len(result) == 1

    def test_mark_read(self, session: Session, user: User) -> None:
        """Mark a notification as read."""
        from finance_ai.database.crud.schedule_crud import (
            AssetNotificationCRUD,
        )

        crud = AssetNotificationCRUD()
        n = crud.create(session, user.id, "GC=F", "ราคา 2,350")

        assert mark_notification_read(session, n.id, user.id) is True
        assert get_unread_notifications(session, user.id) == []

    def test_mark_all_read(self, session: Session, user: User) -> None:
        """Mark all notifications as read."""
        from finance_ai.database.crud.schedule_crud import (
            AssetNotificationCRUD,
        )

        crud = AssetNotificationCRUD()
        crud.create(session, user.id, "GC=F", "ราคา 2,350")
        crud.create(session, user.id, "PTT.BK", "ราคา 35.50")

        count = mark_all_notifications_read(session, user.id)
        assert count == 2
