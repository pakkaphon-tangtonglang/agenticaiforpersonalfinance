"""Tests for AssetScheduleCRUD and AssetNotificationCRUD."""

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from finance_ai.database.base import Base
from finance_ai.database.crud.schedule_crud import (
    AssetNotificationCRUD,
    AssetScheduleCRUD,
)
from finance_ai.database.models.user import User


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
        email="sched-test@example.com",
        hashed_password="hashed",
        full_name="Schedule Test User",
    )
    session.add(u)
    session.commit()
    session.refresh(u)
    return u


class TestAssetScheduleCRUD:
    """Tests for AssetScheduleCRUD."""

    def test_create_schedule(self, session: Session, user: User) -> None:
        """Create a schedule and verify fields."""
        crud = AssetScheduleCRUD(session)
        schedule = crud.create(user.id, "GC=F", "ราคาทอง", "0 21 * * *")

        assert schedule.id is not None
        assert schedule.user_id == user.id
        assert schedule.symbol == "GC=F"
        assert schedule.description == "ราคาทอง"
        assert schedule.cron_expression == "0 21 * * *"
        assert schedule.is_active is True

    def test_create_normalizes_symbol(self, session: Session, user: User) -> None:
        """Symbol should be stripped and uppercased."""
        crud = AssetScheduleCRUD(session)
        schedule = crud.create(user.id, "  gc=f  ", "ทอง", "0 9 * * *")

        assert schedule.symbol == "GC=F"

    def test_get_by_user(self, session: Session, user: User) -> None:
        """Get all schedules for a user."""
        crud = AssetScheduleCRUD(session)
        crud.create(user.id, "GC=F", "ทอง", "0 21 * * *")
        crud.create(user.id, "PTT.BK", "PTT", "0 9 * * 1-5")

        schedules = crud.get_by_user(user.id)
        assert len(schedules) == 2

    def test_get_by_user_empty(self, session: Session, user: User) -> None:
        """Return empty list when no schedules exist."""
        crud = AssetScheduleCRUD(session)
        assert crud.get_by_user(user.id) == []

    def test_get_active_by_user(self, session: Session, user: User) -> None:
        """Only return active schedules."""
        crud = AssetScheduleCRUD(session)
        s1 = crud.create(user.id, "GC=F", "ทอง", "0 21 * * *")
        crud.create(user.id, "PTT.BK", "PTT", "0 9 * * 1-5")
        crud.deactivate(s1.id, user.id)

        active = crud.get_active_by_user(user.id)
        assert len(active) == 1
        assert active[0].symbol == "PTT.BK"

    def test_delete_success(self, session: Session, user: User) -> None:
        """Delete an existing schedule returns True."""
        crud = AssetScheduleCRUD(session)
        schedule = crud.create(user.id, "GC=F", "ทอง", "0 21 * * *")

        assert crud.delete(schedule.id, user.id) is True
        assert crud.get_by_user(user.id) == []

    def test_delete_not_found(self, session: Session, user: User) -> None:
        """Delete a non-existent schedule returns False."""
        crud = AssetScheduleCRUD(session)
        assert crud.delete("nonexistent", user.id) is False

    def test_delete_wrong_user(self, session: Session, user: User) -> None:
        """Cannot delete another user's schedule."""
        crud = AssetScheduleCRUD(session)
        schedule = crud.create(user.id, "GC=F", "ทอง", "0 21 * * *")

        assert crud.delete(schedule.id, "other-user") is False

    def test_deactivate_success(self, session: Session, user: User) -> None:
        """Deactivate an existing schedule."""
        crud = AssetScheduleCRUD(session)
        schedule = crud.create(user.id, "GC=F", "ทอง", "0 21 * * *")

        assert crud.deactivate(schedule.id, user.id) is True
        schedules = crud.get_by_user(user.id)
        assert schedules[0].is_active is False

    def test_deactivate_not_found(self, session: Session, user: User) -> None:
        """Deactivate a non-existent schedule returns False."""
        crud = AssetScheduleCRUD(session)
        assert crud.deactivate("nonexistent", user.id) is False


class TestAssetNotificationCRUD:
    """Tests for AssetNotificationCRUD."""

    def test_create_notification(self, session: Session, user: User) -> None:
        """Create a notification and verify fields."""
        crud = AssetNotificationCRUD(session)
        notification = crud.create(user.id, "GC=F", "ราคาทอง: 2,350 USD")

        assert notification.id is not None
        assert notification.user_id == user.id
        assert notification.symbol == "GC=F"
        assert notification.content == "ราคาทอง: 2,350 USD"
        assert notification.is_read is False
        assert notification.schedule_id is None

    def test_create_with_schedule_id(self, session: Session, user: User) -> None:
        """Create notification linked to a schedule."""
        sched_crud = AssetScheduleCRUD(session)
        schedule = sched_crud.create(user.id, "GC=F", "ทอง", "0 21 * * *")

        notif_crud = AssetNotificationCRUD(session)
        notification = notif_crud.create(
            user.id,
            "GC=F",
            "ราคา 2,350",
            schedule.id,
        )
        assert notification.schedule_id == schedule.id

    def test_get_unread_by_user(self, session: Session, user: User) -> None:
        """Get only unread notifications."""
        crud = AssetNotificationCRUD(session)
        n1 = crud.create(user.id, "GC=F", "ราคา 2,350")
        crud.create(user.id, "PTT.BK", "ราคา 35.50")
        crud.mark_as_read(n1.id, user.id)

        unread = crud.get_unread_by_user(user.id)
        assert len(unread) == 1
        assert unread[0].symbol == "PTT.BK"

    def test_get_unread_empty(self, session: Session, user: User) -> None:
        """Return empty list when no unread notifications."""
        crud = AssetNotificationCRUD(session)
        assert crud.get_unread_by_user(user.id) == []

    def test_mark_as_read(self, session: Session, user: User) -> None:
        """Mark a notification as read."""
        crud = AssetNotificationCRUD(session)
        notification = crud.create(user.id, "GC=F", "ราคา 2,350")

        assert crud.mark_as_read(notification.id, user.id) is True
        assert crud.get_unread_by_user(user.id) == []

    def test_mark_as_read_not_found(self, session: Session, user: User) -> None:
        """Mark non-existent notification returns False."""
        crud = AssetNotificationCRUD(session)
        assert crud.mark_as_read("nonexistent", user.id) is False

    def test_mark_all_as_read(self, session: Session, user: User) -> None:
        """Mark all notifications as read."""
        crud = AssetNotificationCRUD(session)
        crud.create(user.id, "GC=F", "ราคา 2,350")
        crud.create(user.id, "PTT.BK", "ราคา 35.50")

        count = crud.mark_all_as_read(user.id)
        assert count == 2
        assert crud.get_unread_by_user(user.id) == []

    def test_mark_all_as_read_none_unread(self, session: Session, user: User) -> None:
        """Mark all returns 0 when none are unread."""
        crud = AssetNotificationCRUD(session)
        assert crud.mark_all_as_read(user.id) == 0
