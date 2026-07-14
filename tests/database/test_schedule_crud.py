"""Tests for AssetScheduleCRUD and AssetNotificationCRUD."""

from sqlalchemy.orm import Session

from finance_ai.database.crud.schedule_crud import (
    AssetNotificationCRUD,
    AssetScheduleCRUD,
)
from finance_ai.database.models.user import User


class TestAssetScheduleCRUD:
    """Tests for AssetScheduleCRUD."""

    def test_create_schedule(self, test_session: Session, sample_user: User) -> None:
        """Create a schedule and verify fields."""
        crud = AssetScheduleCRUD()
        schedule = crud.create(test_session, sample_user.id, "GC=F", "ราคาทอง", "0 21 * * *")

        assert schedule.id is not None
        assert schedule.user_id == sample_user.id
        assert schedule.symbol == "GC=F"
        assert schedule.description == "ราคาทอง"
        assert schedule.cron_expression == "0 21 * * *"
        assert schedule.is_active is True

    def test_create_normalizes_symbol(self, test_session: Session, sample_user: User) -> None:
        """Symbol should be stripped and uppercased."""
        crud = AssetScheduleCRUD()
        schedule = crud.create(test_session, sample_user.id, "  gc=f  ", "ทอง", "0 9 * * *")

        assert schedule.symbol == "GC=F"

    def test_get_by_user(self, test_session: Session, sample_user: User) -> None:
        """Get all schedules for a user."""
        crud = AssetScheduleCRUD()
        crud.create(test_session, sample_user.id, "GC=F", "ทอง", "0 21 * * *")
        crud.create(test_session, sample_user.id, "PTT.BK", "PTT", "0 9 * * 1-5")

        schedules = crud.get_by_user(test_session, sample_user.id)
        assert len(schedules) == 2

    def test_get_by_user_empty(self, test_session: Session, sample_user: User) -> None:
        """Return empty list when no schedules exist."""
        crud = AssetScheduleCRUD()
        assert crud.get_by_user(test_session, sample_user.id) == []

    def test_get_active_by_user(self, test_session: Session, sample_user: User) -> None:
        """Only return active schedules."""
        crud = AssetScheduleCRUD()
        s1 = crud.create(test_session, sample_user.id, "GC=F", "ทอง", "0 21 * * *")
        crud.create(test_session, sample_user.id, "PTT.BK", "PTT", "0 9 * * 1-5")
        crud.deactivate(test_session, s1.id, sample_user.id)

        active = crud.get_active_by_user(test_session, sample_user.id)
        assert len(active) == 1
        assert active[0].symbol == "PTT.BK"

    def test_delete_success(self, test_session: Session, sample_user: User) -> None:
        """Delete an existing schedule returns True."""
        crud = AssetScheduleCRUD()
        schedule = crud.create(test_session, sample_user.id, "GC=F", "ทอง", "0 21 * * *")

        assert crud.delete(test_session, schedule.id, sample_user.id) is True
        assert crud.get_by_user(test_session, sample_user.id) == []

    def test_delete_not_found(self, test_session: Session, sample_user: User) -> None:
        """Delete a non-existent schedule returns False."""
        crud = AssetScheduleCRUD()
        assert crud.delete(test_session, "nonexistent", sample_user.id) is False

    def test_delete_wrong_user(self, test_session: Session, sample_user: User) -> None:
        """Cannot delete another user's schedule."""
        crud = AssetScheduleCRUD()
        schedule = crud.create(test_session, sample_user.id, "GC=F", "ทอง", "0 21 * * *")

        assert crud.delete(test_session, schedule.id, "other-user") is False

    def test_deactivate_success(self, test_session: Session, sample_user: User) -> None:
        """Deactivate an existing schedule."""
        crud = AssetScheduleCRUD()
        schedule = crud.create(test_session, sample_user.id, "GC=F", "ทอง", "0 21 * * *")

        assert crud.deactivate(test_session, schedule.id, sample_user.id) is True
        schedules = crud.get_by_user(test_session, sample_user.id)
        assert schedules[0].is_active is False

    def test_deactivate_not_found(self, test_session: Session, sample_user: User) -> None:
        """Deactivate a non-existent schedule returns False."""
        crud = AssetScheduleCRUD()
        assert crud.deactivate(test_session, "nonexistent", sample_user.id) is False


class TestAssetNotificationCRUD:
    """Tests for AssetNotificationCRUD."""

    def test_create_notification(self, test_session: Session, sample_user: User) -> None:
        """Create a notification and verify fields."""
        crud = AssetNotificationCRUD()
        notification = crud.create(test_session, sample_user.id, "GC=F", "ราคาทอง: 2,350 USD")

        assert notification.id is not None
        assert notification.user_id == sample_user.id
        assert notification.symbol == "GC=F"
        assert notification.content == "ราคาทอง: 2,350 USD"
        assert notification.is_read is False
        assert notification.schedule_id is None

    def test_create_with_schedule_id(self, test_session: Session, sample_user: User) -> None:
        """Create notification linked to a schedule."""
        sched_crud = AssetScheduleCRUD()
        schedule = sched_crud.create(test_session, sample_user.id, "GC=F", "ทอง", "0 21 * * *")

        notif_crud = AssetNotificationCRUD()
        notification = notif_crud.create(
            test_session,
            sample_user.id,
            "GC=F",
            "ราคา 2,350",
            schedule.id,
        )
        assert notification.schedule_id == schedule.id

    def test_get_unread_by_user(self, test_session: Session, sample_user: User) -> None:
        """Get only unread notifications."""
        crud = AssetNotificationCRUD()
        n1 = crud.create(test_session, sample_user.id, "GC=F", "ราคา 2,350")
        crud.create(test_session, sample_user.id, "PTT.BK", "ราคา 35.50")
        crud.mark_as_read(test_session, n1.id, sample_user.id)

        unread = crud.get_unread_by_user(test_session, sample_user.id)
        assert len(unread) == 1
        assert unread[0].symbol == "PTT.BK"

    def test_get_unread_empty(self, test_session: Session, sample_user: User) -> None:
        """Return empty list when no unread notifications."""
        crud = AssetNotificationCRUD()
        assert crud.get_unread_by_user(test_session, sample_user.id) == []

    def test_mark_as_read(self, test_session: Session, sample_user: User) -> None:
        """Mark a notification as read."""
        crud = AssetNotificationCRUD()
        notification = crud.create(test_session, sample_user.id, "GC=F", "ราคา 2,350")

        assert crud.mark_as_read(test_session, notification.id, sample_user.id) is True
        assert crud.get_unread_by_user(test_session, sample_user.id) == []

    def test_mark_as_read_not_found(self, test_session: Session, sample_user: User) -> None:
        """Mark non-existent notification returns False."""
        crud = AssetNotificationCRUD()
        assert crud.mark_as_read(test_session, "nonexistent", sample_user.id) is False

    def test_mark_all_as_read(self, test_session: Session, sample_user: User) -> None:
        """Mark all notifications as read."""
        crud = AssetNotificationCRUD()
        crud.create(test_session, sample_user.id, "GC=F", "ราคา 2,350")
        crud.create(test_session, sample_user.id, "PTT.BK", "ราคา 35.50")

        count = crud.mark_all_as_read(test_session, sample_user.id)
        assert count == 2
        assert crud.get_unread_by_user(test_session, sample_user.id) == []

    def test_mark_all_as_read_none_unread(self, test_session: Session, sample_user: User) -> None:
        """Mark all returns 0 when none are unread."""
        crud = AssetNotificationCRUD()
        assert crud.mark_all_as_read(test_session, sample_user.id) == 0
