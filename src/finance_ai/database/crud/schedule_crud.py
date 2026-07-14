"""CRUD operations for AssetSchedule and AssetNotification models."""

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from finance_ai.database.crud.base_crud import BaseCRUD
from finance_ai.database.models.asset_notification import AssetNotification
from finance_ai.database.models.asset_schedule import AssetSchedule


class AssetScheduleCRUD(BaseCRUD[AssetSchedule]):
    """CRUD operations for asset monitoring schedules."""

    def __init__(self) -> None:
        """Initialize with the AssetSchedule model."""
        super().__init__(AssetSchedule)

    def create(  # type: ignore[override]  # pylint: disable=arguments-differ,too-many-arguments,too-many-positional-arguments
        self,
        session: Session,
        user_id: str,
        symbol: str,
        description: str,
        cron_expression: str,
        max_runs: int | None = None,
    ) -> AssetSchedule:
        """Create a new schedule."""
        schedule = AssetSchedule(
            user_id=user_id,
            symbol=symbol.strip().upper(),
            description=description,
            cron_expression=cron_expression,
            max_runs=max_runs,
        )
        session.add(schedule)
        session.commit()
        session.refresh(schedule)
        return schedule

    def get_by_user(self, session: Session, user_id: str) -> list[AssetSchedule]:
        """Get all schedules for a user."""
        stmt = (
            select(AssetSchedule)
            .where(AssetSchedule.user_id == user_id)
            .order_by(AssetSchedule.created_at.desc())
        )
        return list(session.execute(stmt).scalars().all())

    def get_active_by_user(self, session: Session, user_id: str) -> list[AssetSchedule]:
        """Get active schedules for a user."""
        stmt = (
            select(AssetSchedule)
            .where(
                AssetSchedule.user_id == user_id,
                AssetSchedule.is_active.is_(True),
            )
            .order_by(AssetSchedule.created_at.desc())
        )
        return list(session.execute(stmt).scalars().all())

    def get_all_active(self, session: Session) -> list[AssetSchedule]:
        """Get all active schedules across all users."""
        stmt = select(AssetSchedule).where(AssetSchedule.is_active.is_(True))
        return list(session.execute(stmt).scalars().all())

    def delete(  # type: ignore[override]  # pylint: disable=arguments-differ
        self,
        session: Session,
        schedule_id: str,
        user_id: str,
    ) -> bool:
        """Delete a schedule by ID and user."""
        stmt = select(AssetSchedule).where(
            AssetSchedule.id == schedule_id,
            AssetSchedule.user_id == user_id,
        )
        schedule = session.execute(stmt).scalar_one_or_none()
        if schedule is None:
            return False
        session.delete(schedule)
        session.commit()
        return True

    def deactivate(self, session: Session, schedule_id: str, user_id: str) -> bool:
        """Deactivate a schedule."""
        stmt = select(AssetSchedule).where(
            AssetSchedule.id == schedule_id,
            AssetSchedule.user_id == user_id,
        )
        schedule = session.execute(stmt).scalar_one_or_none()
        if schedule is None:
            return False
        schedule.is_active = False
        session.commit()
        return True

    def increment_run_count(self, session: Session, schedule_id: str) -> int:
        """Increment the run count for a schedule."""
        stmt = select(AssetSchedule).where(AssetSchedule.id == schedule_id)
        schedule = session.execute(stmt).scalar_one_or_none()
        if schedule is None:
            return -1
        schedule.run_count = (schedule.run_count or 0) + 1
        session.commit()
        return schedule.run_count


class AssetNotificationCRUD(BaseCRUD[AssetNotification]):
    """CRUD operations for asset notifications."""

    def __init__(self) -> None:
        """Initialize with the AssetNotification model."""
        super().__init__(AssetNotification)

    def create(  # type: ignore[override]  # pylint: disable=arguments-differ,too-many-arguments,too-many-positional-arguments
        self,
        session: Session,
        user_id: str,
        symbol: str,
        content: str,
        schedule_id: str | None = None,
    ) -> AssetNotification:
        """Create a new notification."""
        notification = AssetNotification(
            user_id=user_id,
            symbol=symbol.strip().upper(),
            content=content,
            schedule_id=schedule_id,
        )
        session.add(notification)
        session.commit()
        session.refresh(notification)
        return notification

    def get_unread_by_user(self, session: Session, user_id: str) -> list[AssetNotification]:
        """Get unread notifications for a user."""
        stmt = (
            select(AssetNotification)
            .where(
                AssetNotification.user_id == user_id,
                AssetNotification.is_read.is_(False),
            )
            .order_by(AssetNotification.created_at.desc())
        )
        return list(session.execute(stmt).scalars().all())

    def mark_as_read(self, session: Session, notification_id: str, user_id: str) -> bool:
        """Mark a notification as read."""
        stmt = select(AssetNotification).where(
            AssetNotification.id == notification_id,
            AssetNotification.user_id == user_id,
        )
        notification = session.execute(stmt).scalar_one_or_none()
        if notification is None:
            return False
        notification.is_read = True
        session.commit()
        return True

    def mark_all_as_read(self, session: Session, user_id: str) -> int:
        """Mark all notifications as read for a user."""
        stmt = (
            update(AssetNotification)
            .where(
                AssetNotification.user_id == user_id,
                AssetNotification.is_read.is_(False),
            )
            .values(is_read=True)
        )
        result = session.execute(stmt)
        session.commit()
        return int(result.rowcount)  # type: ignore[attr-defined]
