"""CRUD operations for asset schedules and notifications."""

import uuid
from typing import Optional

from sqlalchemy.orm import Session

from finance_ai.database.models.asset_notification import AssetNotification
from finance_ai.database.models.asset_schedule import AssetSchedule


class AssetScheduleCRUD:
    """CRUD for AssetSchedule model.

    Example:
        >>> crud = AssetScheduleCRUD(session)
        >>> schedule = crud.create("user-1", "GC=F", "ราคาทอง", "0 21 * * *")
    """

    def __init__(self, session: Session) -> None:
        """Initialize with database session.

        Args:
            session: SQLAlchemy session.
        """
        self.session = session

    def create(
        self,
        user_id: str,
        symbol: str,
        description: str,
        cron_expression: str,
        max_runs: Optional[int] = None,
    ) -> AssetSchedule:
        """Create a new asset schedule.

        Args:
            user_id: UUID of the user.
            symbol: Ticker symbol to monitor.
            description: Human-readable description.
            cron_expression: Cron expression for timing.
            max_runs: Max number of executions (None = unlimited).

        Returns:
            Created AssetSchedule instance.
        """
        schedule = AssetSchedule(
            id=str(uuid.uuid4()),
            user_id=user_id,
            symbol=symbol.strip().upper(),
            description=description,
            cron_expression=cron_expression,
            max_runs=max_runs,
            run_count=0,
        )
        self.session.add(schedule)
        self.session.commit()
        return schedule

    def get_by_user(self, user_id: str) -> list[AssetSchedule]:
        """Get all schedules for a user.

        Args:
            user_id: UUID of the user.

        Returns:
            List of AssetSchedule instances.
        """
        return (
            self.session.query(AssetSchedule)
            .filter(AssetSchedule.user_id == user_id)
            .order_by(AssetSchedule.created_at.desc())
            .all()
        )

    def get_active_by_user(self, user_id: str) -> list[AssetSchedule]:
        """Get active schedules for a user.

        Args:
            user_id: UUID of the user.

        Returns:
            List of active AssetSchedule instances.
        """
        return (
            self.session.query(AssetSchedule)
            .filter(
                AssetSchedule.user_id == user_id,
                AssetSchedule.is_active.is_(True),
            )
            .order_by(AssetSchedule.created_at.desc())
            .all()
        )

    def get_by_user_and_id(
        self,
        schedule_id: str,
    ) -> Optional[AssetSchedule]:
        """Get a single schedule by ID.

        Args:
            schedule_id: UUID of the schedule.

        Returns:
            AssetSchedule or None if not found.
        """
        return self.session.query(AssetSchedule).filter(AssetSchedule.id == schedule_id).first()

    def get_all_active(self) -> list[AssetSchedule]:
        """Get all active schedules across all users.

        Returns:
            List of active AssetSchedule instances.
        """
        return self.session.query(AssetSchedule).filter(AssetSchedule.is_active.is_(True)).all()

    def delete(self, schedule_id: str, user_id: str) -> bool:
        """Delete a schedule by ID (must belong to user).

        Args:
            schedule_id: UUID of the schedule.
            user_id: UUID of the user.

        Returns:
            True if deleted, False if not found.
        """
        schedule = (
            self.session.query(AssetSchedule)
            .filter(
                AssetSchedule.id == schedule_id,
                AssetSchedule.user_id == user_id,
            )
            .first()
        )
        if schedule is None:
            return False
        self.session.delete(schedule)
        self.session.commit()
        return True

    def increment_run_count(self, schedule_id: str) -> int:
        """Increment run_count for a schedule and return new count.

        Args:
            schedule_id: UUID of the schedule.

        Returns:
            New run_count value, or -1 if not found.
        """
        schedule = self.session.query(AssetSchedule).filter(AssetSchedule.id == schedule_id).first()
        if schedule is None:
            return -1
        schedule.run_count = (schedule.run_count or 0) + 1
        self.session.commit()
        return schedule.run_count

    def deactivate(self, schedule_id: str, user_id: str) -> bool:
        """Deactivate a schedule by ID.

        Args:
            schedule_id: UUID of the schedule.
            user_id: UUID of the user.

        Returns:
            True if deactivated, False if not found.
        """
        schedule = (
            self.session.query(AssetSchedule)
            .filter(
                AssetSchedule.id == schedule_id,
                AssetSchedule.user_id == user_id,
            )
            .first()
        )
        if schedule is None:
            return False
        schedule.is_active = False
        self.session.commit()
        return True


class AssetNotificationCRUD:
    """CRUD for AssetNotification model.

    Example:
        >>> crud = AssetNotificationCRUD(session)
        >>> note = crud.create("user-1", "GC=F", "ราคาทอง: 2,350 USD")
    """

    def __init__(self, session: Session) -> None:
        """Initialize with database session.

        Args:
            session: SQLAlchemy session.
        """
        self.session = session

    def create(
        self,
        user_id: str,
        symbol: str,
        content: str,
        schedule_id: Optional[str] = None,
    ) -> AssetNotification:
        """Create a new notification.

        Args:
            user_id: UUID of the user.
            symbol: Ticker symbol.
            content: Notification content text.
            schedule_id: Optional UUID of the schedule that triggered this.

        Returns:
            Created AssetNotification instance.
        """
        notification = AssetNotification(
            id=str(uuid.uuid4()),
            user_id=user_id,
            schedule_id=schedule_id,
            symbol=symbol,
            content=content,
        )
        self.session.add(notification)
        self.session.commit()
        return notification

    def get_unread_by_user(self, user_id: str) -> list[AssetNotification]:
        """Get unread notifications for a user.

        Args:
            user_id: UUID of the user.

        Returns:
            List of unread AssetNotification instances.
        """
        return (
            self.session.query(AssetNotification)
            .filter(
                AssetNotification.user_id == user_id,
                AssetNotification.is_read.is_(False),
            )
            .order_by(AssetNotification.created_at.desc())
            .all()
        )

    def mark_as_read(self, notification_id: str, user_id: str) -> bool:
        """Mark a notification as read.

        Args:
            notification_id: UUID of the notification.
            user_id: UUID of the user.

        Returns:
            True if marked, False if not found.
        """
        notification = (
            self.session.query(AssetNotification)
            .filter(
                AssetNotification.id == notification_id,
                AssetNotification.user_id == user_id,
            )
            .first()
        )
        if notification is None:
            return False
        notification.is_read = True
        self.session.commit()
        return True

    def mark_all_as_read(self, user_id: str) -> int:
        """Mark all notifications as read for a user.

        Args:
            user_id: UUID of the user.

        Returns:
            Number of notifications marked as read.
        """
        count = (
            self.session.query(AssetNotification)
            .filter(
                AssetNotification.user_id == user_id,
                AssetNotification.is_read.is_(False),
            )
            .update({"is_read": True})
        )
        self.session.commit()
        return count
