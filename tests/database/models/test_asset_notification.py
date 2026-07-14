"""Tests for AssetNotification model."""

from sqlalchemy.orm import Session

from finance_ai.database.models.asset_notification import AssetNotification
from finance_ai.database.models.user import User


class TestAssetNotification:
    """Tests for AssetNotification model."""

    def test_create_notification(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Create a notification with required fields."""
        notification = AssetNotification(
            user_id=sample_user.id,
            symbol="GC=F",
            content="Gold: 2,350 USD",
        )
        test_session.add(notification)
        test_session.commit()
        test_session.refresh(notification)
        assert notification.id is not None
        assert notification.is_read is False
        assert notification.schedule_id is None

    def test_is_read_default(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """is_read defaults to False."""
        notification = AssetNotification(
            user_id=sample_user.id,
            symbol="PTT.BK",
            content="PTT: 35.50",
        )
        test_session.add(notification)
        test_session.commit()
        test_session.refresh(notification)
        assert notification.is_read is False
