"""Asset monitoring notification database model."""

from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finance_ai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from finance_ai.database.models.asset_schedule import AssetSchedule
    from finance_ai.database.models.user import User


class AssetNotification(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Notification from a scheduled asset data fetch.

    Attributes:
        user_id: UUID of the user who owns this notification.
        schedule_id: UUID of the AssetSchedule that triggered this notification.
        symbol: Ticker symbol that was fetched.
        content: The fetched data summary as text.
        is_read: Whether the user has read this notification.

    Example:
        >>> notification = AssetNotification(
        ...     user_id="abc-123",
        ...     schedule_id="sched-456",
        ...     symbol="GC=F",
        ...     content="ราคาทอง: 2,350 USD",
        ... )
    """

    __tablename__ = "asset_notifications"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False, index=True
    )
    schedule_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("asset_schedules.id"), nullable=True
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    is_read: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    user: Mapped["User"] = relationship(back_populates="asset_notifications")
    schedule: Mapped[Optional["AssetSchedule"]] = relationship()
