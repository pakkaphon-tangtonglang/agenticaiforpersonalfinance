"""Asset monitoring schedule database model."""

from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finance_ai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from finance_ai.database.models.user import User


class AssetSchedule(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Schedule for periodic asset data fetching.

    Attributes:
        user_id: UUID of the user who owns this schedule.
        symbol: Ticker symbol to monitor (e.g., "GC=F", "PTT.BK").
        description: Human-readable description (e.g., "ราคาทอง").
        cron_expression: Cron expression for schedule timing (e.g., "0 21 * * *").
        is_active: Whether this schedule is currently active.

    Example:
        >>> schedule = AssetSchedule(
        ...     user_id="abc-123",
        ...     symbol="GC=F",
        ...     description="ราคาทอง",
        ...     cron_expression="0 21 * * *",
        ... )
    """

    __tablename__ = "asset_schedules"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    description: Mapped[str] = mapped_column(String(255), nullable=False)
    cron_expression: Mapped[str] = mapped_column(String(100), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    max_runs: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, default=None)
    run_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    user: Mapped["User"] = relationship(back_populates="asset_schedules")
