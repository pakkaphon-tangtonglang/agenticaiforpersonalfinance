"""WatchedAsset database model for tracking user asset watchlists."""

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finance_ai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from finance_ai.database.models.user import User


class WatchedAsset(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """User asset watchlist — symbols the user wants to track.

    Stores only symbol and display name; no quantities or prices.

    Example:
        >>> asset = WatchedAsset(user_id=uid, symbol="PTT.BK", name="PTT")
    """

    __tablename__ = "watched_assets"
    __table_args__ = (UniqueConstraint("user_id", "symbol", name="uq_watched_asset_user_symbol"),)

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String(30), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False, default="")

    user: Mapped["User"] = relationship(back_populates="watched_assets")
