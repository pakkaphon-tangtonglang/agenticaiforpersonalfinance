"""RecurringEntry model: income or expense that repeats on a day each month."""

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, Date, ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finance_ai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from finance_ai.database.models.user import User


class RecurringEntry(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A monthly income or expense (e.g. rent received on the 3rd).

    Each due month becomes a normal transaction; last_posted_on remembers
    the latest one so posting twice never duplicates.

    Example:
        >>> RecurringEntry(user_id=uid, transaction_type="income", category="rental",
        ...     amount=Decimal("8000"), day_of_month=3, start_date=date(2026, 11, 3))
    """

    __tablename__ = "recurring_entries"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False, index=True
    )
    transaction_type: Mapped[str] = mapped_column(String(20), nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    amount: Mapped[Decimal] = mapped_column(Numeric(16, 2), nullable=False)
    day_of_month: Mapped[int] = mapped_column(Integer, nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    last_posted_on: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    user: Mapped["User"] = relationship(back_populates="recurring_entries")
