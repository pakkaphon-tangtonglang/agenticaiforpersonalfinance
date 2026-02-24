"""Financial goal database model for Personal Finance AI."""

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, Date, ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finance_ai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from finance_ai.database.models.user import User


class FinancialGoal(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Financial goal model for tracking savings and investment targets."""

    __tablename__ = "financial_goals"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False, index=True
    )
    goal_type: Mapped[str] = mapped_column(String(50), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    target_amount: Mapped[Decimal] = mapped_column(Numeric(16, 2), nullable=False)
    current_amount: Mapped[Decimal] = mapped_column(
        Numeric(16, 2), default=Decimal("0"), nullable=False
    )
    target_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    priority: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    user: Mapped["User"] = relationship(back_populates="financial_goals")
