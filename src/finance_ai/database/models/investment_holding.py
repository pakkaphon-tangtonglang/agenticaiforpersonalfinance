"""Investment holding database model for Personal Finance AI."""

from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Date, DateTime, ForeignKey, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finance_ai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from finance_ai.database.models.transaction import Transaction
    from finance_ai.database.models.user import User


class InvestmentHolding(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Investment holding model tracking stocks, mutual funds, and other assets."""

    __tablename__ = "investment_holdings"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False, index=True
    )
    asset_type: Mapped[str] = mapped_column(String(20), nullable=False)
    symbol: Mapped[str] = mapped_column(String(50), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    quantity: Mapped[Decimal] = mapped_column(Numeric(16, 4), nullable=False)
    average_cost_per_unit: Mapped[Decimal] = mapped_column(Numeric(16, 4), nullable=False)
    total_cost: Mapped[Decimal] = mapped_column(Numeric(16, 2), nullable=False)
    current_price_per_unit: Mapped[Optional[Decimal]] = mapped_column(Numeric(16, 4), nullable=True)
    current_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(16, 2), nullable=True)
    unrealized_gain_loss: Mapped[Optional[Decimal]] = mapped_column(Numeric(16, 2), nullable=True)
    purchase_date: Mapped[date] = mapped_column(Date, nullable=False)
    last_price_update: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    user: Mapped["User"] = relationship(back_populates="investment_holdings")
    transactions: Mapped[list["Transaction"]] = relationship(
        back_populates="holding", cascade="all, delete-orphan"
    )
