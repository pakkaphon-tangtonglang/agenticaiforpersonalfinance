"""Transaction database model for Personal Finance AI."""

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Date, ForeignKey, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finance_ai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from finance_ai.database.models.investment_holding import InvestmentHolding
    from finance_ai.database.models.user import User


class Transaction(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Financial transaction model for tracking buys, sells, expenses, and income."""

    __tablename__ = "transactions"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False, index=True
    )
    holding_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("investment_holdings.id"), nullable=True, index=True
    )
    transaction_type: Mapped[str] = mapped_column(String(20), nullable=False)
    category: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(16, 2), nullable=False)
    quantity: Mapped[Optional[Decimal]] = mapped_column(Numeric(16, 4), nullable=True)
    price_per_unit: Mapped[Optional[Decimal]] = mapped_column(Numeric(16, 4), nullable=True)
    transaction_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    reference_number: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    user: Mapped["User"] = relationship(back_populates="transactions")
    holding: Mapped[Optional["InvestmentHolding"]] = relationship(back_populates="transactions")
