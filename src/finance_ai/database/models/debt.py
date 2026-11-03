"""Debt model: money the user still owes (a friend, a bank, a credit card...)."""

from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, ForeignKey, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finance_ai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from finance_ai.database.models.user import User


class Debt(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One debt and how much of it is left.

    Owing money is not an expense; paying it back is. Payments are recorded
    as "debt" expenses and lower remaining_balance until it reaches zero.

    Example:
        >>> Debt(user_id=uid, creditor="เพื่อน", debt_type="personal",
        ...     original_amount=Decimal("2000"), remaining_balance=Decimal("2000"))
    """

    __tablename__ = "debts"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False, index=True
    )
    creditor: Mapped[str] = mapped_column(String(255), nullable=False)
    debt_type: Mapped[str] = mapped_column(String(30), nullable=False)
    original_amount: Mapped[Decimal] = mapped_column(Numeric(16, 2), nullable=False)
    remaining_balance: Mapped[Decimal] = mapped_column(Numeric(16, 2), nullable=False)
    interest_rate_percent: Mapped[Optional[Decimal]] = mapped_column(Numeric(7, 4), nullable=True)
    monthly_payment: Mapped[Optional[Decimal]] = mapped_column(Numeric(16, 2), nullable=True)
    is_paid_off: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    user: Mapped["User"] = relationship(back_populates="debts")
