"""Income database model for Personal Finance AI."""

from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finance_ai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from finance_ai.database.models.user import User


class Income(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Income record model tracking user earnings by type and tax year."""

    __tablename__ = "incomes"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False, index=True
    )
    income_type: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(16, 2), nullable=False)
    tax_year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    pay_period: Mapped[str] = mapped_column(String(20), nullable=False)
    employer_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    withholding_tax: Mapped[Decimal] = mapped_column(
        Numeric(16, 2), default=Decimal("0"), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="incomes")
