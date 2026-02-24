"""Deduction database model for Personal Finance AI."""

from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finance_ai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from finance_ai.database.models.user import User


class Deduction(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Tax deduction record model for Thai tax deduction tracking."""

    __tablename__ = "deductions"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False, index=True
    )
    deduction_type: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(16, 2), nullable=False)
    maximum_allowed: Mapped[Decimal] = mapped_column(Numeric(16, 2), nullable=False)
    tax_year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    document_reference: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    user: Mapped["User"] = relationship(back_populates="deductions")
