"""Tax filing database model for Personal Finance AI."""

from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finance_ai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from finance_ai.database.models.user import User


class TaxFiling(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Tax filing record model for annual Thai tax calculations and filings."""

    __tablename__ = "tax_filings"
    __table_args__ = (UniqueConstraint("user_id", "tax_year", name="uq_user_tax_year"),)

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False, index=True
    )
    tax_year: Mapped[int] = mapped_column(Integer, nullable=False)
    gross_income: Mapped[Decimal] = mapped_column(Numeric(16, 2), nullable=False)
    total_deductions: Mapped[Decimal] = mapped_column(Numeric(16, 2), nullable=False)
    net_income: Mapped[Decimal] = mapped_column(Numeric(16, 2), nullable=False)
    total_tax: Mapped[Decimal] = mapped_column(Numeric(16, 2), nullable=False)
    effective_tax_rate: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=False)
    withholding_tax_paid: Mapped[Decimal] = mapped_column(
        Numeric(16, 2), default=Decimal("0"), nullable=False
    )
    tax_due_or_refund: Mapped[Decimal] = mapped_column(Numeric(16, 2), nullable=False)
    filing_status: Mapped[str] = mapped_column(String(20), default="draft", nullable=False)
    tax_breakdown_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    user: Mapped["User"] = relationship(back_populates="tax_filings")
