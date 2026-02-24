"""User database model for Personal Finance AI."""

from datetime import date
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, Date, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finance_ai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from finance_ai.database.models.deduction import Deduction
    from finance_ai.database.models.financial_goal import FinancialGoal
    from finance_ai.database.models.income import Income
    from finance_ai.database.models.investment_holding import InvestmentHolding
    from finance_ai.database.models.tax_filing import TaxFiling
    from finance_ai.database.models.transaction import Transaction


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """User account model with personal and financial information."""

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    tax_id: Mapped[Optional[str]] = mapped_column(String(13), unique=True, nullable=True)
    date_of_birth: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    marital_status: Mapped[str] = mapped_column(String(20), default="single", nullable=False)
    number_of_children: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    number_of_parents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    incomes: Mapped[list["Income"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    deductions: Mapped[list["Deduction"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    investment_holdings: Mapped[list["InvestmentHolding"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    transactions: Mapped[list["Transaction"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    tax_filings: Mapped[list["TaxFiling"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    financial_goals: Mapped[list["FinancialGoal"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
