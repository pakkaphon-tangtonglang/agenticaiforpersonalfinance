"""Pydantic models for structured financial report output.

Each section of the report is a separate model, composed into
a single FinancialReport that covers all financial domains.
"""

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field


class ReportHighlight(BaseModel):
    """A single highlight or notable finding in the report.

    Attributes:
        highlight_type: Classification (achievement, warning, info).
        title: Short title in Thai.
        description: Detailed description in Thai.

    Example:
        >>> h = ReportHighlight(
        ...     highlight_type="achievement",
        ...     title="อัตราการออมดี", description="ออมได้ 25%"
        ... )
    """

    highlight_type: str
    title: str
    description: str


class MonthlyOverviewSection(BaseModel):
    """Monthly income vs expense overview.

    Attributes:
        total_income: Monthly income (annual / 12).
        total_expenses: Total expenses for the month.
        net_savings: Income minus expenses.
        savings_rate: Savings as fraction of income (0.0-1.0).

    Example:
        >>> s = MonthlyOverviewSection(
        ...     total_income=Decimal("50000"),
        ...     total_expenses=Decimal("35000"),
        ...     net_savings=Decimal("15000"),
        ...     savings_rate=Decimal("0.30"),
        ... )
    """

    total_income: Decimal = Field(default=Decimal("0"))
    total_expenses: Decimal = Field(default=Decimal("0"))
    net_savings: Decimal = Field(default=Decimal("0"))
    savings_rate: Decimal = Field(default=Decimal("0"))


class ExpenseBreakdownSection(BaseModel):
    """Expense breakdown by category.

    Attributes:
        total_amount: Total expense amount.
        transaction_count: Number of transactions.
        categories: List of category dicts with name, amount, percentage.

    Example:
        >>> s = ExpenseBreakdownSection(
        ...     total_amount=Decimal("35000"), transaction_count=42,
        ...     categories=[{"category": "food", "amount": "15000"}],
        ... )
    """

    total_amount: Decimal = Field(default=Decimal("0"))
    transaction_count: int = 0
    categories: list[dict[str, Any]] = Field(default_factory=list)


class InvestmentPortfolioSection(BaseModel):
    """Investment portfolio status summary.

    Attributes:
        total_value: Current market value of all holdings.
        total_cost: Total cost basis.
        total_gain_loss: Unrealized gain/loss.
        gain_loss_percentage: Gain/loss as percentage of cost.
        holding_count: Number of holdings.
        holdings: List of holding detail dicts.

    Example:
        >>> s = InvestmentPortfolioSection(
        ...     total_value=Decimal("500000"),
        ...     total_cost=Decimal("450000"),
        ...     total_gain_loss=Decimal("50000"),
        ...     gain_loss_percentage=Decimal("11.11"),
        ... )
    """

    total_value: Decimal = Field(default=Decimal("0"))
    total_cost: Decimal = Field(default=Decimal("0"))
    total_gain_loss: Decimal = Field(default=Decimal("0"))
    gain_loss_percentage: Decimal = Field(default=Decimal("0"))
    holding_count: int = 0
    holdings: list[dict[str, Any]] = Field(default_factory=list)


class GoalProgressSection(BaseModel):
    """Financial goal progress summary.

    Attributes:
        total_goals: Total number of goals.
        active_goals: Number of active (incomplete) goals.
        completed_goals: Number of completed goals.
        overall_percentage: Average completion percentage.
        goals: List of individual goal dicts.

    Example:
        >>> s = GoalProgressSection(total_goals=3, active_goals=2)
    """

    total_goals: int = 0
    active_goals: int = 0
    completed_goals: int = 0
    overall_percentage: Decimal = Field(default=Decimal("0"))
    goals: list[dict[str, Any]] = Field(default_factory=list)


class TaxStatusSection(BaseModel):
    """Tax filing status summary.

    Attributes:
        status: Filing status (e.g., "filed", "not_filed").
        gross_income: Annual gross income.
        total_deductions: Total deductions claimed.
        total_tax: Total tax calculated.
        effective_rate: Effective tax rate as percentage.
        tax_year: The tax year.

    Example:
        >>> s = TaxStatusSection(
        ...     status="filed", gross_income=Decimal("600000"),
        ...     total_deductions=Decimal("160000"),
        ...     total_tax=Decimal("29000"),
        ...     effective_rate=Decimal("4.83"), tax_year=2026,
        ... )
    """

    status: str = "not_filed"
    gross_income: Decimal = Field(default=Decimal("0"))
    total_deductions: Decimal = Field(default=Decimal("0"))
    total_tax: Decimal = Field(default=Decimal("0"))
    effective_rate: Decimal = Field(default=Decimal("0"))
    tax_year: int = 0


class FinancialReport(BaseModel):
    """Complete autonomous financial report.

    Combines all section models into a single report structure.

    Attributes:
        user_id: UUID of the user.
        generated_at: ISO datetime string of report generation.
        report_type: Type of report (monthly, annual).
        year: Report year.
        month: Report month.
        monthly_overview: Income vs expense overview.
        expense_breakdown: Expense by category.
        investment_portfolio: Portfolio status.
        goal_progress: Goal completion status.
        tax_status: Tax filing status.
        health_score: Financial health score 0-100.
        highlights: Notable findings.

    Example:
        >>> report = FinancialReport(
        ...     user_id="abc", generated_at="2026-03-10T00:00:00",
        ...     report_type="monthly", year=2026, month=3,
        ... )
    """

    user_id: str
    generated_at: str
    report_type: str = "monthly"
    year: int = 0
    month: int = 0
    monthly_overview: MonthlyOverviewSection = Field(default_factory=MonthlyOverviewSection)
    expense_breakdown: ExpenseBreakdownSection = Field(default_factory=ExpenseBreakdownSection)
    investment_portfolio: InvestmentPortfolioSection = Field(
        default_factory=InvestmentPortfolioSection
    )
    goal_progress: GoalProgressSection = Field(default_factory=GoalProgressSection)
    tax_status: TaxStatusSection = Field(default_factory=TaxStatusSection)
    health_score: int = Field(default=100, ge=0, le=100)
    highlights: list[ReportHighlight] = Field(default_factory=list)
