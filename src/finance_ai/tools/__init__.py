"""Tool implementations for agents."""

from finance_ai.tools.expense_calculator import (
    CategorySummary,
    ExpenseRecord,
    ExpenseSummaryResult,
    summarize_expenses,
)
from finance_ai.tools.expense_service import (
    create_expense_transaction,
    summarize_expenses_for_user,
)
from finance_ai.tools.investment_calculator import (
    HoldingRecord,
    HoldingSummary,
    PortfolioSummaryResult,
    summarize_portfolio,
)
from finance_ai.tools.investment_service import (
    add_investment_holding,
    get_portfolio_summary,
)
from finance_ai.tools.tax_calculator import (
    TaxBracketResult,
    TaxCalculationResult,
    calculate_tax,
)
from finance_ai.tools.tax_service import calculate_tax_for_user

__all__ = [
    "CategorySummary",
    "ExpenseRecord",
    "ExpenseSummaryResult",
    "HoldingRecord",
    "HoldingSummary",
    "PortfolioSummaryResult",
    "TaxBracketResult",
    "TaxCalculationResult",
    "add_investment_holding",
    "calculate_tax",
    "calculate_tax_for_user",
    "create_expense_transaction",
    "get_portfolio_summary",
    "summarize_expenses",
    "summarize_expenses_for_user",
    "summarize_portfolio",
]
