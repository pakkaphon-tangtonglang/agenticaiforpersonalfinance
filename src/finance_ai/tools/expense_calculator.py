"""Pure expense calculation functions for aggregation and summarization.

All functions accept typed values directly with no database dependency,
making them easy to test and reuse across different contexts.
"""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field

from finance_ai.tools.expense_constants import (
    EXPENSE_CATEGORIES,
    MAX_SINGLE_EXPENSE_AMOUNT,
    MIN_EXPENSE_AMOUNT,
    VALID_EXPENSE_CATEGORIES,
)


class ExpenseRecord(BaseModel):
    """A single expense record for calculation purposes.

    Attributes:
        amount: Expense amount in THB.
        category: Expense category key.
        description: Optional description of the expense.
        transaction_date: Date of the expense.
    """

    amount: Decimal
    category: str
    description: str = ""
    transaction_date: date


class CategorySummary(BaseModel):
    """Summary of expenses for a single category.

    Attributes:
        category: Category key.
        category_label: Thai display label.
        total_amount: Sum of expenses in this category.
        transaction_count: Number of transactions.
        percentage_of_total: Percentage of total spending.
    """

    category: str
    category_label: str
    total_amount: Decimal
    transaction_count: int
    percentage_of_total: Decimal = Field(default=Decimal("0.0000"))


class ExpenseSummaryResult(BaseModel):
    """Complete expense summary for a date range.

    Attributes:
        start_date: Start of the summary period.
        end_date: End of the summary period.
        total_amount: Grand total of all expenses.
        category_breakdown: Per-category summaries.
        transaction_count: Total number of transactions.
    """

    start_date: date
    end_date: date
    total_amount: Decimal
    category_breakdown: list[CategorySummary]
    transaction_count: int


def validate_expense_amount(amount: Decimal) -> None:
    """Validate that expense amount is within the allowed range.

    Args:
        amount: Expense amount to validate.

    Raises:
        ValueError: If amount is below minimum or above maximum.

    Example:
        >>> validate_expense_amount(Decimal("100"))
    """
    if amount < MIN_EXPENSE_AMOUNT:
        raise ValueError(
            f"Expense amount must be at least {MIN_EXPENSE_AMOUNT} THB. " f"Received: {amount} THB"
        )
    if amount > MAX_SINGLE_EXPENSE_AMOUNT:
        raise ValueError(
            f"Expense amount exceeds maximum of {MAX_SINGLE_EXPENSE_AMOUNT} THB. "
            f"Received: {amount} THB"
        )


def validate_expense_category(category: str) -> str:
    """Validate and normalize an expense category key.

    Args:
        category: Category key to validate.

    Returns:
        Normalized (lowercase) category key.

    Raises:
        ValueError: If category is not recognized.

    Example:
        >>> validate_expense_category("Food")
        'food'
    """
    normalized = category.strip().lower()
    if normalized not in VALID_EXPENSE_CATEGORIES:
        raise ValueError(
            f"Unknown expense category: '{category}'. "
            f"Valid categories: {VALID_EXPENSE_CATEGORIES}"
        )
    return normalized


def calculate_category_total(
    expenses: list[ExpenseRecord],
    category: str,
) -> Decimal:
    """Sum expense amounts for a single category.

    Args:
        expenses: List of expense records to aggregate.
        category: Category key to filter by.

    Returns:
        Total amount for the specified category.

    Example:
        >>> calculate_category_total([], "food")
        Decimal('0')
    """
    return sum(
        (expense.amount for expense in expenses if expense.category == category),
        Decimal("0"),
    )


def calculate_category_percentage(
    category_total: Decimal,
    grand_total: Decimal,
) -> Decimal:
    """Calculate a category's percentage of total spending.

    Args:
        category_total: Total for one category.
        grand_total: Grand total of all expenses.

    Returns:
        Percentage as decimal with 4 decimal places (e.g., 0.2500 = 25%).

    Example:
        >>> calculate_category_percentage(Decimal("250"), Decimal("1000"))
        Decimal('0.2500')
    """
    if grand_total == Decimal("0"):
        return Decimal("0.0000")
    return (category_total / grand_total).quantize(Decimal("0.0001"))


def count_expenses_by_category(
    expenses: list[ExpenseRecord],
    category: str,
) -> int:
    """Count number of expenses in a category.

    Args:
        expenses: List of expense records.
        category: Category key to count.

    Returns:
        Number of expenses in the specified category.

    Example:
        >>> count_expenses_by_category([], "food")
        0
    """
    return sum(1 for expense in expenses if expense.category == category)


def build_category_summary(
    expenses: list[ExpenseRecord],
    category: str,
    grand_total: Decimal,
) -> CategorySummary:
    """Build a CategorySummary for one category.

    Args:
        expenses: All expense records (will filter by category).
        category: Category key to summarize.
        grand_total: Grand total for percentage calculation.

    Returns:
        CategorySummary with totals and percentage.

    Example:
        >>> build_category_summary([], "food", Decimal("1000"))
    """
    total = calculate_category_total(expenses, category)
    count = count_expenses_by_category(expenses, category)
    percentage = calculate_category_percentage(total, grand_total)
    label = EXPENSE_CATEGORIES.get(category, category)
    return CategorySummary(
        category=category,
        category_label=label,
        total_amount=total,
        transaction_count=count,
        percentage_of_total=percentage,
    )


def calculate_grand_total(expenses: list[ExpenseRecord]) -> Decimal:
    """Sum all expense amounts.

    Args:
        expenses: List of expense records.

    Returns:
        Grand total of all expenses.

    Example:
        >>> calculate_grand_total([])
        Decimal('0')
    """
    return sum((expense.amount for expense in expenses), Decimal("0"))


def get_active_categories(expenses: list[ExpenseRecord]) -> list[str]:
    """Return sorted list of categories that have at least one expense.

    Args:
        expenses: List of expense records.

    Returns:
        Sorted list of category keys with expenses.

    Example:
        >>> get_active_categories([])
        []
    """
    categories = {expense.category for expense in expenses}
    return sorted(categories)


def summarize_expenses(
    expenses: list[ExpenseRecord],
    start_date: date,
    end_date: date,
) -> ExpenseSummaryResult:
    """Build complete expense summary with category breakdown.

    Args:
        expenses: List of expense records for the period.
        start_date: Start of the summary period.
        end_date: End of the summary period.

    Returns:
        ExpenseSummaryResult with totals and per-category breakdown.

    Example:
        >>> from datetime import date
        >>> summarize_expenses([], date(2026, 1, 1), date(2026, 1, 31))
    """
    grand_total = calculate_grand_total(expenses)
    active_categories = get_active_categories(expenses)
    breakdown = [
        build_category_summary(expenses, category, grand_total) for category in active_categories
    ]
    return ExpenseSummaryResult(
        start_date=start_date,
        end_date=end_date,
        total_amount=grand_total,
        category_breakdown=breakdown,
        transaction_count=len(expenses),
    )
