"""LangGraph tool wrappers for expense tracking.

Tools accept string inputs from LLM, parse them to proper types,
and persist records via the expense service layer. InjectedState
provides user_id and db_session_factory from the agent graph state.
"""

import calendar
from datetime import date, datetime
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from finance_ai.agents.session_helper import get_tool_session
from finance_ai.agents.tax_tools import parse_decimal_value
from finance_ai.tools.expense_calculator import (
    validate_expense_amount,
    validate_expense_category,
)
from finance_ai.tools.expense_constants import EXPENSE_CATEGORIES


def parse_date_value(value: str, field_name: str) -> date:
    """Parse a date string (YYYY-MM-DD) into a date object.

    Args:
        value: Date string in YYYY-MM-DD format.
        field_name: Name of the field for error messages.

    Returns:
        Parsed date object.

    Raises:
        ValueError: If the date string is invalid.

    Example:
        >>> parse_date_value("2026-02-24", "transaction_date")
        datetime.date(2026, 2, 24)
    """
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse {field_name} as date (YYYY-MM-DD). Received: '{value}'"
        ) from exc


def get_current_month_date_range() -> tuple[date, date]:
    """Return the first and last day of the current month.

    Returns:
        Tuple of (first_day, last_day) for the current month.

    Example:
        >>> start, end = get_current_month_date_range()
    """
    today = date.today()
    first_day = today.replace(day=1)
    last_day_num = calendar.monthrange(today.year, today.month)[1]
    last_day = today.replace(day=last_day_num)
    return first_day, last_day


@tool
def add_expense(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    amount: str,
    category: str,
    description: str = "",
    transaction_date: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """Record a new expense transaction.

    Use this tool when the user wants to add or record an expense.

    Args:
        amount: Expense amount in THB (e.g., "80").
        category: Category key (food, transport, entertainment,
            utilities, health, education, shopping, other).
        description: Optional description (e.g., "กาแฟ").
        transaction_date: Date in YYYY-MM-DD format. Defaults to today.
        user_id: UUID of the user (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict confirming the recorded expense with parsed values.
    """
    parsed_amount = parse_decimal_value(amount, "amount")
    validate_expense_amount(parsed_amount)
    normalized_category = validate_expense_category(category)
    parsed_date = (
        parse_date_value(transaction_date, "transaction_date") if transaction_date else date.today()
    )
    category_label = EXPENSE_CATEGORIES.get(normalized_category, normalized_category)

    _persist_expense(
        db_session_factory,
        user_id,
        parsed_amount,
        normalized_category,
        description,
        parsed_date,
    )

    return {
        "status": "recorded",
        "amount": str(parsed_amount),
        "category": normalized_category,
        "category_label": category_label,
        "description": description,
        "transaction_date": parsed_date.isoformat(),
    }


def _persist_expense(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    db_session_factory: Any,
    user_id: str,
    amount: Any,
    category: str,
    description: str,
    transaction_date: date,
) -> None:
    """Save expense to database via the service layer.

    Args:
        db_session_factory: Session factory or None.
        user_id: UUID of the user.
        amount: Expense amount (Decimal).
        category: Normalized category key.
        description: Expense description.
        transaction_date: Date of the expense.
    """
    from finance_ai.tools.expense_service import (  # noqa: PLC0415
        create_expense_transaction,
    )

    with get_tool_session(db_session_factory) as session:
        create_expense_transaction(
            session,
            user_id,
            amount,
            category,
            description,
            transaction_date,
        )


@tool
def summarize_monthly_expenses(
    year: str = "",
    month: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """Summarize expenses for a month with category breakdown.

    Use this tool when the user asks for an expense summary.

    Args:
        year: Year (e.g., "2026"). Defaults to current year.
        month: Month number (e.g., "2"). Defaults to current month.
        user_id: UUID of the user (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict with expense summary including category breakdown.
    """
    today = date.today()
    parsed_year = int(year) if year else today.year
    parsed_month = int(month) if month else today.month
    first_day = date(parsed_year, parsed_month, 1)
    last_day_num = calendar.monthrange(parsed_year, parsed_month)[1]
    last_day = date(parsed_year, parsed_month, last_day_num)

    summary = _fetch_expense_summary(
        db_session_factory,
        user_id,
        first_day,
        last_day,
    )

    return {
        "action": "summarize",
        "start_date": first_day.isoformat(),
        "end_date": last_day.isoformat(),
        "year": parsed_year,
        "month": parsed_month,
        **summary,
    }


def _fetch_expense_summary(
    db_session_factory: Any,
    user_id: str,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    """Query expense summary from database.

    Args:
        db_session_factory: Session factory or None.
        user_id: UUID of the user.
        start_date: Start of period (inclusive).
        end_date: End of period (inclusive).

    Returns:
        Dict with total_amount, category_breakdown, transaction_count.
    """
    from finance_ai.tools.expense_service import (  # noqa: PLC0415
        summarize_expenses_for_user,
    )

    with get_tool_session(db_session_factory) as session:
        result = summarize_expenses_for_user(
            session,
            user_id,
            start_date,
            end_date,
        )

    return {
        "total_amount": str(result.total_amount),
        "category_breakdown": [
            {
                "category": cs.category,
                "label": cs.category_label,
                "amount": str(cs.total_amount),
            }
            for cs in result.category_breakdown
        ],
        "transaction_count": result.transaction_count,
    }


@tool
def query_expenses_by_category(
    category: str,
    start_date: str = "",
    end_date: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """Query total expenses for a specific category in a date range.

    Use this tool when the user asks about spending in a specific category.

    Args:
        category: Category key (e.g., "food").
        start_date: Start date YYYY-MM-DD. Defaults to first of current month.
        end_date: End date YYYY-MM-DD. Defaults to today.
        user_id: UUID of the user (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict with category expense data from the database.
    """
    normalized_category = validate_expense_category(category)
    category_label = EXPENSE_CATEGORIES.get(normalized_category, normalized_category)
    if start_date:
        parsed_start = parse_date_value(start_date, "start_date")
    else:
        parsed_start = date.today().replace(day=1)
    if end_date:
        parsed_end = parse_date_value(end_date, "end_date")
    else:
        parsed_end = date.today()

    expenses = _fetch_category_expenses(
        db_session_factory,
        user_id,
        normalized_category,
        parsed_start,
        parsed_end,
    )

    return {
        "action": "query_by_category",
        "category": normalized_category,
        "category_label": category_label,
        "start_date": parsed_start.isoformat(),
        "end_date": parsed_end.isoformat(),
        **expenses,
    }


def _fetch_category_expenses(
    db_session_factory: Any,
    user_id: str,
    category: str,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    """Query expenses for a category from the database.

    Args:
        db_session_factory: Session factory or None.
        user_id: UUID of the user.
        category: Normalized expense category key.
        start_date: Start of period (inclusive).
        end_date: End of period (inclusive).

    Returns:
        Dict with total_amount and transaction_count for the category.
    """
    from decimal import Decimal  # noqa: PLC0415

    from finance_ai.tools.expense_service import (  # noqa: PLC0415
        get_expenses_by_category_and_date_range,
    )

    with get_tool_session(db_session_factory) as session:
        records = get_expenses_by_category_and_date_range(
            session,
            user_id,
            category,
            start_date,
            end_date,
        )

    total = sum((r.amount for r in records), Decimal("0"))
    return {
        "total_amount": str(total),
        "transaction_count": len(records),
    }
