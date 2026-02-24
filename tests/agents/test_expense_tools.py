"""Tests for LangGraph expense tool wrappers."""

from collections.abc import Callable
from datetime import date
from decimal import Decimal
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from finance_ai.agents.expense_tools import (
    add_expense,
    get_current_month_date_range,
    parse_date_value,
    query_expenses_by_category,
    summarize_monthly_expenses,
)
from finance_ai.database.crud.transaction_crud import TransactionCRUD
from finance_ai.database.models.user import User
from finance_ai.tools.expense_constants import EXPENSE_TRANSACTION_TYPE


class TestParseDateValue:
    """Tests for parse_date_value helper."""

    def test_valid_date(self) -> None:
        """Parses a valid date string."""
        result = parse_date_value("2026-02-24", "transaction_date")
        assert result == date(2026, 2, 24)

    def test_strips_whitespace(self) -> None:
        """Strips whitespace from date string."""
        result = parse_date_value("  2026-02-24  ", "transaction_date")
        assert result == date(2026, 2, 24)

    def test_invalid_format_raises(self) -> None:
        """Rejects invalid date format."""
        with pytest.raises(ValueError, match="Cannot parse transaction_date"):
            parse_date_value("24/02/2026", "transaction_date")

    def test_empty_string_raises(self) -> None:
        """Rejects empty string."""
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_date_value("", "date_field")

    def test_nonsense_string_raises(self) -> None:
        """Rejects non-date string."""
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_date_value("yesterday", "date_field")


class TestGetCurrentMonthDateRange:
    """Tests for get_current_month_date_range helper."""

    @patch("finance_ai.agents.expense_tools.date")
    def test_returns_first_and_last_day(self, mock_date: object) -> None:
        """Returns correct first and last day of the month."""
        mock_date.today.return_value = date(2026, 2, 15)  # type: ignore[attr-defined]
        mock_date.side_effect = lambda *args, **kw: date(*args, **kw)  # type: ignore[attr-defined]
        start, end = get_current_month_date_range()
        assert start == date(2026, 2, 1)
        assert end == date(2026, 2, 28)

    def test_returns_date_objects(self) -> None:
        """Returns date objects (not datetime)."""
        start, end = get_current_month_date_range()
        assert isinstance(start, date)
        assert isinstance(end, date)

    def test_start_is_day_one(self) -> None:
        """Start date is always the first of the month."""
        start, _ = get_current_month_date_range()
        assert start.day == 1


class TestAddExpense:
    """Tests for the add_expense LangGraph tool."""

    def test_valid_expense_persists_to_database(
        self,
        test_session: Session,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Records a valid expense and persists it to the database."""
        result = add_expense.invoke(
            {
                "amount": "80",
                "category": "food",
                "description": "กาแฟ",
                "transaction_date": "2026-02-24",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert isinstance(result, dict)
        assert result["status"] == "recorded"
        assert result["amount"] == "80"
        assert result["category"] == "food"
        assert result["category_label"] == "อาหาร"
        assert result["description"] == "กาแฟ"
        assert result["transaction_date"] == "2026-02-24"

        crud = TransactionCRUD()
        transactions = crud.get_by_user_type_and_date_range(
            test_session,
            sample_user.id,
            EXPENSE_TRANSACTION_TYPE,
            date(2026, 2, 1),
            date(2026, 2, 28),
        )
        assert len(transactions) == 1
        assert transactions[0].amount == Decimal("80")
        assert transactions[0].category == "food"

    def test_default_date_is_today(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Uses today's date when not specified."""
        result = add_expense.invoke(
            {
                "amount": "100",
                "category": "transport",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["transaction_date"] == date.today().isoformat()

    def test_normalizes_category(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Normalizes uppercase category."""
        result = add_expense.invoke(
            {
                "amount": "50",
                "category": "Food",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["category"] == "food"

    def test_invalid_amount_raises(self) -> None:
        """Rejects negative amount."""
        with pytest.raises(ValueError, match="must be at least"):
            add_expense.invoke(
                {
                    "amount": "-10",
                    "category": "food",
                }
            )

    def test_invalid_category_raises(self) -> None:
        """Rejects unknown category."""
        with pytest.raises(ValueError, match="Unknown expense category"):
            add_expense.invoke(
                {
                    "amount": "100",
                    "category": "gambling",
                }
            )

    def test_non_numeric_amount_raises(self) -> None:
        """Rejects non-numeric amount."""
        with pytest.raises(ValueError, match="Cannot parse amount"):
            add_expense.invoke(
                {
                    "amount": "abc",
                    "category": "food",
                }
            )

    def test_empty_description_default(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Description defaults to empty string."""
        result = add_expense.invoke(
            {
                "amount": "50",
                "category": "food",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["description"] == ""


class TestSummarizeMonthlyExpenses:
    """Tests for the summarize_monthly_expenses LangGraph tool."""

    def test_with_explicit_year_month(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Returns correct date range and summary for specified year/month."""
        result = summarize_monthly_expenses.invoke(
            {
                "year": "2026",
                "month": "2",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["action"] == "summarize"
        assert result["start_date"] == "2026-02-01"
        assert result["end_date"] == "2026-02-28"
        assert result["year"] == 2026
        assert result["month"] == 2
        assert "total_amount" in result
        assert "transaction_count" in result

    def test_defaults_to_current_month(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Uses current year and month when not specified."""
        result = summarize_monthly_expenses.invoke(
            {
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        today = date.today()
        assert result["year"] == today.year
        assert result["month"] == today.month

    def test_month_with_31_days(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Handles months with 31 days correctly."""
        result = summarize_monthly_expenses.invoke(
            {
                "year": "2026",
                "month": "1",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["end_date"] == "2026-01-31"

    def test_returns_summary_with_expenses(
        self,
        test_session: Session,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Returns actual expense data after adding expenses."""
        add_expense.invoke(
            {
                "amount": "100",
                "category": "food",
                "transaction_date": "2026-02-15",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        result = summarize_monthly_expenses.invoke(
            {
                "year": "2026",
                "month": "2",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert Decimal(result["total_amount"]) == Decimal("100")
        assert result["transaction_count"] == 1


class TestQueryExpensesByCategory:
    """Tests for the query_expenses_by_category LangGraph tool."""

    def test_with_explicit_dates(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Returns correct parameters with explicit date range."""
        result = query_expenses_by_category.invoke(
            {
                "category": "food",
                "start_date": "2026-02-01",
                "end_date": "2026-02-28",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["action"] == "query_by_category"
        assert result["category"] == "food"
        assert result["category_label"] == "อาหาร"
        assert result["start_date"] == "2026-02-01"
        assert result["end_date"] == "2026-02-28"
        assert "total_amount" in result
        assert "transaction_count" in result

    def test_default_dates(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Uses first of month and today when dates not specified."""
        result = query_expenses_by_category.invoke(
            {
                "category": "transport",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        today = date.today()
        assert result["start_date"] == today.replace(day=1).isoformat()
        assert result["end_date"] == today.isoformat()

    def test_invalid_category_raises(self) -> None:
        """Rejects invalid category."""
        with pytest.raises(ValueError, match="Unknown expense category"):
            query_expenses_by_category.invoke(
                {
                    "category": "gambling",
                }
            )

    def test_normalizes_category(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Normalizes category to lowercase."""
        result = query_expenses_by_category.invoke(
            {
                "category": "Transport",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["category"] == "transport"

    def test_returns_category_total(
        self,
        test_session: Session,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Returns actual expense total for a category."""
        add_expense.invoke(
            {
                "amount": "200",
                "category": "food",
                "transaction_date": "2026-02-10",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        add_expense.invoke(
            {
                "amount": "150",
                "category": "food",
                "transaction_date": "2026-02-20",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        result = query_expenses_by_category.invoke(
            {
                "category": "food",
                "start_date": "2026-02-01",
                "end_date": "2026-02-28",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert Decimal(result["total_amount"]) == Decimal("350")
        assert result["transaction_count"] == 2
