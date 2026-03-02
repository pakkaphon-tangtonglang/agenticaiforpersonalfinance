"""Tests for cross-agent tool wrappers."""

from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy.orm import Session

from finance_ai.agents.cross_agent_tools import (
    EXPENSE_CROSS_TOOLS,
    INVESTMENT_CROSS_TOOLS,
    PLANNING_CROSS_TOOLS,
    TAX_CROSS_TOOLS,
    get_expense_summary_cross,
    get_goals_summary_cross,
    get_income_summary_cross,
    get_portfolio_summary_cross,
    get_tax_summary_cross,
)
from finance_ai.database.crud.income_crud import IncomeCRUD
from finance_ai.database.crud.transaction_crud import TransactionCRUD
from finance_ai.database.models.user import User


class TestToolLists:
    """Tests for pre-grouped tool lists."""

    def test_tax_cross_tools_has_portfolio_and_expense(self) -> None:
        """Tax agent should have portfolio and expense cross-tools."""
        tool_names = [t.name for t in TAX_CROSS_TOOLS]
        assert "get_portfolio_summary_cross" in tool_names
        assert "get_expense_summary_cross" in tool_names

    def test_planning_cross_tools_has_three_tools(self) -> None:
        """Planning agent should have expense, income, portfolio tools."""
        tool_names = [t.name for t in PLANNING_CROSS_TOOLS]
        assert "get_expense_summary_cross" in tool_names
        assert "get_income_summary_cross" in tool_names
        assert "get_portfolio_summary_cross" in tool_names

    def test_expense_cross_tools_has_goals(self) -> None:
        """Expense agent should have goals cross-tool."""
        tool_names = [t.name for t in EXPENSE_CROSS_TOOLS]
        assert "get_goals_summary_cross" in tool_names

    def test_investment_cross_tools_has_tax(self) -> None:
        """Investment agent should have tax cross-tool."""
        tool_names = [t.name for t in INVESTMENT_CROSS_TOOLS]
        assert "get_tax_summary_cross" in tool_names


class TestGetExpenseSummaryCross:
    """Tests for get_expense_summary_cross tool."""

    def test_returns_empty_summary(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """No expenses returns zero total."""
        result = get_expense_summary_cross.invoke(
            {
                "year": "2026",
                "month": "3",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            },
        )
        assert result["domain"] == "expense"
        assert result["total_amount"] == "0"

    def test_defaults_to_current_month(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Empty year/month defaults to current period."""
        result = get_expense_summary_cross.invoke(
            {
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            },
        )
        assert result["domain"] == "expense"

    def test_returns_expense_data(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """With expenses, returns correct totals."""
        session = db_session_factory()
        crud = TransactionCRUD()
        crud.create(
            session,
            user_id=sample_user.id,
            transaction_type="expense",
            category="food",
            amount=Decimal("200.00"),
            transaction_date=date(2026, 3, 10),
        )
        session.commit()
        session.close()

        result = get_expense_summary_cross.invoke(
            {
                "year": "2026",
                "month": "3",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            },
        )
        assert result["total_amount"] == "200.00"
        assert result["transaction_count"] == 1


class TestGetPortfolioSummaryCross:
    """Tests for get_portfolio_summary_cross tool."""

    def test_returns_empty_portfolio(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """No holdings returns zero values."""
        result = get_portfolio_summary_cross.invoke(
            {
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            },
        )
        assert result["domain"] == "investment"
        assert result["holding_count"] == 0


class TestGetGoalsSummaryCross:
    """Tests for get_goals_summary_cross tool."""

    def test_returns_empty_goals(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """No goals returns zero counts."""
        result = get_goals_summary_cross.invoke(
            {
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            },
        )
        assert result["domain"] == "planning"
        assert result["total_goals"] == 0


class TestGetIncomeSummaryCross:
    """Tests for get_income_summary_cross tool."""

    def test_returns_empty_income(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """No income returns zero total."""
        result = get_income_summary_cross.invoke(
            {
                "tax_year": "2025",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            },
        )
        assert result["domain"] == "income"
        assert result["total_income"] == "0"

    def test_returns_income_data(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """With income, returns correct totals."""
        session = db_session_factory()
        crud = IncomeCRUD()
        crud.create(
            session,
            user_id=sample_user.id,
            income_type="salary",
            description="เงินเดือน",
            amount=Decimal("60000.00"),
            tax_year=2025,
            pay_period="monthly",
        )
        session.commit()
        session.close()

        result = get_income_summary_cross.invoke(
            {
                "tax_year": "2025",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            },
        )
        assert result["total_income"] == "60000.00"
        assert result["source_count"] == 1


class TestGetTaxSummaryCross:
    """Tests for get_tax_summary_cross tool."""

    def test_returns_not_found(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """No filing returns not_found status."""
        result = get_tax_summary_cross.invoke(
            {
                "tax_year": "2025",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            },
        )
        assert result["domain"] == "tax"
        assert result["status"] == "not_found"
