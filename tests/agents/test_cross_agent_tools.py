"""Tests for cross-agent tool wrappers."""

from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy.orm import Session

from finance_ai.agents.cross_agent_tools import (
    EXPENSE_CROSS_TOOLS,
    ASSET_MONITORING_CROSS_TOOLS,
    PLANNING_CROSS_TOOLS,
    TAX_CROSS_TOOLS,
    get_expense_summary_cross,
    get_goals_summary_cross,
    get_income_summary_cross,
    update_savings_goal_cross,
)
from finance_ai.database.crud.income_crud import IncomeCRUD
from finance_ai.database.crud.transaction_crud import TransactionCRUD
from finance_ai.database.models.user import User
from finance_ai.tools.planning_service import create_goal, update_goal_progress


class TestToolLists:
    """Tests for pre-grouped tool lists."""

    def test_tax_cross_tools_has_expense(self) -> None:
        """Tax agent should have expense cross-tool."""
        tool_names = [t.name for t in TAX_CROSS_TOOLS]
        assert "get_expense_summary_cross" in tool_names
        assert "get_portfolio_summary_cross" not in tool_names

    def test_planning_cross_tools_has_expense_and_income(self) -> None:
        """Planning agent should have expense and income tools."""
        tool_names = [t.name for t in PLANNING_CROSS_TOOLS]
        assert "get_expense_summary_cross" in tool_names
        assert "get_income_summary_cross" in tool_names
        assert "get_portfolio_summary_cross" not in tool_names

    def test_expense_cross_tools_has_goals(self) -> None:
        """Expense agent should have goals cross-tool."""
        tool_names = [t.name for t in EXPENSE_CROSS_TOOLS]
        assert "get_goals_summary_cross" in tool_names

    def test_asset_monitoring_cross_tools_is_empty(self) -> None:
        """Asset monitoring agent has no cross-tools."""
        assert ASSET_MONITORING_CROSS_TOOLS == []


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


class TestUpdateSavingsGoalCross:
    """Tests for update_savings_goal_cross (set a total or add a deposit)."""

    @pytest.fixture(name="goal_id")
    def fixture_goal_id(self, test_session: Session, sample_user: User) -> str:
        """Create an emergency-fund goal with 96,000 already saved."""
        goal = create_goal(
            test_session, sample_user.id, "emergency_fund", "กองทุนสำรองฉุกเฉิน", Decimal("180000")
        )
        update_goal_progress(test_session, goal.id, Decimal("96000"))
        test_session.commit()
        return str(goal.id)

    def _invoke(self, user: User, factory: Callable[[], Session], **amounts: str) -> Any:
        """Call the tool for the emergency fund with the given amount fields."""
        payload: dict[str, Any] = {
            "search_term": "ฉุกเฉิน",
            "user_id": user.id,
            "db_session_factory": factory,
        }
        payload.update(amounts)
        return update_savings_goal_cross.invoke(payload)

    def test_deposit_adds_to_saved_amount(
        self, goal_id: str, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """ "เพิ่มเงิน 10,000" keeps the 96,000 already saved."""
        result = self._invoke(sample_user, db_session_factory, added_amount="10000")
        assert result["status"] == "updated" and result["goal_id"] == goal_id
        assert Decimal(result["current_amount"]) == Decimal("106000")

    def test_total_sets_saved_amount(
        self, goal_id: str, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """ "มีเงินออม 120,000" sets the total."""
        result = self._invoke(sample_user, db_session_factory, current_amount="120,000")
        assert result["goal_id"] == goal_id
        assert Decimal(result["current_amount"]) == Decimal("120000")

    def test_neither_amount_is_an_error(
        self, goal_id: str, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Without an amount the tool reports an error instead of guessing."""
        assert goal_id
        result = self._invoke(sample_user, db_session_factory)
        assert result["status"] == "error" and "exactly one" in result["message"]

    def test_no_goals(self, sample_user: User, db_session_factory: Callable[[], Session]) -> None:
        """A user without goals gets no_goals."""
        result = self._invoke(sample_user, db_session_factory, added_amount="10000")
        assert result["status"] == "no_goals"
