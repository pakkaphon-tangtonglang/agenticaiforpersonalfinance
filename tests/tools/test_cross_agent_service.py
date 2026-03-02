"""Tests for cross-agent service layer."""

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session

from finance_ai.database.crud.income_crud import IncomeCRUD
from finance_ai.database.crud.tax_filing_crud import TaxFilingCRUD
from finance_ai.database.crud.transaction_crud import TransactionCRUD
from finance_ai.database.models.user import User
from finance_ai.tools.cross_agent_service import (
    get_expense_summary,
    get_goals_summary,
    get_income_summary,
    get_portfolio_summary,
    get_tax_filing_summary,
)


class TestGetExpenseSummary:
    """Tests for get_expense_summary."""

    def test_returns_empty_summary(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """No expenses returns zero total."""
        result = get_expense_summary(test_session, sample_user.id, 2026, 3)
        assert result["domain"] == "expense"
        assert result["total_amount"] == "0"
        assert result["transaction_count"] == 0

    def test_returns_expense_data(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """With expenses, returns totals and breakdown."""
        crud = TransactionCRUD()
        crud.create(
            test_session,
            user_id=sample_user.id,
            transaction_type="expense",
            category="food",
            amount=Decimal("500.00"),
            transaction_date=date(2026, 3, 15),
        )
        result = get_expense_summary(test_session, sample_user.id, 2026, 3)
        assert result["total_amount"] == "500.00"
        assert result["transaction_count"] == 1
        assert len(result["category_breakdown"]) == 1

    def test_date_range_boundaries(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Expenses outside the month are excluded."""
        crud = TransactionCRUD()
        crud.create(
            test_session,
            user_id=sample_user.id,
            transaction_type="expense",
            category="food",
            amount=Decimal("100.00"),
            transaction_date=date(2026, 2, 28),
        )
        result = get_expense_summary(test_session, sample_user.id, 2026, 3)
        assert result["total_amount"] == "0"


class TestGetPortfolioSummary:
    """Tests for get_portfolio_summary."""

    def test_returns_empty_portfolio(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """No holdings returns zero values."""
        result = get_portfolio_summary(test_session, sample_user.id)
        assert result["domain"] == "investment"
        assert result["holding_count"] == 0
        assert result["total_cost"] == "0"


class TestGetGoalsSummary:
    """Tests for get_goals_summary."""

    def test_returns_empty_goals(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """No goals returns zero counts."""
        result = get_goals_summary(test_session, sample_user.id)
        assert result["domain"] == "planning"
        assert result["total_goals"] == 0
        assert result["active_goals"] == 0


class TestGetIncomeSummary:
    """Tests for get_income_summary."""

    def test_returns_empty_income(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """No income records returns zero total."""
        result = get_income_summary(test_session, sample_user.id, 2025)
        assert result["domain"] == "income"
        assert result["total_income"] == "0"
        assert result["source_count"] == 0

    def test_returns_income_data(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """With income records, returns totals and sources."""
        crud = IncomeCRUD()
        crud.create(
            test_session,
            user_id=sample_user.id,
            income_type="salary",
            description="เงินเดือน",
            amount=Decimal("50000.00"),
            tax_year=2025,
            pay_period="monthly",
        )
        result = get_income_summary(test_session, sample_user.id, 2025)
        assert result["total_income"] == "50000.00"
        assert result["source_count"] == 1
        assert result["sources"][0]["income_type"] == "salary"


class TestGetTaxFilingSummary:
    """Tests for get_tax_filing_summary."""

    def test_returns_not_found(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """No filing returns not_found status."""
        result = get_tax_filing_summary(test_session, sample_user.id, 2025)
        assert result["domain"] == "tax"
        assert result["status"] == "not_found"
        assert result["total_tax"] == "0"

    def test_returns_filing_data(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """With filing, returns tax data."""
        crud = TaxFilingCRUD()
        crud.create(
            test_session,
            user_id=sample_user.id,
            tax_year=2025,
            gross_income=Decimal("1200000.00"),
            total_deductions=Decimal("60000.00"),
            net_income=Decimal("1140000.00"),
            total_tax=Decimal("136000.00"),
            effective_tax_rate=Decimal("11.33"),
            tax_due_or_refund=Decimal("136000.00"),
            filing_status="filed",
        )
        result = get_tax_filing_summary(test_session, sample_user.id, 2025)
        assert result["status"] == "filed"
        assert result["gross_income"] == "1200000.00"
        assert result["total_tax"] == "136000.00"
