"""Cross-agent data retrieval service.

Provides functions that allow one agent to query data from
another agent's domain. Each function calls existing service
layers and returns serializable dicts for LLM consumption.
"""

import calendar
from datetime import date
from decimal import Decimal
from typing import Any

from sqlalchemy.orm import Session

from finance_ai.database.crud.income_crud import IncomeCRUD
from finance_ai.database.crud.tax_filing_crud import TaxFilingCRUD


def get_expense_summary(
    session: Session,
    user_id: str,
    year: int,
    month: int,
) -> dict[str, Any]:
    """Get monthly expense summary for cross-agent use.

    Args:
        session: Database session.
        user_id: UUID of the user.
        year: Year to query.
        month: Month to query (1-12).

    Returns:
        Dict with total_amount, category_breakdown, transaction_count.

    Example:
        >>> summary = get_expense_summary(session, uid, 2026, 3)
    """
    from finance_ai.tools.expense_service import (  # noqa: PLC0415
        summarize_expenses_for_user,
    )

    first_day = date(year, month, 1)
    last_day_num = calendar.monthrange(year, month)[1]
    last_day = date(year, month, last_day_num)
    result = summarize_expenses_for_user(session, user_id, first_day, last_day)
    return _format_expense_result(result, first_day, last_day)


def _format_expense_result(
    result: Any,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    """Format expense summary result to serializable dict.

    Args:
        result: ExpenseSummaryResult from calculator.
        start_date: Period start date.
        end_date: Period end date.

    Returns:
        Serializable dict with expense summary data.
    """
    return {
        "domain": "expense",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "total_amount": str(result.total_amount),
        "transaction_count": result.transaction_count,
        "category_breakdown": [
            {
                "category": cs.category,
                "label": cs.category_label,
                "amount": str(cs.total_amount),
            }
            for cs in result.category_breakdown
        ],
    }


def get_portfolio_summary(
    session: Session,
    user_id: str,
) -> dict[str, Any]:
    """Get investment portfolio summary for cross-agent use.

    Args:
        session: Database session.
        user_id: UUID of the user.

    Returns:
        Dict with portfolio value, cost, gains, and holdings.

    Example:
        >>> summary = get_portfolio_summary(session, uid)
    """
    from finance_ai.tools.investment_service import (  # noqa: PLC0415
        get_portfolio_summary as inv_summary,
    )

    result = inv_summary(session, user_id)
    return {
        "domain": "investment",
        "total_value": str(result.total_current_value or Decimal("0")),
        "total_cost": str(result.total_cost or Decimal("0")),
        "total_gain_loss": str(result.total_unrealized_gain_loss or Decimal("0")),
        "holding_count": result.holding_count,
        "holdings": [
            {
                "symbol": h.symbol,
                "name": h.name,
                "asset_type": h.asset_type,
                "current_value": str(h.current_value or Decimal("0")),
                "gain_loss": str(h.unrealized_gain_loss or Decimal("0")),
            }
            for h in result.holdings
        ],
    }


def get_goals_summary(
    session: Session,
    user_id: str,
) -> dict[str, Any]:
    """Get financial goals summary for cross-agent use.

    Args:
        session: Database session.
        user_id: UUID of the user.

    Returns:
        Dict with goal counts, progress, and per-goal details.

    Example:
        >>> summary = get_goals_summary(session, uid)
    """
    from finance_ai.tools.planning_service import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        summarize_goals_for_user,
    )

    result = summarize_goals_for_user(session, user_id)
    return {
        "domain": "planning",
        "total_goals": result.total_goals,
        "active_goals": result.active_goals,
        "completed_goals": result.completed_goals,
        "overall_percentage": str(result.overall_percentage),
        "goals": [
            {
                "name": g.name,
                "goal_type": g.goal_type,
                "target_amount": str(g.target_amount),
                "current_amount": str(g.current_amount),
                "percentage": str(g.percentage),
                "is_completed": g.is_completed,
            }
            for g in result.goals
        ],
    }


def get_income_summary(
    session: Session,
    user_id: str,
    tax_year: int,
) -> dict[str, Any]:
    """Get income summary for a tax year for cross-agent use.

    Args:
        session: Database session.
        user_id: UUID of the user.
        tax_year: Tax year to query.

    Returns:
        Dict with total income and per-source breakdown.

    Example:
        >>> summary = get_income_summary(session, uid, 2025)
    """
    crud = IncomeCRUD()
    total = crud.get_total_income_for_year(session, user_id, tax_year)
    records = crud.get_by_user_and_year(session, user_id, tax_year)
    return {
        "domain": "income",
        "tax_year": tax_year,
        "total_income": str(total),
        "source_count": len(records),
        "sources": [
            {
                "income_type": r.income_type,
                "description": r.description or "",
                "amount": str(r.amount),
            }
            for r in records
        ],
    }


def get_monthly_income_summary(
    session: Session,
    user_id: str,
    year: int,
    month: int,
) -> dict[str, Any]:
    """Get income transactions for a specific month from the transactions table.

    Used by the dashboard to show monthly income without dividing by 12.
    Income is read from the transactions table (date-specific records).

    Args:
        session: Database session.
        user_id: UUID of the user.
        year: Year to query.
        month: Month to query (1-12).

    Returns:
        Dict with total_income for the month and source count.

    Example:
        >>> summary = get_monthly_income_summary(session, uid, 2026, 3)
    """
    from finance_ai.database.crud.transaction_crud import (  # noqa: PLC0415
        TransactionCRUD,
    )

    first_day = date(year, month, 1)
    last_day_num = calendar.monthrange(year, month)[1]
    last_day = date(year, month, last_day_num)
    crud = TransactionCRUD()
    txns = crud.get_by_user_type_and_date_range(session, user_id, "income", first_day, last_day)
    total = sum((t.amount for t in txns), Decimal("0"))
    return {
        "domain": "income_monthly",
        "total_income": str(total),
        "source_count": len(txns),
    }


def get_tax_filing_summary(
    session: Session,
    user_id: str,
    tax_year: int,
) -> dict[str, Any]:
    """Get tax filing summary for cross-agent use.

    Args:
        session: Database session.
        user_id: UUID of the user.
        tax_year: Tax year to query.

    Returns:
        Dict with tax filing data or not_found status.

    Example:
        >>> summary = get_tax_filing_summary(session, uid, 2025)
    """
    crud = TaxFilingCRUD()
    filing = crud.get_by_user_and_year(session, user_id, tax_year)
    if filing is None:
        return _build_empty_tax_summary(tax_year)
    return _build_tax_summary(filing, tax_year)


def _build_empty_tax_summary(tax_year: int) -> dict[str, Any]:
    """Build empty tax summary when no filing exists.

    Args:
        tax_year: Tax year queried.

    Returns:
        Dict with not_found status.
    """
    return {
        "domain": "tax",
        "tax_year": tax_year,
        "status": "not_found",
        "total_tax": str(Decimal("0")),
    }


def _build_tax_summary(filing: Any, tax_year: int) -> dict[str, Any]:
    """Build tax summary from a filing record.

    Args:
        filing: TaxFiling model instance.
        tax_year: Tax year queried.

    Returns:
        Serializable dict with tax filing data.
    """
    return {
        "domain": "tax",
        "tax_year": tax_year,
        "status": filing.filing_status or "draft",
        "gross_income": str(filing.gross_income or Decimal("0")),
        "total_deductions": str(filing.total_deductions or Decimal("0")),
        "net_income": str(filing.net_income or Decimal("0")),
        "total_tax": str(filing.total_tax or Decimal("0")),
        "effective_tax_rate": str(filing.effective_tax_rate or Decimal("0")),
        "tax_due_or_refund": str(filing.tax_due_or_refund or Decimal("0")),
    }
