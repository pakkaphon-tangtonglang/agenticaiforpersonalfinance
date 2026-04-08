"""Pure functions for generating autonomous financial reports.

Reuses gather_all_financial_data() and generate_recommendations()
from recommendation_service, then builds structured report sections.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy.orm import Session

from finance_ai.tools.report_constants import (
    DEFAULT_REPORT_TYPE,
    GOAL_NEAR_COMPLETE_THRESHOLD,
    MONTHS_PER_YEAR,
    SAVINGS_RATE_GOOD_THRESHOLD,
)
from finance_ai.tools.report_models import (
    ExpenseBreakdownSection,
    FinancialReport,
    GoalProgressSection,
    InvestmentPortfolioSection,
    MonthlyOverviewSection,
    ReportHighlight,
    TaxStatusSection,
)


def generate_financial_report(
    session: Session,
    user_id: str,
    year: int,
    month: int,
) -> FinancialReport:
    """Generate a comprehensive financial report for a user.

    Gathers all financial data, builds each section, generates
    highlights, and calculates health score.

    Args:
        session: Database session.
        user_id: UUID of the user.
        year: Year to report on.
        month: Month to report on.

    Returns:
        Complete FinancialReport with all sections populated.

    Example:
        >>> report = generate_financial_report(session, uid, 2026, 3)
    """
    data = _gather_data(session, user_id, year, month)
    recommendations = _get_recommendations(session, user_id, year, month)
    return _assemble_report(user_id, year, month, data, recommendations)


def _gather_data(
    session: Session,
    user_id: str,
    year: int,
    month: int,
) -> dict[str, dict[str, Any]]:
    """Gather all financial data via recommendation service.

    Args:
        session: Database session.
        user_id: UUID of the user.
        year: Year to analyze.
        month: Month to analyze.

    Returns:
        Dict keyed by domain name with data from each service.
    """
    from finance_ai.tools.recommendation_service import (  # noqa: PLC0415
        gather_all_financial_data,
    )

    return gather_all_financial_data(session, user_id, year, month)


def _get_recommendations(
    session: Session,
    user_id: str,
    year: int,
    month: int,
) -> Any:
    """Get recommendation report for health score and highlights.

    Args:
        session: Database session.
        user_id: UUID of the user.
        year: Year to analyze.
        month: Month to analyze.

    Returns:
        RecommendationReport instance.
    """
    from finance_ai.tools.recommendation_service import (  # noqa: PLC0415
        generate_recommendations,
    )

    return generate_recommendations(session, user_id, year, month)


def _assemble_report(
    user_id: str,
    year: int,
    month: int,
    data: dict[str, dict[str, Any]],
    recommendations: Any,
) -> FinancialReport:
    """Assemble all sections into a FinancialReport.

    Args:
        user_id: UUID of the user.
        year: Report year.
        month: Report month.
        data: Raw financial data from all domains.
        recommendations: RecommendationReport with health_score.

    Returns:
        Complete FinancialReport.
    """
    overview = build_monthly_overview(data["income_monthly"], data["expense"])
    highlights = _generate_highlights(data, overview, recommendations)

    return FinancialReport(
        user_id=user_id,
        generated_at=datetime.now().isoformat(),
        report_type=DEFAULT_REPORT_TYPE,
        year=year,
        month=month,
        monthly_overview=overview,
        expense_breakdown=build_expense_breakdown(data["expense"]),
        investment_portfolio=build_investment_portfolio(data["portfolio"]),
        goal_progress=build_goal_progress(data["goals"]),
        tax_status=build_tax_status(data["tax"], year),
        health_score=recommendations.health_score,
        highlights=highlights,
    )


def build_monthly_overview(
    income_data: dict[str, Any],
    expense_data: dict[str, Any],
) -> MonthlyOverviewSection:
    """Build monthly income vs expense overview.

    Args:
        income_data: Income summary from cross-agent service.
        expense_data: Expense summary from cross-agent service.

    Returns:
        MonthlyOverviewSection with calculated savings.

    Example:
        >>> overview = build_monthly_overview(income, expense)
    """
    monthly_income = Decimal(str(income_data.get("total_income", "0")))
    total_expenses = Decimal(str(expense_data.get("total_amount", "0")))
    net_savings = monthly_income - total_expenses
    savings_rate = _safe_divide(net_savings, monthly_income)

    return MonthlyOverviewSection(
        total_income=monthly_income,
        total_expenses=total_expenses,
        net_savings=net_savings,
        savings_rate=savings_rate,
    )


def _safe_divide(numerator: Decimal, denominator: Decimal | int) -> Decimal:
    """Safely divide two numbers, returning 0 on division by zero.

    Args:
        numerator: The dividend.
        denominator: The divisor.

    Returns:
        Result of division, or Decimal("0") if denominator is zero.
    """
    denom = Decimal(str(denominator))
    if denom == 0:
        return Decimal("0")
    return numerator / denom


def build_expense_breakdown(
    expense_data: dict[str, Any],
) -> ExpenseBreakdownSection:
    """Build expense breakdown by category.

    Args:
        expense_data: Expense summary from cross-agent service.

    Returns:
        ExpenseBreakdownSection with category percentages.

    Example:
        >>> breakdown = build_expense_breakdown(expense_data)
    """
    total = Decimal(str(expense_data.get("total_amount", "0")))
    categories = expense_data.get("category_breakdown", [])
    enriched = _enrich_categories(categories, total)

    return ExpenseBreakdownSection(
        total_amount=total,
        transaction_count=int(expense_data.get("transaction_count", 0)),
        categories=enriched,
    )


def _enrich_categories(
    categories: list[dict[str, Any]],
    total: Decimal,
) -> list[dict[str, Any]]:
    """Add percentage to each category entry.

    Args:
        categories: Raw category list from expense summary.
        total: Total expense amount for percentage calculation.

    Returns:
        Categories with added 'percentage' field.
    """
    result: list[dict[str, Any]] = []
    for cat in categories:
        amount = Decimal(str(cat.get("amount", "0")))
        pct = _safe_divide(amount * 100, total)
        entry = dict(cat)
        entry["percentage"] = str(pct.quantize(Decimal("0.01")))
        result.append(entry)
    return result


def build_investment_portfolio(
    portfolio_data: dict[str, Any],
) -> InvestmentPortfolioSection:
    """Build investment portfolio status section.

    Args:
        portfolio_data: Portfolio summary from cross-agent service.

    Returns:
        InvestmentPortfolioSection with gain/loss percentage.

    Example:
        >>> portfolio = build_investment_portfolio(portfolio_data)
    """
    total_value = Decimal(str(portfolio_data.get("total_value", "0")))
    total_cost = Decimal(str(portfolio_data.get("total_cost", "0")))
    gain_loss = Decimal(str(portfolio_data.get("total_gain_loss", "0")))
    pct = _safe_divide(gain_loss * 100, total_cost)

    return InvestmentPortfolioSection(
        total_value=total_value,
        total_cost=total_cost,
        total_gain_loss=gain_loss,
        gain_loss_percentage=pct.quantize(Decimal("0.01")),
        holding_count=int(portfolio_data.get("holding_count", 0)),
        holdings=portfolio_data.get("holdings", []),
    )


def build_goal_progress(
    goals_data: dict[str, Any],
) -> GoalProgressSection:
    """Build goal progress section.

    Args:
        goals_data: Goals summary from cross-agent service.

    Returns:
        GoalProgressSection with completion counts.

    Example:
        >>> progress = build_goal_progress(goals_data)
    """
    total = int(goals_data.get("total_goals", 0))
    active = int(goals_data.get("active_goals", 0))

    return GoalProgressSection(
        total_goals=total,
        active_goals=active,
        completed_goals=total - active,
        overall_percentage=Decimal(str(goals_data.get("overall_percentage", "0"))),
        goals=goals_data.get("goals", []),
    )


def build_tax_status(
    tax_data: dict[str, Any],
    year: int,
) -> TaxStatusSection:
    """Build tax filing status section.

    Args:
        tax_data: Tax filing summary from cross-agent service.
        year: Tax year for the report.

    Returns:
        TaxStatusSection with filing details.

    Example:
        >>> tax = build_tax_status(tax_data, 2026)
    """
    return TaxStatusSection(
        status=str(tax_data.get("status", "not_filed")),
        gross_income=Decimal(str(tax_data.get("gross_income", "0"))),
        total_deductions=Decimal(str(tax_data.get("total_deductions", "0"))),
        total_tax=Decimal(str(tax_data.get("total_tax", "0"))),
        effective_rate=Decimal(str(tax_data.get("effective_tax_rate", "0"))),
        tax_year=year,
    )


def _generate_highlights(
    data: dict[str, dict[str, Any]],
    overview: MonthlyOverviewSection,
    recommendations: Any,
) -> list[ReportHighlight]:
    """Generate report highlights from financial data.

    Args:
        data: Raw financial data from all domains.
        overview: Computed monthly overview section.
        recommendations: RecommendationReport for warnings.

    Returns:
        List of ReportHighlight items.
    """
    highlights: list[ReportHighlight] = []
    highlights.extend(_check_savings_highlight(overview))
    highlights.extend(_check_goal_highlights(data["goals"]))
    highlights.extend(_check_portfolio_highlight(data["portfolio"]))
    highlights.extend(_check_recommendation_highlights(recommendations))
    return highlights


def _check_savings_highlight(
    overview: MonthlyOverviewSection,
) -> list[ReportHighlight]:
    """Check if savings rate is noteworthy.

    Args:
        overview: Monthly overview with savings rate.

    Returns:
        Achievement highlight if savings rate is good.
    """
    if overview.savings_rate >= SAVINGS_RATE_GOOD_THRESHOLD:
        pct = int(overview.savings_rate * 100)
        return [
            ReportHighlight(
                highlight_type="achievement",
                title="อัตราการออมดี",
                description=f"ออมได้ {pct}% ของรายได้ (เป้าหมาย ≥20%)",
            )
        ]
    return []


def _check_goal_highlights(
    goals_data: dict[str, Any],
) -> list[ReportHighlight]:
    """Check for goals near completion.

    Args:
        goals_data: Goals summary from cross-agent service.

    Returns:
        Achievement highlights for near-complete goals.
    """
    highlights: list[ReportHighlight] = []
    for goal in goals_data.get("goals", []):
        pct = Decimal(str(goal.get("percentage", "0")))
        if pct >= GOAL_NEAR_COMPLETE_THRESHOLD:
            name = goal.get("name", "เป้าหมาย")
            highlights.append(
                ReportHighlight(
                    highlight_type="achievement",
                    title=f"เป้าหมายใกล้สำเร็จ: {name}",
                    description=f"ความคืบหน้า {pct}%",
                )
            )
    return highlights


def _check_portfolio_highlight(
    portfolio_data: dict[str, Any],
) -> list[ReportHighlight]:
    """Check portfolio for notable gains or losses.

    Args:
        portfolio_data: Portfolio summary from cross-agent service.

    Returns:
        Warning highlight if portfolio has significant losses.
    """
    gain_loss = Decimal(str(portfolio_data.get("total_gain_loss", "0")))
    if gain_loss < 0:
        return [
            ReportHighlight(
                highlight_type="warning",
                title="พอร์ตขาดทุน",
                description=f"ขาดทุนรวม {gain_loss} บาท",
            )
        ]
    return []


def _check_recommendation_highlights(
    recommendations: Any,
) -> list[ReportHighlight]:
    """Convert top recommendation issues to warning highlights.

    Args:
        recommendations: RecommendationReport with recommendations list.

    Returns:
        Warning highlights from high-priority recommendations.
    """
    highlights: list[ReportHighlight] = []
    for rec in recommendations.recommendations[:3]:
        if rec.priority >= 4:
            highlights.append(
                ReportHighlight(
                    highlight_type="warning",
                    title=rec.title,
                    description=rec.description,
                )
            )
    return highlights
