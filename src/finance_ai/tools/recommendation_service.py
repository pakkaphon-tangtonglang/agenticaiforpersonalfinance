"""Proactive financial recommendation service.

Rule-based analysis engine that gathers all user financial data
and generates prioritized recommendations. Each analyze function
checks a specific domain for potential improvements.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from finance_ai.tools.recommendation_constants import (
    EMERGENCY_FUND_MONTHS,
    EXPENSE_CONCENTRATION_THRESHOLD,
    GOAL_BEHIND_SCHEDULE_PERCENTAGE,
    HEALTH_SCORE_BASE,
    HEALTH_SCORE_PENALTY_PER_PRIORITY,
    INVESTMENT_CONCENTRATION_THRESHOLD,
    OVERSPENDING_THRESHOLD,
    SAVINGS_RATE_WARNING_THRESHOLD,
)


class Recommendation(BaseModel):
    """A single proactive financial recommendation.

    Attributes:
        category: Category key (maps to RECOMMENDATION_CATEGORIES).
        priority: Priority level 1-5 (5=critical).
        title: Short recommendation title in Thai.
        description: Detailed explanation in Thai.
        action_items: List of actionable steps.
        estimated_impact: Estimated financial impact description.

    Example:
        >>> rec = Recommendation(
        ...     category="tax_optimization", priority=4,
        ...     title="ยังไม่ได้ยื่นภาษี", description="...",
        ...     action_items=["ยื่นภาษี"], estimated_impact="..."
        ... )
    """

    category: str
    priority: int = Field(ge=1, le=5)
    title: str
    description: str
    action_items: list[str] = Field(default_factory=list)
    estimated_impact: str = ""


class RecommendationReport(BaseModel):
    """Full recommendation report for a user.

    Attributes:
        user_id: UUID of the user.
        generated_at: ISO datetime of generation.
        total_recommendations: Number of recommendations.
        health_score: Financial health score (0-100).
        recommendations: List of recommendations sorted by priority.

    Example:
        >>> report = RecommendationReport(
        ...     user_id="abc", generated_at="2026-01-01T00:00:00",
        ...     total_recommendations=0, health_score=100, recommendations=[],
        ... )
    """

    user_id: str
    generated_at: str
    total_recommendations: int
    health_score: int = Field(ge=0, le=100)
    recommendations: list[Recommendation] = Field(default_factory=list)


def gather_all_financial_data(
    session: Session,
    user_id: str,
    year: int,
    month: int,
) -> dict[str, dict[str, Any]]:
    """Gather data from all financial domains.

    Args:
        session: Database session.
        user_id: UUID of the user.
        year: Year to analyze.
        month: Month to analyze.

    Returns:
        Dict keyed by domain name with data from each service.

    Example:
        >>> data = gather_all_financial_data(session, uid, 2026, 3)
    """
    from finance_ai.tools.cross_agent_service import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        get_expense_summary,
        get_goals_summary,
        get_income_summary,
        get_monthly_income_summary,
        get_portfolio_summary,
        get_tax_filing_summary,
    )

    return {
        "expense": get_expense_summary(session, user_id, year, month),
        "portfolio": get_portfolio_summary(session, user_id),
        "goals": get_goals_summary(session, user_id),
        "income": get_income_summary(session, user_id, year),
        "income_monthly": get_monthly_income_summary(session, user_id, year, month),
        "tax": get_tax_filing_summary(session, user_id, year),
    }


def analyze_expense_patterns(
    expense_data: dict[str, Any],
    income_data: dict[str, Any],
) -> list[Recommendation]:
    """Detect overspending and expense concentration.

    Args:
        expense_data: Expense summary from cross-agent service.
        income_data: Income summary from cross-agent service.

    Returns:
        List of expense-related recommendations.

    Example:
        >>> recs = analyze_expense_patterns(expense, income)
    """
    recommendations: list[Recommendation] = []
    total_expense = Decimal(expense_data.get("total_amount", "0"))
    total_income = Decimal(income_data.get("total_income", "0"))

    if total_income > 0 and total_expense > 0:
        recommendations.extend(_check_overspending(total_expense, total_income))

    recommendations.extend(_check_expense_concentration(expense_data, total_expense))
    return recommendations


def _check_overspending(
    total_expense: Decimal,
    total_income: Decimal,
) -> list[Recommendation]:
    """Check if expenses exceed safe threshold of income.

    Args:
        total_expense: Total monthly expenses.
        total_income: Total annual income.

    Returns:
        List with overspending recommendation if triggered.
    """
    if total_income <= 0:
        return []
    ratio = total_expense / total_income
    if ratio > OVERSPENDING_THRESHOLD:
        pct = int(ratio * 100)
        return [
            Recommendation(
                category="savings_rate",
                priority=4,
                title="ค่าใช้จ่ายสูงเกินไป",
                description=f"ค่าใช้จ่ายคิดเป็น {pct}% ของรายได้รายเดือน" f" (แนะนำไม่เกิน 80%)",
                action_items=[
                    "ทบทวนรายจ่ายที่ไม่จำเป็น",
                    "ตั้งงบประมาณรายเดือน",
                ],
                estimated_impact=f"ลดรายจ่ายเพื่อเพิ่มเงินออม ~{total_expense - total_income * OVERSPENDING_THRESHOLD:.0f} บาท/เดือน",
            )
        ]
    return []


def _check_expense_concentration(
    expense_data: dict[str, Any],
    total_expense: Decimal,
) -> list[Recommendation]:
    """Check if any expense category is too concentrated.

    Args:
        expense_data: Expense summary with category_breakdown.
        total_expense: Total monthly expenses.

    Returns:
        List of concentration recommendations.
    """
    if total_expense <= 0:
        return []
    recommendations: list[Recommendation] = []
    for cat in expense_data.get("category_breakdown", []):
        amount = Decimal(cat.get("amount", "0"))
        ratio = amount / total_expense
        if ratio > EXPENSE_CONCENTRATION_THRESHOLD:
            label = cat.get("label", cat.get("category", ""))
            pct = int(ratio * 100)
            recommendations.append(
                Recommendation(
                    category="expense_optimization",
                    priority=3,
                    title=f"ค่าใช้จ่ายหมวด{label}สูง",
                    description=f"หมวด{label}คิดเป็น {pct}% ของรายจ่ายทั้งหมด"
                    f" (แนะนำไม่เกิน 40%)",
                    action_items=[f"หาทางลดค่า{label}"],
                    estimated_impact=f"ลดค่า{label}ได้ ~{amount - total_expense * EXPENSE_CONCENTRATION_THRESHOLD:.0f} บาท/เดือน",
                )
            )
    return recommendations


def analyze_tax_optimization(
    tax_data: dict[str, Any],
    income_data: dict[str, Any],
) -> list[Recommendation]:
    """Detect unused tax deduction capacity.

    Args:
        tax_data: Tax filing summary from cross-agent service.
        income_data: Income summary from cross-agent service.

    Returns:
        List of tax optimization recommendations.

    Example:
        >>> recs = analyze_tax_optimization(tax, income)
    """
    recommendations: list[Recommendation] = []
    total_income = Decimal(income_data.get("total_income", "0"))
    if total_income <= 0:
        return recommendations

    if tax_data.get("status") == "not_found":
        recommendations.append(
            Recommendation(
                category="tax_optimization",
                priority=5,
                title="ยังไม่ได้ยื่นภาษี",
                description=f"ยังไม่พบข้อมูลการยื่นภาษีปี {tax_data.get('tax_year', '')}",
                action_items=[
                    "รวบรวมเอกสารรายได้",
                    "คำนวณภาษีด้วยระบบ",
                ],
                estimated_impact="ยื่นภาษีตรงเวลาเพื่อหลีกเลี่ยงค่าปรับ",
            )
        )
        return recommendations

    recommendations.extend(_check_deduction_utilization(tax_data))
    return recommendations


def _check_deduction_utilization(
    tax_data: dict[str, Any],
) -> list[Recommendation]:
    """Check if tax deductions are underutilized.

    Args:
        tax_data: Tax filing summary with deduction data.

    Returns:
        List with recommendation if deductions are low.
    """
    gross = Decimal(tax_data.get("gross_income", "0"))
    deductions = Decimal(tax_data.get("total_deductions", "0"))
    if gross <= 0:
        return []
    # Simple heuristic: if deductions < 50% of reasonable maximum
    # Reasonable max ~ 30% of gross income for most Thai taxpayers
    reasonable_max = gross * Decimal("0.30")
    if reasonable_max <= 0:
        return []
    utilization = deductions / reasonable_max
    if utilization < Decimal("0.50"):
        pct = int(utilization * 100)
        return [
            Recommendation(
                category="tax_optimization",
                priority=4,
                title="ใช้สิทธิ์ลดหย่อนภาษีน้อย",
                description=f"ใช้สิทธิ์ลดหย่อนเพียง {pct}% ของที่ควรใช้ได้",
                action_items=[
                    "พิจารณาซื้อ RMF/SSF",
                    "ตรวจสอบค่าลดหย่อนประกันชีวิต",
                    "ตรวจสอบค่าลดหย่อนดอกเบี้ยบ้าน",
                ],
                estimated_impact="ประหยัดภาษีได้เพิ่มเติม",
            )
        ]
    return []


def analyze_investment_risk(
    portfolio_data: dict[str, Any],
) -> list[Recommendation]:
    """Detect concentration risk in investment portfolio.

    Args:
        portfolio_data: Portfolio summary from cross-agent service.

    Returns:
        List of investment-related recommendations.

    Example:
        >>> recs = analyze_investment_risk(portfolio)
    """
    recommendations: list[Recommendation] = []
    holdings = portfolio_data.get("holdings", [])
    if not holdings:
        return recommendations

    total_value = Decimal(portfolio_data.get("total_value", "0"))
    if total_value <= 0:
        return recommendations

    recommendations.extend(_check_holding_concentration(holdings, total_value))
    recommendations.extend(_check_portfolio_loss(portfolio_data))
    return recommendations


def _check_holding_concentration(
    holdings: list[dict[str, Any]],
    total_value: Decimal,
) -> list[Recommendation]:
    """Check if any single holding is too concentrated.

    Args:
        holdings: List of holding dicts from portfolio summary.
        total_value: Total portfolio value.

    Returns:
        List of concentration recommendations.
    """
    recommendations: list[Recommendation] = []
    for holding in holdings:
        value = Decimal(holding.get("current_value", "0"))
        if value <= 0:
            continue
        ratio = value / total_value
        if ratio > INVESTMENT_CONCENTRATION_THRESHOLD:
            symbol = holding.get("symbol", "N/A")
            pct = int(ratio * 100)
            recommendations.append(
                Recommendation(
                    category="investment_rebalancing",
                    priority=3,
                    title=f"{symbol} สัดส่วนสูงเกินไป",
                    description=f"{symbol} คิดเป็น {pct}% ของพอร์ต" f" (แนะนำไม่เกิน 30%)",
                    action_items=["กระจายการลงทุนไปสินทรัพย์อื่น"],
                    estimated_impact="ลดความเสี่ยงจากการกระจุกตัว",
                )
            )
    return recommendations


def _check_portfolio_loss(
    portfolio_data: dict[str, Any],
) -> list[Recommendation]:
    """Check if portfolio has unrealized losses.

    Args:
        portfolio_data: Portfolio summary from cross-agent service.

    Returns:
        List with loss recommendation if applicable.
    """
    gain_loss = Decimal(portfolio_data.get("total_gain_loss", "0"))
    if gain_loss < 0:
        return [
            Recommendation(
                category="investment_rebalancing",
                priority=2,
                title="พอร์ตขาดทุน",
                description=f"พอร์ตมีผลขาดทุนที่ยังไม่รับรู้ {gain_loss} บาท",
                action_items=[
                    "ทบทวนกลยุทธ์การลงทุน",
                    "พิจารณา tax-loss harvesting",
                ],
                estimated_impact="ปรับกลยุทธ์เพื่อลดการขาดทุน",
            )
        ]
    return []


def analyze_goal_progress(
    goals_data: dict[str, Any],
) -> list[Recommendation]:
    """Detect goals that are behind schedule.

    Args:
        goals_data: Goals summary from cross-agent service.

    Returns:
        List of goal-related recommendations.

    Example:
        >>> recs = analyze_goal_progress(goals)
    """
    recommendations: list[Recommendation] = []
    goals = goals_data.get("goals", [])
    if not goals:
        return recommendations

    for goal in goals:
        if goal.get("is_completed", False):
            continue
        percentage = Decimal(goal.get("percentage", "0"))
        if percentage < GOAL_BEHIND_SCHEDULE_PERCENTAGE:
            name = goal.get("name", "ไม่ระบุ")
            recommendations.append(
                Recommendation(
                    category="goal_progress",
                    priority=4,
                    title=f"เป้าหมาย '{name}' ล่าช้า",
                    description=f"ความคืบหน้า {percentage}%" f" (ควรมากกว่า 50%)",
                    action_items=["เพิ่มเงินออมรายเดือน"],
                    estimated_impact="ให้ถึงเป้าหมายตามกำหนด",
                )
            )
    return recommendations


def analyze_savings_rate(
    income_data: dict[str, Any],
    expense_data: dict[str, Any],
    goals_data: dict[str, Any],
) -> list[Recommendation]:
    """Check savings rate and emergency fund status.

    Args:
        income_data: Income summary from cross-agent service.
        expense_data: Expense summary from cross-agent service.
        goals_data: Goals summary from cross-agent service.

    Returns:
        List of savings-related recommendations.

    Example:
        >>> recs = analyze_savings_rate(income, expense, goals)
    """
    recommendations: list[Recommendation] = []
    total_income = Decimal(income_data.get("total_income", "0"))
    total_expense = Decimal(expense_data.get("total_amount", "0"))

    recommendations.extend(_check_savings_rate(total_income, total_expense))
    recommendations.extend(_check_emergency_fund(goals_data, total_expense))
    return recommendations


def _check_savings_rate(
    total_income: Decimal,
    total_expense: Decimal,
) -> list[Recommendation]:
    """Check if monthly savings rate is below threshold.

    Args:
        total_income: Total annual income.
        total_expense: Total monthly expenses.

    Returns:
        List with recommendation if savings rate is low.
    """
    if total_income <= 0:
        return []
    savings = total_income - total_expense
    savings_rate = savings / total_income
    if savings_rate < SAVINGS_RATE_WARNING_THRESHOLD:
        pct = int(savings_rate * 100)
        return [
            Recommendation(
                category="savings_rate",
                priority=3,
                title="อัตราการออมต่ำ",
                description=f"อัตราการออม {pct}% ของรายได้" f" (แนะนำอย่างน้อย 20%)",
                action_items=["ตั้งเป้าออมอย่างน้อย 20% ของรายได้"],
                estimated_impact=f"เพิ่มเงินออม ~{total_income * SAVINGS_RATE_WARNING_THRESHOLD - savings:.0f} บาท/เดือน",
            )
        ]
    return []


def _check_emergency_fund(
    goals_data: dict[str, Any],
    total_expense: Decimal,
) -> list[Recommendation]:
    """Check if user has an emergency fund goal.

    Args:
        goals_data: Goals summary from cross-agent service.
        total_expense: Monthly expense total.

    Returns:
        List with recommendation if no emergency fund goal exists.
    """
    goals = goals_data.get("goals", [])
    has_emergency = any(g.get("goal_type") == "emergency_fund" for g in goals)
    if not has_emergency:
        target = total_expense * EMERGENCY_FUND_MONTHS if total_expense > 0 else Decimal("0")
        return [
            Recommendation(
                category="emergency_fund",
                priority=5,
                title="ไม่มีเงินสำรองฉุกเฉิน",
                description=f"แนะนำให้มีเงินสำรอง {EMERGENCY_FUND_MONTHS} เดือน"
                f" (~{target:.0f} บาท)",
                action_items=["สร้างเป้าหมายเงินสำรองฉุกเฉิน"],
                estimated_impact="ป้องกันความเสี่ยงทางการเงินฉุกเฉิน",
            )
        ]
    return []


def calculate_health_score(
    recommendations: list[Recommendation],
) -> int:
    """Calculate financial health score based on recommendations.

    Higher score = healthier finances. Fewer and less severe
    recommendations result in a higher score.

    Args:
        recommendations: List of generated recommendations.

    Returns:
        Health score from 0 to 100.

    Example:
        >>> calculate_health_score([])
        100
    """
    penalty = sum(r.priority * HEALTH_SCORE_PENALTY_PER_PRIORITY for r in recommendations)
    return max(0, HEALTH_SCORE_BASE - penalty)


def generate_recommendations(
    session: Session,
    user_id: str,
    year: int,
    month: int,
) -> RecommendationReport:
    """Generate full recommendation report for a user.

    Gathers all financial data, runs all analysis rules,
    and returns a prioritized recommendation report.

    Args:
        session: Database session.
        user_id: UUID of the user.
        year: Year to analyze.
        month: Month to analyze.

    Returns:
        RecommendationReport with prioritized recommendations.

    Example:
        >>> report = generate_recommendations(session, uid, 2026, 3)
    """
    data = gather_all_financial_data(session, user_id, year, month)
    recommendations = _run_all_analyses(data)
    recommendations.sort(key=lambda r: r.priority, reverse=True)
    score = calculate_health_score(recommendations)
    return _build_report(user_id, recommendations, score)


def _run_all_analyses(
    data: dict[str, dict[str, Any]],
) -> list[Recommendation]:
    """Run all analysis rules against gathered data.

    Args:
        data: Combined financial data from all domains.

    Returns:
        Combined list of all recommendations.
    """
    recommendations: list[Recommendation] = []
    recommendations.extend(analyze_expense_patterns(data["expense"], data["income_monthly"]))
    recommendations.extend(analyze_tax_optimization(data["tax"], data["income"]))
    recommendations.extend(analyze_investment_risk(data["portfolio"]))
    recommendations.extend(analyze_goal_progress(data["goals"]))
    recommendations.extend(
        analyze_savings_rate(data["income_monthly"], data["expense"], data["goals"])
    )
    return recommendations


def _build_report(
    user_id: str,
    recommendations: list[Recommendation],
    health_score: int,
) -> RecommendationReport:
    """Build the final recommendation report.

    Args:
        user_id: UUID of the user.
        recommendations: Sorted list of recommendations.
        health_score: Calculated health score.

    Returns:
        RecommendationReport instance.
    """
    return RecommendationReport(
        user_id=user_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        total_recommendations=len(recommendations),
        health_score=health_score,
        recommendations=recommendations,
    )
