"""Pillar-based financial health scoring.

Each pillar (spend, save, invest, plan) is scored 0-100 from data that
exists, or None when its data is missing. The overall score is the mean
of known pillars, and is withheld (None) until the minimum data gate is
met: monthly income and monthly expenses for the selected month.
"""

from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from finance_ai.tools.health_score_constants import (
    HEALTH_PILLARS,
    HEALTH_STATUS_INSUFFICIENT,
    HEALTH_STATUS_SCORED,
    INVEST_CONCENTRATED_FLOOR,
    INVEST_CONCENTRATION_BEST,
    INVEST_CONCENTRATION_WORST,
    INVEST_LOSS_PENALTY,
    MISSING_MONTHLY_EXPENSE,
    MISSING_MONTHLY_INCOME,
    SAVINGS_RATE_BEST,
    SAVINGS_RATE_WORST,
    SCORE_MAX,
    SCORE_MIN,
    SPEND_RATIO_BEST,
    SPEND_RATIO_WORST,
    TAX_DRAFT_SCORE,
    TAX_DRAFT_STATUS,
    TAX_FILED_SCORE,
    TAX_NOT_FILED_SCORE,
    TAX_NOT_FOUND_STATUS,
)
from finance_ai.tools.health_score_models import HealthScoreResult


def _round_score(value: Decimal) -> int:
    """Round a Decimal score half-up and clamp to 0-100.

    Args:
        value: Raw score.

    Returns:
        Integer score within 0-100.
    """
    rounded = int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return max(SCORE_MIN, min(SCORE_MAX, rounded))


def linear_score(value: Decimal, best: Decimal, worst: Decimal) -> int:
    """Map a value linearly onto 0-100, where best=100 and worst=0.

    Works in either direction (best may be above or below worst).
    Values beyond the range are clamped.

    Args:
        value: Measured value.
        best: Value that earns 100.
        worst: Value that earns 0.

    Returns:
        Integer score 0-100.

    Raises:
        ValueError: If best equals worst.

    Example:
        >>> linear_score(Decimal("0.10"), Decimal("0.20"), Decimal("0"))
        50
    """
    if best == worst:
        raise ValueError(f"best and worst must differ. Received: best={best}, worst={worst}")
    fraction = (value - worst) / (best - worst)
    return _round_score(fraction * SCORE_MAX)


def _mean_score(scores: list[int]) -> int | None:
    """Average integer scores, or None for an empty list.

    Args:
        scores: Scores to average.

    Returns:
        Rounded mean, or None if there are no scores.
    """
    if not scores:
        return None
    return _round_score(Decimal(sum(scores)) / Decimal(len(scores)))


def score_spend_pillar(monthly_income: Decimal, monthly_expense: Decimal) -> int | None:
    """Score spending as the expense-to-income ratio.

    Args:
        monthly_income: Total income for the month.
        monthly_expense: Total expenses for the month.

    Returns:
        Score 0-100, or None if income or expenses are missing.

    Example:
        >>> score_spend_pillar(Decimal("50000"), Decimal("20000"))
        100
    """
    if monthly_income <= 0 or monthly_expense <= 0:
        return None
    ratio = monthly_expense / monthly_income
    return linear_score(ratio, SPEND_RATIO_BEST, SPEND_RATIO_WORST)


def score_save_pillar(monthly_income: Decimal, monthly_expense: Decimal) -> int | None:
    """Score saving as the monthly savings rate.

    Args:
        monthly_income: Total income for the month.
        monthly_expense: Total expenses for the month.

    Returns:
        Score 0-100, or None if income or expenses are missing.

    Example:
        >>> score_save_pillar(Decimal("10000"), Decimal("9000"))
        50
    """
    if monthly_income <= 0 or monthly_expense <= 0:
        return None
    savings_rate = (monthly_income - monthly_expense) / monthly_income
    return linear_score(savings_rate, SAVINGS_RATE_BEST, SAVINGS_RATE_WORST)


def score_invest_pillar(portfolio_data: dict[str, Any]) -> int | None:
    """Score investing by diversification, minus a penalty for net loss.

    Users without holdings get None (not penalised: investing is optional).

    Args:
        portfolio_data: Portfolio summary from cross-agent service.

    Returns:
        Score 0-100, or None if there are no holdings.

    Example:
        >>> score_invest_pillar({"holdings": [], "total_value": "0"}) is None
        True
    """
    total_value = Decimal(str(portfolio_data.get("total_value", "0")))
    holdings = portfolio_data.get("holdings", [])
    if not holdings or total_value <= 0:
        return None
    largest = max(Decimal(str(h.get("current_value", "0"))) for h in holdings)
    diversification = linear_score(
        largest / total_value, INVEST_CONCENTRATION_BEST, INVEST_CONCENTRATION_WORST
    )
    floor_range = Decimal(SCORE_MAX - INVEST_CONCENTRATED_FLOOR) / Decimal(SCORE_MAX)
    scaled = Decimal(INVEST_CONCENTRATED_FLOOR) + Decimal(diversification) * floor_range
    if Decimal(str(portfolio_data.get("total_gain_loss", "0"))) < 0:
        scaled -= INVEST_LOSS_PENALTY
    return _round_score(scaled)


def _goal_percentage(goal: dict[str, Any]) -> int:
    """Get a goal's progress as a 0-100 score (completed counts as 100).

    Args:
        goal: Goal dict from goals summary.

    Returns:
        Progress score 0-100.
    """
    if goal.get("is_completed", False):
        return SCORE_MAX
    return _round_score(Decimal(str(goal.get("percentage", "0"))))


def _tax_filing_score(tax_data: dict[str, Any], income_data: dict[str, Any]) -> int | None:
    """Score tax filing status; only applies when the user has annual income.

    Args:
        tax_data: Tax filing summary from cross-agent service.
        income_data: Annual income summary from cross-agent service.

    Returns:
        Tax filing score, or None if the user has no recorded income.
    """
    if Decimal(str(income_data.get("total_income", "0"))) <= 0:
        return None
    status = tax_data.get("status", TAX_NOT_FOUND_STATUS)
    if status == TAX_NOT_FOUND_STATUS:
        return TAX_NOT_FILED_SCORE
    if status == TAX_DRAFT_STATUS:
        return TAX_DRAFT_SCORE
    return TAX_FILED_SCORE


def score_plan_pillar(
    goals_data: dict[str, Any],
    tax_data: dict[str, Any],
    income_data: dict[str, Any],
) -> int | None:
    """Score planning from emergency fund, goal progress, and tax filing.

    Averages whichever sub-indicators have data.

    Args:
        goals_data: Goals summary from cross-agent service.
        tax_data: Tax filing summary from cross-agent service.
        income_data: Annual income summary from cross-agent service.

    Returns:
        Score 0-100, or None if no sub-indicator has data.

    Example:
        >>> score_plan_pillar({"goals": []}, {}, {"total_income": "0"}) is None
        True
    """
    goals = goals_data.get("goals", [])
    sub_scores = [_goal_percentage(g) for g in goals if g.get("goal_type") == "emergency_fund"]
    goal_progress = _mean_score([_goal_percentage(g) for g in goals])
    tax_score = _tax_filing_score(tax_data, income_data)
    sub_scores.extend(s for s in (goal_progress, tax_score) if s is not None)
    return _mean_score(sub_scores)


def _find_missing_data(monthly_income: Decimal, monthly_expense: Decimal) -> list[str]:
    """List data required by the minimum scoring gate that is missing.

    Args:
        monthly_income: Total income for the month.
        monthly_expense: Total expenses for the month.

    Returns:
        Thai labels of missing data (empty when the gate passes).
    """
    missing: list[str] = []
    if monthly_income <= 0:
        missing.append(MISSING_MONTHLY_INCOME)
    if monthly_expense <= 0:
        missing.append(MISSING_MONTHLY_EXPENSE)
    return missing


def _score_all_pillars(
    data: dict[str, dict[str, Any]],
    monthly_income: Decimal,
    monthly_expense: Decimal,
) -> dict[str, int | None]:
    """Score every pillar from gathered financial data.

    Args:
        data: Output of gather_all_financial_data().
        monthly_income: Total income for the month.
        monthly_expense: Total expenses for the month.

    Returns:
        Pillar key -> score or None.
    """
    return {
        "spend": score_spend_pillar(monthly_income, monthly_expense),
        "save": score_save_pillar(monthly_income, monthly_expense),
        "invest": score_invest_pillar(data["portfolio"]),
        "plan": score_plan_pillar(data["goals"], data["tax"], data["income"]),
    }


def calculate_health_score(data: dict[str, dict[str, Any]]) -> HealthScoreResult:
    """Calculate the overall financial health score.

    Args:
        data: Output of gather_all_financial_data().

    Returns:
        HealthScoreResult; score is None until monthly income and
        expenses both exist.

    Example:
        >>> calculate_health_score(data).score  # doctest: +SKIP
        72
    """
    monthly_income = Decimal(str(data["income_monthly"].get("total_income", "0")))
    monthly_expense = Decimal(str(data["expense"].get("total_amount", "0")))
    pillars = _score_all_pillars(data, monthly_income, monthly_expense)
    known = [s for s in pillars.values() if s is not None]
    coverage = (Decimal(len(known)) / Decimal(len(HEALTH_PILLARS))).quantize(Decimal("0.0001"))
    missing = _find_missing_data(monthly_income, monthly_expense)
    return HealthScoreResult(
        score=None if missing else _mean_score(known),
        status=HEALTH_STATUS_INSUFFICIENT if missing else HEALTH_STATUS_SCORED,
        pillars=pillars,
        coverage=coverage,
        missing_data=missing,
    )
