"""Pure spending-trend rules shared by recommendations and dashboard cards.

A category "spikes" when this month is well above the user's own average
of recent months: both a relative (SPENDING_TREND_MIN_INCREASE) and an
absolute (SPENDING_TREND_MIN_AMOUNT) threshold must be crossed, and at
least SPENDING_TREND_MIN_HISTORY_MONTHS months with spending are needed.
"""

from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from finance_ai.tools.recommendation_constants import (
    SPENDING_TREND_MIN_AMOUNT,
    SPENDING_TREND_MIN_HISTORY_MONTHS,
    SPENDING_TREND_MIN_INCREASE,
)

MONEY_PLACES = Decimal("0.01")


class CategorySpike(BaseModel):
    """One category spending well above the user's recent average.

    Example:
        >>> CategorySpike(category="food", label="อาหาร", current=Decimal("6000"),
        ...     average=Decimal("4000"), increase_pct=50, months_compared=3).increase_pct
        50
    """

    category: str
    label: str
    current: Decimal
    average: Decimal
    increase_pct: int
    months_compared: int


def category_amounts(summary: dict[str, Any]) -> dict[str, Decimal]:
    """Map category key to amount for one month's expense summary.

    Example:
        >>> category_amounts({"category_breakdown": [{"category": "food", "amount": "10"}]})
        {'food': Decimal('10')}
    """
    return {
        cat.get("category", ""): Decimal(str(cat.get("amount", "0")))
        for cat in summary.get("category_breakdown", [])
    }


def find_category_spikes(
    current_month: dict[str, Any], history: list[dict[str, Any]]
) -> list[CategorySpike]:
    """Categories this month that are well above the user's own average.

    Args:
        current_month: This month's expense summary (category_breakdown).
        history: Earlier months' summaries; months without spending are skipped.

    Returns:
        One spike per flagged category, in breakdown order.

    Example:
        >>> find_category_spikes(this_month, [m1, m2, m3])  # doctest: +SKIP
    """
    past = [category_amounts(m) for m in history if _month_total(m) > 0]
    if len(past) < SPENDING_TREND_MIN_HISTORY_MONTHS:
        return []
    labels = {
        c.get("category", ""): c.get("label", "")
        for c in current_month.get("category_breakdown", [])
    }
    spikes: list[CategorySpike] = []
    for category, current in category_amounts(current_month).items():
        average = sum((m.get(category, Decimal("0")) for m in past), Decimal("0")) / len(past)
        if _is_spike(current, average):
            spikes.append(_spike(category, labels[category], current, average, len(past)))
    return spikes


def average_monthly_total(summaries: list[dict[str, Any]]) -> Decimal:
    """Average total of the months that had any spending (0.00 when none).

    Example:
        >>> average_monthly_total([{"total_amount": "3000"}, {"total_amount": "0"}])
        Decimal('3000.00')
    """
    totals = [total for total in (_month_total(m) for m in summaries) if total > 0]
    if not totals:
        return Decimal("0").quantize(MONEY_PLACES)
    return (sum(totals, Decimal("0")) / len(totals)).quantize(MONEY_PLACES)


def _month_total(summary: dict[str, Any]) -> Decimal:
    """The month's total spending."""
    return Decimal(str(summary.get("total_amount", "0")))


def _is_spike(current: Decimal, average: Decimal) -> bool:
    """Both the relative and the absolute increase thresholds are crossed."""
    increase = current - average
    return (
        average > 0
        and increase >= SPENDING_TREND_MIN_AMOUNT
        and increase / average > SPENDING_TREND_MIN_INCREASE
    )


def _spike(
    category: str, label: str, current: Decimal, average: Decimal, months: int
) -> CategorySpike:
    """Build the spike record (percent truncated like the recommendation text)."""
    return CategorySpike(
        category=category,
        label=label,
        current=current,
        average=average.quantize(MONEY_PLACES),
        increase_pct=int((current - average) / average * 100),
        months_compared=months,
    )
