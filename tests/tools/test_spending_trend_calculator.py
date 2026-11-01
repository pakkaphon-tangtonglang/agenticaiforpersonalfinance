"""Tests for spending_trend_calculator (pure category-spike rule)."""

from decimal import Decimal
from typing import Any

from finance_ai.tools.spending_trend_calculator import (
    CategorySpike,
    average_monthly_total,
    find_category_spikes,
)


def month(**amounts: str) -> dict[str, Any]:
    """One month's expense summary from category=amount pairs."""
    breakdown = [
        {"category": category, "label": f"หมวด-{category}", "amount": amount}
        for category, amount in amounts.items()
    ]
    total = sum((Decimal(amount) for amount in amounts.values()), Decimal("0"))
    return {"total_amount": str(total), "category_breakdown": breakdown}


class TestFindCategorySpikes:
    """A category is a spike when well above the user's own recent average."""

    def test_flags_category_well_above_average(self) -> None:
        """Food 6,000 vs a 4,000 average is +50% and +2,000 THB."""
        history = [month(food="4000"), month(food="3800"), month(food="4200")]

        spikes = find_category_spikes(month(food="6000", fun="100"), history)

        assert spikes == [
            CategorySpike(
                category="food",
                label="หมวด-food",
                current=Decimal("6000"),
                average=Decimal("4000.00"),
                increase_pct=50,
                months_compared=3,
            )
        ]

    def test_small_or_stable_changes_are_not_spikes(self) -> None:
        """+20%, and +40% of a tiny amount (< 500 THB), are not flagged."""
        history = [month(food="5000", fun="300"), month(food="5000", fun="300")]

        assert not find_category_spikes(month(food="6000", fun="420"), history)

    def test_needs_two_months_with_spending(self) -> None:
        """Empty months do not count toward the baseline."""
        history = [month(food="3000"), month()]

        assert not find_category_spikes(month(food="9000"), history)

    def test_new_category_without_history_is_not_a_spike(self) -> None:
        """A category never seen before has no average to compare with."""
        history = [month(food="3000"), month(food="3000")]

        assert not find_category_spikes(month(food="3000", travel="9000"), history)


class TestAverageMonthlyTotal:
    """Average spending of the months that had any spending."""

    def test_skips_empty_months(self) -> None:
        """Months with no spending are not averaged in."""
        summaries = [month(food="3000"), month(), month(food="4001")]

        assert average_monthly_total(summaries) == Decimal("3500.50")

    def test_no_spending_is_zero(self) -> None:
        """No data → 0.00, never a division error."""
        assert average_monthly_total([]) == Decimal("0.00")
        assert average_monthly_total([month()]) == Decimal("0.00")
