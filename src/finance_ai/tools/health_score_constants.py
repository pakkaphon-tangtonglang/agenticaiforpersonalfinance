"""Constants for pillar-based financial health scoring.

Modeled after the FinHealth Score (Financial Health Network): each pillar
is scored 0-100 and the overall score is the mean of known pillars.
Like credit scoring's "minimum scoring criteria", no overall score is
produced until the minimum data exists.
"""

from decimal import Decimal

# Pillar keys -> Thai labels
HEALTH_PILLARS: dict[str, str] = {
    "spend": "การใช้จ่าย",
    "save": "การออม",
    "invest": "การลงทุน",
    "plan": "การวางแผน",
}

# Score status values
HEALTH_STATUS_SCORED: str = "scored"
HEALTH_STATUS_INSUFFICIENT: str = "insufficient_data"

# Minimum-data gate: missing item labels (Thai)
MISSING_MONTHLY_INCOME: str = "รายรับเดือนนี้"
MISSING_MONTHLY_EXPENSE: str = "รายจ่ายเดือนนี้"

# Spend pillar: expense/income ratio at best (100) and worst (0).
# Source: the 50/30/20 rule (Warren & Tyagi, 2005, "All Your Worth") caps
# needs at 50% of income; spending at or below half of income scores 100.
SPEND_RATIO_BEST: Decimal = Decimal("0.50")
# Project-defined: 120% marks spending a fifth more than income (a clear
# deficit funded by savings or debt) as the worst case; no cited standard.
SPEND_RATIO_WORST: Decimal = Decimal("1.20")

# Save pillar: savings rate at best (100) and worst (0).
# Source: the 50/30/20 rule (Warren & Tyagi, 2005) sets saving at 20% of income.
SAVINGS_RATE_BEST: Decimal = Decimal("0.20")
# Project-defined: saving nothing is the natural zero point.
SAVINGS_RATE_WORST: Decimal = Decimal("0")

# Invest pillar: share of the portfolio in the largest single stock.
# Pooled funds (mutual funds, ETFs) are already diversified, so they never
# count as a concentrated position (see score_invest_pillar).
DIVERSIFIED_ASSET_TYPES: frozenset[str] = frozenset({"mutual_fund", "etf"})
# Project-defined: up to 30% in one stock is treated as diversified for a
# retail portfolio; no single cited standard fits individual investors.
INVEST_CONCENTRATION_BEST: Decimal = Decimal("0.30")
# Project-defined: everything in one stock is the fully concentrated case.
INVEST_CONCENTRATION_WORST: Decimal = Decimal("1.00")
# Project-defined: holding investments at all earns a floor of 30, so a
# concentrated portfolio still scores above someone with no plan at all.
INVEST_CONCENTRATED_FLOOR: int = 30

# Plan pillar: tax filing sub-indicator scores
TAX_NOT_FOUND_STATUS: str = "not_found"
TAX_DRAFT_STATUS: str = "draft"
TAX_FILED_SCORE: int = 100
# Project-defined: a saved draft is half-way between not started and filed.
TAX_DRAFT_SCORE: int = 50
TAX_NOT_FILED_SCORE: int = 0

# Score bounds
SCORE_MAX: int = 100
SCORE_MIN: int = 0
