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

# Spend pillar: expense/income ratio at best (100) and worst (0)
SPEND_RATIO_BEST: Decimal = Decimal("0.50")
SPEND_RATIO_WORST: Decimal = Decimal("1.20")

# Save pillar: savings rate at best (100) and worst (0)
SAVINGS_RATE_BEST: Decimal = Decimal("0.20")
SAVINGS_RATE_WORST: Decimal = Decimal("0")

# Invest pillar: largest-holding share at best (100) and fully concentrated
INVEST_CONCENTRATION_BEST: Decimal = Decimal("0.30")
INVEST_CONCENTRATION_WORST: Decimal = Decimal("1.00")
INVEST_CONCENTRATED_FLOOR: int = 30
INVEST_LOSS_PENALTY: int = 20

# Plan pillar: tax filing sub-indicator scores
TAX_NOT_FOUND_STATUS: str = "not_found"
TAX_DRAFT_STATUS: str = "draft"
TAX_FILED_SCORE: int = 100
TAX_DRAFT_SCORE: int = 50
TAX_NOT_FILED_SCORE: int = 0

# Score bounds
SCORE_MAX: int = 100
SCORE_MIN: int = 0
