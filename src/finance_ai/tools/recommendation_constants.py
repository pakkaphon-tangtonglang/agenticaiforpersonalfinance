"""Constants for proactive financial recommendations.

Contains recommendation categories, priority levels, and threshold values
for rule-based financial analysis.
"""

from decimal import Decimal

# Recommendation categories: key -> Thai label
RECOMMENDATION_CATEGORIES: dict[str, str] = {
    "expense_optimization": "ลดค่าใช้จ่าย",
    "tax_optimization": "ลดหย่อนภาษี",
    "investment_rebalancing": "ปรับพอร์ตการลงทุน",
    "goal_progress": "ติดตามเป้าหมาย",
    "savings_rate": "อัตราการออม",
    "emergency_fund": "เงินสำรองฉุกเฉิน",
}

# Priority levels: value -> Thai label (1=lowest, 5=critical)
RECOMMENDATION_PRIORITIES: dict[int, str] = {
    1: "ต่ำ",
    2: "ปานกลาง",
    3: "สูง",
    4: "สูงมาก",
    5: "เร่งด่วน",
}

# --- Rule thresholds ---

# Savings rate: recommend saving more if below this ratio
SAVINGS_RATE_WARNING_THRESHOLD: Decimal = Decimal("0.20")

# Overspending: flag if expenses exceed this ratio of income
OVERSPENDING_THRESHOLD: Decimal = Decimal("0.80")

# Emergency fund: recommended months of expenses to cover
EMERGENCY_FUND_MONTHS: int = 6

# Expense concentration: flag if one category exceeds this ratio
EXPENSE_CONCENTRATION_THRESHOLD: Decimal = Decimal("0.40")

# Investment concentration: flag if single holding exceeds this ratio
INVESTMENT_CONCENTRATION_THRESHOLD: Decimal = Decimal("0.30")

# Tax deduction utilization: suggest optimization if below this ratio
TAX_DEDUCTION_UTILIZATION_THRESHOLD: Decimal = Decimal("0.50")

# Goal progress: behind schedule if below this percentage with deadline near
GOAL_BEHIND_SCHEDULE_PERCENTAGE: Decimal = Decimal("50.00")

# Goal deadline warning: flag if deadline is within this many months
GOAL_DEADLINE_WARNING_MONTHS: int = 6

# Health score: base score (100) minus penalty per recommendation
HEALTH_SCORE_BASE: int = 100

# Health score: penalty multiplier per priority level
HEALTH_SCORE_PENALTY_PER_PRIORITY: int = 5
