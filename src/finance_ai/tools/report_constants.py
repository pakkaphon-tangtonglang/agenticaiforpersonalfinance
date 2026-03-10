"""Constants for the autonomous financial report agent.

Defines report section labels, report types, highlight classifications,
and thresholds for generating report highlights.
"""

from decimal import Decimal

# Report section keys mapped to Thai display labels
REPORT_SECTIONS: dict[str, str] = {
    "monthly_overview": "ภาพรวมรายเดือน",
    "expense_breakdown": "รายละเอียดค่าใช้จ่าย",
    "investment_portfolio": "พอร์ตการลงทุน",
    "goal_progress": "ความคืบหน้าเป้าหมาย",
    "tax_status": "สถานะภาษี",
    "health_score": "คะแนนสุขภาพการเงิน",
    "highlights": "จุดเด่นและข้อสังเกต",
}

# Supported report types
REPORT_TYPES: dict[str, str] = {
    "monthly": "รายงานประจำเดือน",
    "annual": "รายงานประจำปี",
}

# Highlight classification types
HIGHLIGHT_TYPES: dict[str, str] = {
    "achievement": "ความสำเร็จ",
    "warning": "คำเตือน",
    "info": "ข้อมูล",
}

# Savings rate >= 20% is considered good
SAVINGS_RATE_GOOD_THRESHOLD: Decimal = Decimal("0.20")

# Goals with >= 80% progress are highlighted as near-complete
GOAL_NEAR_COMPLETE_THRESHOLD: Decimal = Decimal("80.00")

# Number of months to annualize monthly income
MONTHS_PER_YEAR: int = 12

# Default report type when not specified
DEFAULT_REPORT_TYPE: str = "monthly"
