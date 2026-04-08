"""Expense tracking constants for category validation and display.

Contains expense categories with Thai labels, validation limits,
and default transaction type for expense records.
"""

from decimal import Decimal

# Expense categories: key -> Thai display label
EXPENSE_CATEGORIES: dict[str, str] = {
    "food": "อาหาร",
    "transport": "การเดินทาง",
    "housing": "ที่พักอาศัย",
    "entertainment": "บันเทิง",
    "utilities": "สาธารณูปโภค",
    "health": "สุขภาพ",
    "education": "การศึกษา",
    "shopping": "ช้อปปิ้ง",
    "investment": "ลงทุน",
    "other": "อื่นๆ",
}

# Valid category keys for validation
VALID_EXPENSE_CATEGORIES: list[str] = list(EXPENSE_CATEGORIES.keys())

# Transaction type value for expense records
EXPENSE_TRANSACTION_TYPE: str = "expense"

# Maximum single expense amount (fraud/input protection)
MAX_SINGLE_EXPENSE_AMOUNT: Decimal = Decimal("10000000.00")

# Minimum expense amount (must be positive)
MIN_EXPENSE_AMOUNT: Decimal = Decimal("0.01")
