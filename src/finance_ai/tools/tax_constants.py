"""Thai tax law constants for tax year 2024.

Contains progressive tax brackets, deduction limits, and percentage caps
per Thai Revenue Department regulations.
"""

from decimal import Decimal

# Thai Progressive Tax Brackets (2024)
# Format: (bracket_start, bracket_end, tax_rate)
TAX_BRACKETS: list[tuple[Decimal, Decimal, Decimal]] = [
    (Decimal("0"), Decimal("150000"), Decimal("0.00")),
    (Decimal("150001"), Decimal("300000"), Decimal("0.05")),
    (Decimal("300001"), Decimal("500000"), Decimal("0.10")),
    (Decimal("500001"), Decimal("750000"), Decimal("0.15")),
    (Decimal("750001"), Decimal("1000000"), Decimal("0.20")),
    (Decimal("1000001"), Decimal("2000000"), Decimal("0.25")),
    (Decimal("2000001"), Decimal("5000000"), Decimal("0.30")),
    (Decimal("5000001"), Decimal("99999999999"), Decimal("0.35")),
]

# Maximum deduction amounts per type (THB)
DEDUCTION_LIMITS: dict[str, Decimal] = {
    "personal_allowance": Decimal("60000"),
    "spouse_allowance": Decimal("60000"),
    "child_allowance": Decimal("30000"),
    "parent_allowance": Decimal("30000"),
    "social_security": Decimal("9000"),
    "life_insurance": Decimal("100000"),
    "health_insurance": Decimal("25000"),
    "provident_fund": Decimal("500000"),
    "rmf": Decimal("500000"),
    "ssf": Decimal("200000"),
    "mortgage_interest": Decimal("100000"),
}

# Deduction types that have percentage-of-income caps
# Format: {deduction_type: max_percentage_of_gross_income}
PERCENTAGE_CAPPED_DEDUCTIONS: dict[str, Decimal] = {
    "provident_fund": Decimal("0.15"),
    "rmf": Decimal("0.30"),
    "ssf": Decimal("0.30"),
}

# Standard expense deduction for employment income (40(1))
# 50% of gross income, capped at 100,000 THB
EXPENSE_DEDUCTION_RATE: Decimal = Decimal("0.50")
EXPENSE_DEDUCTION_MAX: Decimal = Decimal("100000")

# Donation cap: max 10% of net income (after all other deductions)
DONATION_CAP_PERCENTAGE: Decimal = Decimal("0.10")

# Number of tax brackets
NUMBER_OF_BRACKETS: int = len(TAX_BRACKETS)
