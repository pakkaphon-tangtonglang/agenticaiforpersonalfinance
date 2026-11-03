"""Thai tax law constants (brackets unchanged since 2560; limits as of tax year 2569).

Contains progressive tax brackets, deduction limits, and percentage caps
per Thai Revenue Department regulations.
"""

from decimal import Decimal
from typing import Optional

# Thai Progressive Tax Brackets (unchanged since tax year 2560)
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
    "social_security": Decimal("10500"),
    "life_insurance": Decimal("100000"),
    "health_insurance": Decimal("25000"),
    "provident_fund": Decimal("500000"),
    "rmf": Decimal("500000"),
    "ssf": Decimal("200000"),
    "mortgage_interest": Decimal("100000"),
}

# Allowances claimed per person: DEDUCTION_LIMITS holds the amount per person,
# so the cap is that amount x the most people allowed (None = no limit on people;
# a second child born from 2018 counts 60,000, so children are not capped here).
MAX_PERSONS_PER_ALLOWANCE: dict[str, Optional[int]] = {
    "child_allowance": None,
    "parent_allowance": 4,  # own parents + spouse's parents
}

# Deduction types that have percentage-of-income caps
# Format: {deduction_type: max_percentage_of_gross_income}
PERCENTAGE_CAPPED_DEDUCTIONS: dict[str, Decimal] = {
    "provident_fund": Decimal("0.15"),
    "rmf": Decimal("0.30"),
    "ssf": Decimal("0.30"),
}

# Employee social security contribution (Section 33): 5% of wage up to the ceiling.
# Ceiling is 17,500 from 1 Jan 2569 (875/month, 10,500/year; docs/all_known_problem.md 3.3).
SOCIAL_SECURITY_RATE: Decimal = Decimal("0.05")
SOCIAL_SECURITY_WAGE_CEILING: Decimal = Decimal("17500")

# SSF purchases are deductible only up to tax year 2567 (2024).
SSF_FINAL_TAX_YEAR: int = 2024

# Standard expense deduction for employment income (40(1))
# 50% of gross income, capped at 100,000 THB
EXPENSE_DEDUCTION_RATE: Decimal = Decimal("0.50")
EXPENSE_DEDUCTION_MAX: Decimal = Decimal("100000")

# Donation cap: max 10% of net income (after all other deductions)
DONATION_CAP_PERCENTAGE: Decimal = Decimal("0.10")

# Number of tax brackets
NUMBER_OF_BRACKETS: int = len(TAX_BRACKETS)
