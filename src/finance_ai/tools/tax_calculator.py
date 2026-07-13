"""Pure tax calculation functions for Thai personal income tax.

All functions accept Decimal values directly with no database dependency,
making them easy to test and reuse across different contexts.
"""

from decimal import Decimal

from pydantic import BaseModel

from finance_ai.tools.tax_constants import (
    DEDUCTION_LIMITS,
    DONATION_CAP_PERCENTAGE,
    EXPENSE_DEDUCTION_MAX,
    EXPENSE_DEDUCTION_RATE,
    PERCENTAGE_CAPPED_DEDUCTIONS,
    TAX_BRACKETS,
)


class TaxBracketResult(BaseModel):
    """Tax calculation result for a single bracket."""

    bracket_start: Decimal
    bracket_end: Decimal
    rate: Decimal
    taxable_in_bracket: Decimal
    tax_in_bracket: Decimal


class TaxCalculationResult(BaseModel):
    """Complete tax calculation result with breakdown."""

    gross_income: Decimal
    expense_deduction: Decimal
    total_deductions: Decimal
    net_income: Decimal
    tax_breakdown: list[TaxBracketResult]
    total_tax: Decimal
    effective_tax_rate: Decimal
    withholding_tax_paid: Decimal
    tax_due_or_refund: Decimal


def validate_non_negative_income(gross_income: Decimal) -> None:
    """
    Validate that gross income is not negative.

    Args:
        gross_income: Income amount to validate.

    Raises:
        ValueError: If income is negative.

    Example:
        >>> validate_non_negative_income(Decimal("100000"))
    """
    if gross_income < Decimal("0"):
        raise ValueError(f"Income cannot be negative. Received: {gross_income} THB")


def calculate_bracket_tax(
    net_income: Decimal,
    bracket_start: Decimal,
    bracket_end: Decimal,
    rate: Decimal,
) -> TaxBracketResult:
    """
    Calculate tax for a single bracket.

    Args:
        net_income: Total net income.
        bracket_start: Lower bound of this bracket.
        bracket_end: Upper bound of this bracket.
        rate: Tax rate for this bracket.

    Returns:
        TaxBracketResult with taxable amount and tax for this bracket.

    Example:
        >>> calculate_bracket_tax(Decimal("200000"), Decimal("150001"), Decimal("300000"), Decimal("0.05"))
    """
    if net_income < bracket_start:
        taxable = Decimal("0")
    elif net_income >= bracket_end:
        taxable = bracket_end - bracket_start + Decimal("1")
    else:
        taxable = net_income - bracket_start + Decimal("1")
    return TaxBracketResult(
        bracket_start=bracket_start,
        bracket_end=bracket_end,
        rate=rate,
        taxable_in_bracket=taxable,
        tax_in_bracket=taxable * rate,
    )


def calculate_progressive_tax(net_income: Decimal) -> list[TaxBracketResult]:
    """
    Calculate tax across all Thai progressive brackets.

    Args:
        net_income: Net income after deductions.

    Returns:
        List of TaxBracketResult for each bracket.

    Example:
        >>> results = calculate_progressive_tax(Decimal("500000"))
        >>> len(results)
        8
    """
    return [
        calculate_bracket_tax(net_income, start, end, rate) for start, end, rate in TAX_BRACKETS
    ]


def calculate_total_tax_from_breakdown(
    breakdown: list[TaxBracketResult],
) -> Decimal:
    """
    Sum total tax from bracket breakdown.

    Args:
        breakdown: List of bracket results.

    Returns:
        Total tax amount.

    Example:
        >>> calculate_total_tax_from_breakdown([])
        Decimal('0')
    """
    return sum((result.tax_in_bracket for result in breakdown), Decimal("0"))


def cap_deduction_amount(
    deduction_type: str,
    claimed_amount: Decimal,
    gross_income: Decimal,
) -> Decimal:
    """
    Apply legal cap to a deduction amount.

    Args:
        deduction_type: Type of deduction.
        claimed_amount: Amount the user is claiming.
        gross_income: Gross income for percentage-based caps.

    Returns:
        Capped deduction amount (never exceeds legal limit).

    Example:
        >>> cap_deduction_amount("rmf", Decimal("600000"), Decimal("1000000"))
        Decimal('300000')
    """
    absolute_limit = DEDUCTION_LIMITS.get(deduction_type, claimed_amount)
    capped = min(claimed_amount, absolute_limit)
    if deduction_type in PERCENTAGE_CAPPED_DEDUCTIONS:
        percentage_limit = gross_income * PERCENTAGE_CAPPED_DEDUCTIONS[deduction_type]
        capped = min(capped, percentage_limit)
    return capped


def calculate_donation_cap(
    claimed_donation: Decimal,
    net_income_before_donations: Decimal,
) -> Decimal:
    """
    Cap donation deduction at 10% of net income before donations.

    Args:
        claimed_donation: Donation amount claimed.
        net_income_before_donations: Net income excluding donation deduction.

    Returns:
        Capped donation amount.

    Example:
        >>> calculate_donation_cap(Decimal("100000"), Decimal("500000"))
        Decimal('50000.00')
    """
    max_donation = net_income_before_donations * DONATION_CAP_PERCENTAGE
    return min(claimed_donation, max_donation)


def sum_capped_deductions(
    deductions_by_type: dict[str, Decimal],
    gross_income: Decimal,
) -> tuple[Decimal, Decimal]:
    """
    Sum all deductions after applying caps, handling donations separately.

    Args:
        deductions_by_type: Mapping of deduction type to claimed amount.
        gross_income: Gross income for percentage caps.

    Returns:
        Tuple of (total_deductions, donation_amount_after_cap).

    Example:
        >>> sum_capped_deductions({"personal_allowance": Decimal("60000")}, Decimal("1000000"))
    """
    non_donation_total = Decimal("0")
    claimed_donation = Decimal("0")
    for deduction_type, amount in deductions_by_type.items():
        if deduction_type == "donations":
            claimed_donation = amount
            continue
        non_donation_total += cap_deduction_amount(deduction_type, amount, gross_income)
    net_before_donations = gross_income - non_donation_total
    capped_donation = calculate_donation_cap(claimed_donation, net_before_donations)
    return non_donation_total + capped_donation, capped_donation


def calculate_effective_rate(total_tax: Decimal, gross_income: Decimal) -> Decimal:
    """
    Calculate effective tax rate.

    Args:
        total_tax: Total tax payable.
        gross_income: Total gross income.

    Returns:
        Effective rate as decimal (e.g., 0.0750 for 7.5%).

    Example:
        >>> calculate_effective_rate(Decimal("75000"), Decimal("1000000"))
        Decimal('0.0750')
    """
    if gross_income == Decimal("0"):
        return Decimal("0.0000")
    return (total_tax / gross_income).quantize(Decimal("0.0001"))


def calculate_expense_deduction(gross_income: Decimal) -> Decimal:
    """Calculate standard expense deduction (50% of income, max 100,000 THB).

    Args:
        gross_income: Total gross income for the tax year.

    Returns:
        Expense deduction amount (capped at 100,000 THB).

    Example:
        >>> calculate_expense_deduction(Decimal("600000"))
        Decimal('100000')
    """
    return min(gross_income * EXPENSE_DEDUCTION_RATE, EXPENSE_DEDUCTION_MAX)


def calculate_tax(
    gross_income: Decimal,
    deductions_by_type: dict[str, Decimal],
    withholding_tax_paid: Decimal = Decimal("0"),
) -> TaxCalculationResult:
    """
    Calculate Thai personal income tax with full breakdown.

    Args:
        gross_income: Total gross income for the tax year.
        deductions_by_type: Mapping of deduction type to claimed amount.
        withholding_tax_paid: Tax already withheld during the year.

    Returns:
        TaxCalculationResult with complete breakdown.

    Raises:
        ValueError: If gross_income is negative.

    Example:
        >>> result = calculate_tax(Decimal("960000"), {"personal_allowance": Decimal("60000")})
        >>> result.total_tax > Decimal("0")
        True
    """
    validate_non_negative_income(gross_income)
    expense_deduction = calculate_expense_deduction(gross_income)
    total_deductions, _ = sum_capped_deductions(deductions_by_type, gross_income)
    net_income = max(gross_income - expense_deduction - total_deductions, Decimal("0"))
    breakdown = calculate_progressive_tax(net_income)
    total_tax = calculate_total_tax_from_breakdown(breakdown)
    effective_rate = calculate_effective_rate(total_tax, gross_income)
    tax_due_or_refund = total_tax - withholding_tax_paid
    return TaxCalculationResult(
        gross_income=gross_income,
        expense_deduction=expense_deduction,
        total_deductions=total_deductions,
        net_income=net_income,
        tax_breakdown=breakdown,
        total_tax=total_tax,
        effective_tax_rate=effective_rate,
        withholding_tax_paid=withholding_tax_paid,
        tax_due_or_refund=tax_due_or_refund,
    )
