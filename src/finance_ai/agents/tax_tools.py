"""LangGraph tool wrappers for Thai personal income tax calculation."""

from decimal import Decimal, InvalidOperation
from typing import Any

from langchain_core.tools import tool

from finance_ai.tools.tax_calculator import TaxCalculationResult, calculate_tax


def parse_decimal_value(value: Any, field_name: str) -> Decimal:
    """Parse a value into Decimal, raising clear errors on failure.

    Args:
        value: The value to parse (str, int, float, Decimal).
        field_name: Name of the field for error messages.

    Returns:
        Parsed Decimal value.

    Raises:
        ValueError: If the value cannot be parsed.

    Example:
        >>> parse_decimal_value("1200000", "gross_income")
        Decimal('1200000')
    """
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Cannot parse {field_name} as Decimal. Received: {value}") from exc


def parse_deductions_input(
    raw_deductions: dict[str, Any],
) -> dict[str, Decimal]:
    """Parse raw deduction dict from LLM into typed Decimal dict.

    Args:
        raw_deductions: Deductions as received from LLM (string values).

    Returns:
        Dict mapping deduction type to Decimal amount.

    Example:
        >>> parse_deductions_input({"rmf": "100000"})
        {'rmf': Decimal('100000')}
    """
    return {key: parse_decimal_value(value, key) for key, value in raw_deductions.items()}


@tool
def calculate_thai_tax(
    gross_income: str | int | float,
    deductions_by_type: dict[str, str | int | float],
    withholding_tax_paid: str | int | float = 0,
) -> dict[str, Any]:
    """Calculate Thai personal income tax with full breakdown.

    Use this tool when you need to calculate Thai personal income tax.
    Provide gross income, deductions by type, and optionally withholding tax.
    Numbers and numeric strings are both accepted (e.g. 1200000 or "1200000").

    Args:
        gross_income: Annual gross income in THB (e.g., 1200000).
        deductions_by_type: Deductions by type (e.g., {"personal_allowance": 60000}).
        withholding_tax_paid: Tax already withheld (e.g., 120000). Defaults to 0.

    Returns:
        Dict with tax calculation results including total_tax, effective_tax_rate, etc.
    """
    income = parse_decimal_value(gross_income, "gross_income")
    deductions = parse_deductions_input(raw_deductions=deductions_by_type)
    withholding = parse_decimal_value(withholding_tax_paid, "withholding_tax_paid")
    result: TaxCalculationResult = calculate_tax(income, deductions, withholding)
    return result.model_dump(mode="json")
