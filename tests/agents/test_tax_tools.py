"""Tests for LangGraph tax tool wrappers."""

from decimal import Decimal

import pytest

from finance_ai.agents.tax_tools import (
    calculate_thai_tax,
    parse_decimal_value,
    parse_deductions_input,
)


class TestParseDecimalValue:
    """Tests for parse_decimal_value helper."""

    def test_valid_string(self) -> None:
        """Parses a numeric string to Decimal."""
        result = parse_decimal_value("1200000", "gross_income")
        assert result == Decimal("1200000")

    def test_valid_integer(self) -> None:
        """Parses an integer to Decimal."""
        result = parse_decimal_value(1200000, "gross_income")
        assert result == Decimal("1200000")

    def test_valid_float(self) -> None:
        """Parses a float to Decimal (via string conversion)."""
        result = parse_decimal_value(100000.50, "amount")
        assert result == Decimal("100000.50")

    def test_valid_decimal(self) -> None:
        """Passes through an existing Decimal."""
        result = parse_decimal_value(Decimal("60000"), "allowance")
        assert result == Decimal("60000")

    def test_invalid_string_raises(self) -> None:
        """Raises ValueError for non-numeric strings."""
        with pytest.raises(ValueError, match="Cannot parse gross_income"):
            parse_decimal_value("abc", "gross_income")

    def test_empty_string_raises(self) -> None:
        """Raises ValueError for empty strings."""
        with pytest.raises(ValueError, match="Cannot parse amount"):
            parse_decimal_value("", "amount")


class TestParseDeductionsInput:
    """Tests for parse_deductions_input helper."""

    def test_multiple_deduction_types(self) -> None:
        """Parses multiple deduction types correctly."""
        raw = {"personal_allowance": "60000", "rmf": "100000", "child_allowance": "60000"}
        result = parse_deductions_input(raw)
        assert result == {
            "personal_allowance": Decimal("60000"),
            "rmf": Decimal("100000"),
            "child_allowance": Decimal("60000"),
        }

    def test_empty_dict(self) -> None:
        """Returns empty dict for empty input."""
        result = parse_deductions_input({})
        assert result == {}

    def test_single_deduction(self) -> None:
        """Parses a single deduction type."""
        result = parse_deductions_input({"personal_allowance": "60000"})
        assert result == {"personal_allowance": Decimal("60000")}


class TestCalculateThaiTax:
    """Tests for the calculate_thai_tax LangGraph tool."""

    def test_basic_calculation(self) -> None:
        """Calculates tax for a basic income with personal allowance."""
        result = calculate_thai_tax.invoke(
            {
                "gross_income": "1200000",
                "deductions_by_type": {"personal_allowance": "60000"},
            }
        )
        assert isinstance(result, dict)
        assert "total_tax" in result
        assert "effective_tax_rate" in result
        assert "net_income" in result
        assert "expense_deduction" in result

    def test_with_multiple_deductions(self) -> None:
        """Calculates tax with multiple deduction types."""
        result = calculate_thai_tax.invoke(
            {
                "gross_income": "1200000",
                "deductions_by_type": {
                    "personal_allowance": "60000",
                    "child_allowance": "60000",
                    "rmf": "100000",
                },
            }
        )
        assert isinstance(result, dict)
        assert Decimal(str(result["total_deductions"])) > Decimal("60000")

    def test_with_withholding_tax(self) -> None:
        """Includes withholding tax in calculation."""
        result = calculate_thai_tax.invoke(
            {
                "gross_income": "1200000",
                "deductions_by_type": {"personal_allowance": "60000"},
                "withholding_tax_paid": "120000",
            }
        )
        assert isinstance(result, dict)
        assert "tax_due_or_refund" in result

    def test_zero_income(self) -> None:
        """Handles zero income correctly."""
        result = calculate_thai_tax.invoke(
            {
                "gross_income": "0",
                "deductions_by_type": {},
            }
        )
        assert Decimal(str(result["total_tax"])) == Decimal("0")

    def test_negative_income_raises(self) -> None:
        """Raises ValueError for negative income."""
        with pytest.raises(ValueError, match="Income cannot be negative"):
            calculate_thai_tax.invoke(
                {
                    "gross_income": "-100000",
                    "deductions_by_type": {},
                }
            )

    def test_default_withholding_is_zero(self) -> None:
        """Withholding tax defaults to zero when not provided."""
        result = calculate_thai_tax.invoke(
            {
                "gross_income": "1200000",
                "deductions_by_type": {"personal_allowance": "60000"},
            }
        )
        assert Decimal(str(result["withholding_tax_paid"])) == Decimal("0")

    def test_result_has_tax_breakdown(self) -> None:
        """Result includes tax breakdown by bracket."""
        result = calculate_thai_tax.invoke(
            {
                "gross_income": "1200000",
                "deductions_by_type": {"personal_allowance": "60000"},
            }
        )
        assert "tax_breakdown" in result
        assert isinstance(result["tax_breakdown"], list)
        assert len(result["tax_breakdown"]) > 0
