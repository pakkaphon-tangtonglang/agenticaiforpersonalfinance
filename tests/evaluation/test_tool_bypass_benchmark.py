"""Tests for tool-bypass benchmark helper functions."""

from decimal import Decimal

from finance_ai.evaluation.tool_bypass_benchmark import (
    BYPASS_TEST_CASES,
    _compute_tool_tax,
)


class TestToolBypassBenchmark:
    """Tests for tool-bypass benchmark helpers."""

    def test_compute_tool_tax_matches_expected(self) -> None:
        """_compute_tool_tax returns expected tax for known cases."""
        case = BYPASS_TEST_CASES[0]
        tax = _compute_tool_tax(case)
        assert abs(tax - case["expected_tax"]) <= case["tolerance"]

    def test_compute_tool_tax_low_income(self) -> None:
        """_compute_tool_tax handles low-income case."""
        case = {
            "gross_income": Decimal("360000"),
            "deductions": {
                "personal_allowance": Decimal("60000"),
                "social_security": Decimal("9000"),
            },
        }
        tax = _compute_tool_tax(case)
        assert tax == Decimal("2050")
