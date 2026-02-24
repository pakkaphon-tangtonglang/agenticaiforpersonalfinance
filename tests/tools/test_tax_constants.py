"""Tests for Thai tax law constants."""

from decimal import Decimal

from finance_ai.tools.tax_constants import (
    DEDUCTION_LIMITS,
    DONATION_CAP_PERCENTAGE,
    NUMBER_OF_BRACKETS,
    PERCENTAGE_CAPPED_DEDUCTIONS,
    TAX_BRACKETS,
)


class TestTaxBrackets:
    """Tests for Thai progressive tax bracket constants."""

    def test_number_of_brackets(self) -> None:
        """Test that there are exactly 8 Thai tax brackets."""
        assert NUMBER_OF_BRACKETS == 8

    def test_first_bracket_is_tax_free(self) -> None:
        """Test that income up to 150,000 THB is tax-free."""
        start, end, rate = TAX_BRACKETS[0]
        assert start == Decimal("0")
        assert end == Decimal("150000")
        assert rate == Decimal("0.00")

    def test_top_bracket_rate(self) -> None:
        """Test that the top bracket rate is 35%."""
        _, _, rate = TAX_BRACKETS[-1]
        assert rate == Decimal("0.35")

    def test_brackets_are_contiguous(self) -> None:
        """Test that brackets cover all income ranges without gaps."""
        for i in range(1, len(TAX_BRACKETS)):
            prev_end = TAX_BRACKETS[i - 1][1]
            curr_start = TAX_BRACKETS[i][0]
            assert curr_start == prev_end + Decimal("1")

    def test_brackets_rates_are_ascending(self) -> None:
        """Test that tax rates increase with each bracket."""
        for i in range(1, len(TAX_BRACKETS)):
            assert TAX_BRACKETS[i][2] > TAX_BRACKETS[i - 1][2]


class TestDeductionLimits:
    """Tests for deduction limit constants."""

    def test_personal_allowance_limit(self) -> None:
        """Test personal allowance is 60,000 THB."""
        assert DEDUCTION_LIMITS["personal_allowance"] == Decimal("60000")

    def test_social_security_limit(self) -> None:
        """Test social security limit is 9,000 THB."""
        assert DEDUCTION_LIMITS["social_security"] == Decimal("9000")

    def test_rmf_absolute_limit(self) -> None:
        """Test RMF absolute limit is 500,000 THB."""
        assert DEDUCTION_LIMITS["rmf"] == Decimal("500000")

    def test_ssf_absolute_limit(self) -> None:
        """Test SSF absolute limit is 200,000 THB."""
        assert DEDUCTION_LIMITS["ssf"] == Decimal("200000")

    def test_all_deduction_types_present(self) -> None:
        """Test that all 11 deduction types are defined."""
        expected_types = {
            "personal_allowance",
            "spouse_allowance",
            "child_allowance",
            "parent_allowance",
            "social_security",
            "life_insurance",
            "health_insurance",
            "provident_fund",
            "rmf",
            "ssf",
            "mortgage_interest",
        }
        assert set(DEDUCTION_LIMITS.keys()) == expected_types

    def test_percentage_capped_deductions(self) -> None:
        """Test that RMF, SSF, and PF have percentage caps."""
        assert "rmf" in PERCENTAGE_CAPPED_DEDUCTIONS
        assert "ssf" in PERCENTAGE_CAPPED_DEDUCTIONS
        assert "provident_fund" in PERCENTAGE_CAPPED_DEDUCTIONS
        assert PERCENTAGE_CAPPED_DEDUCTIONS["rmf"] == Decimal("0.30")
        assert PERCENTAGE_CAPPED_DEDUCTIONS["provident_fund"] == Decimal("0.15")

    def test_donation_cap_percentage(self) -> None:
        """Test donation cap is 10% of net income."""
        assert DONATION_CAP_PERCENTAGE == Decimal("0.10")
