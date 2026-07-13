"""Tests for Thai tax calculator pure functions."""

from decimal import Decimal

import pytest

from finance_ai.tools.tax_calculator import (
    TaxBracketResult,
    TaxCalculationResult,
    calculate_bracket_tax,
    calculate_donation_cap,
    calculate_effective_rate,
    calculate_expense_deduction,
    calculate_progressive_tax,
    calculate_tax,
    calculate_total_tax_from_breakdown,
    cap_deduction_amount,
    sum_capped_deductions,
    validate_non_negative_income,
)


class TestValidation:
    """Tests for input validation."""

    def test_negative_income_raises_error(self) -> None:
        """Test that negative income raises ValueError."""
        with pytest.raises(ValueError, match="Income cannot be negative"):
            validate_non_negative_income(Decimal("-50000"))

    def test_zero_income_is_valid(self) -> None:
        """Test that zero income does not raise."""
        validate_non_negative_income(Decimal("0"))

    def test_positive_income_is_valid(self) -> None:
        """Test that positive income does not raise."""
        validate_non_negative_income(Decimal("1000000"))


class TestBracketTax:
    """Tests for single bracket tax calculation."""

    def test_income_below_bracket(self) -> None:
        """Test zero tax when income is below bracket start."""
        result = calculate_bracket_tax(
            Decimal("100000"), Decimal("150001"), Decimal("300000"), Decimal("0.05")
        )
        assert result.taxable_in_bracket == Decimal("0")
        assert result.tax_in_bracket == Decimal("0")

    def test_income_within_bracket(self) -> None:
        """Test partial tax when income falls within bracket."""
        result = calculate_bracket_tax(
            Decimal("200000"), Decimal("150001"), Decimal("300000"), Decimal("0.05")
        )
        assert result.taxable_in_bracket == Decimal("50000")
        assert result.tax_in_bracket == Decimal("2500.00")

    def test_income_exceeds_bracket(self) -> None:
        """Test full bracket tax when income exceeds bracket end."""
        result = calculate_bracket_tax(
            Decimal("500000"), Decimal("150001"), Decimal("300000"), Decimal("0.05")
        )
        assert result.taxable_in_bracket == Decimal("150000")
        assert result.tax_in_bracket == Decimal("7500.00")


class TestProgressiveTax:
    """Tests for progressive tax calculation across all brackets."""

    def test_zero_income(self) -> None:
        """Test that zero income produces zero tax in all brackets."""
        results = calculate_progressive_tax(Decimal("0"))
        total = calculate_total_tax_from_breakdown(results)
        assert total == Decimal("0")

    def test_income_under_150k_tax_free(self) -> None:
        """Test that income under 150,000 is entirely tax-free."""
        results = calculate_progressive_tax(Decimal("150000"))
        total = calculate_total_tax_from_breakdown(results)
        assert total == Decimal("0")

    def test_income_at_300k(self) -> None:
        """Test tax at 300,000 (fills first two brackets)."""
        results = calculate_progressive_tax(Decimal("300000"))
        total = calculate_total_tax_from_breakdown(results)
        assert total == Decimal("7500.00")

    def test_income_at_500k(self) -> None:
        """Test tax at 500,000 (fills first three brackets)."""
        results = calculate_progressive_tax(Decimal("500000"))
        total = calculate_total_tax_from_breakdown(results)
        expected = Decimal("7500") + Decimal("20000")
        assert total == expected

    def test_income_at_750k(self) -> None:
        """Test tax at 750,000."""
        results = calculate_progressive_tax(Decimal("750000"))
        total = calculate_total_tax_from_breakdown(results)
        expected = Decimal("7500") + Decimal("20000") + Decimal("37500")
        assert total == expected

    def test_income_at_1m(self) -> None:
        """Test tax at 1,000,000."""
        results = calculate_progressive_tax(Decimal("1000000"))
        total = calculate_total_tax_from_breakdown(results)
        expected = Decimal("7500") + Decimal("20000") + Decimal("37500") + Decimal("50000")
        assert total == expected

    def test_income_at_2m(self) -> None:
        """Test tax at 2,000,000."""
        results = calculate_progressive_tax(Decimal("2000000"))
        total = calculate_total_tax_from_breakdown(results)
        expected = (
            Decimal("7500")
            + Decimal("20000")
            + Decimal("37500")
            + Decimal("50000")
            + Decimal("250000")
        )
        assert total == expected

    def test_income_at_5m(self) -> None:
        """Test tax at 5,000,000."""
        results = calculate_progressive_tax(Decimal("5000000"))
        total = calculate_total_tax_from_breakdown(results)
        expected = (
            Decimal("7500")
            + Decimal("20000")
            + Decimal("37500")
            + Decimal("50000")
            + Decimal("250000")
            + Decimal("900000")
        )
        assert total == expected

    def test_income_at_10m_top_bracket(self) -> None:
        """Test tax at 10,000,000 (top bracket applies)."""
        results = calculate_progressive_tax(Decimal("10000000"))
        total = calculate_total_tax_from_breakdown(results)
        brackets_up_to_5m = (
            Decimal("7500")
            + Decimal("20000")
            + Decimal("37500")
            + Decimal("50000")
            + Decimal("250000")
            + Decimal("900000")
        )
        top_bracket_tax = Decimal("5000000") * Decimal("0.35")
        assert total == brackets_up_to_5m + top_bracket_tax

    def test_returns_eight_brackets(self) -> None:
        """Test that result always contains 8 bracket entries."""
        results = calculate_progressive_tax(Decimal("500000"))
        assert len(results) == 8


class TestDeductionCaps:
    """Tests for deduction capping logic."""

    def test_personal_allowance_under_limit(self) -> None:
        """Test that amount under limit passes through."""
        result = cap_deduction_amount("personal_allowance", Decimal("50000"), Decimal("1000000"))
        assert result == Decimal("50000")

    def test_personal_allowance_over_limit(self) -> None:
        """Test that amount over limit is capped."""
        result = cap_deduction_amount("personal_allowance", Decimal("100000"), Decimal("1000000"))
        assert result == Decimal("60000")

    def test_rmf_absolute_cap(self) -> None:
        """Test RMF capped at 500,000 when percentage allows more."""
        result = cap_deduction_amount("rmf", Decimal("600000"), Decimal("5000000"))
        assert result == Decimal("500000")

    def test_rmf_percentage_cap(self) -> None:
        """Test RMF capped at 30% of salary when lower than 500k."""
        result = cap_deduction_amount("rmf", Decimal("500000"), Decimal("1000000"))
        assert result == Decimal("300000")

    def test_ssf_percentage_cap(self) -> None:
        """Test SSF capped at 30% of salary when lower than 200k."""
        result = cap_deduction_amount("ssf", Decimal("200000"), Decimal("500000"))
        assert result == Decimal("150000")

    def test_provident_fund_percentage_cap(self) -> None:
        """Test provident fund capped at 15% of salary."""
        result = cap_deduction_amount("provident_fund", Decimal("500000"), Decimal("1000000"))
        assert result == Decimal("150000")

    def test_unknown_deduction_type_passthrough(self) -> None:
        """Test that unknown deduction types pass through uncapped."""
        result = cap_deduction_amount("other", Decimal("50000"), Decimal("1000000"))
        assert result == Decimal("50000")


class TestDonationCap:
    """Tests for donation cap calculation."""

    def test_donation_under_cap(self) -> None:
        """Test donation under 10% passes through."""
        result = calculate_donation_cap(Decimal("30000"), Decimal("500000"))
        assert result == Decimal("30000")

    def test_donation_over_cap(self) -> None:
        """Test donation over 10% is capped."""
        result = calculate_donation_cap(Decimal("100000"), Decimal("500000"))
        assert result == Decimal("50000.00")

    def test_donation_exactly_at_cap(self) -> None:
        """Test donation exactly at 10%."""
        result = calculate_donation_cap(Decimal("50000"), Decimal("500000"))
        assert result == Decimal("50000")

    def test_zero_net_income_caps_donation(self) -> None:
        """Test that zero net income caps donation to zero."""
        result = calculate_donation_cap(Decimal("10000"), Decimal("0"))
        assert result == Decimal("0")


class TestSumCappedDeductions:
    """Tests for aggregating capped deductions."""

    def test_single_deduction(self) -> None:
        """Test summing a single deduction."""
        total, _ = sum_capped_deductions(
            {"personal_allowance": Decimal("60000")}, Decimal("1000000")
        )
        assert total == Decimal("60000")

    def test_multiple_deductions(self) -> None:
        """Test summing multiple deductions."""
        deductions = {
            "personal_allowance": Decimal("60000"),
            "social_security": Decimal("9000"),
        }
        total, _ = sum_capped_deductions(deductions, Decimal("1000000"))
        assert total == Decimal("69000")

    def test_with_donation(self) -> None:
        """Test that donations are capped based on net after other deductions."""
        deductions = {
            "personal_allowance": Decimal("60000"),
            "donations": Decimal("200000"),
        }
        total, donation = sum_capped_deductions(deductions, Decimal("1000000"))
        expected_donation = Decimal("94000.0")
        assert donation == expected_donation
        assert total == Decimal("60000") + expected_donation


class TestEffectiveRate:
    """Tests for effective tax rate calculation."""

    def test_typical_rate(self) -> None:
        """Test effective rate calculation."""
        rate = calculate_effective_rate(Decimal("75000"), Decimal("1000000"))
        assert rate == Decimal("0.0750")

    def test_zero_income_rate(self) -> None:
        """Test that zero income gives zero rate."""
        rate = calculate_effective_rate(Decimal("0"), Decimal("0"))
        assert rate == Decimal("0.0000")


class TestCalculateTax:
    """Tests for the main calculate_tax function."""

    def test_negative_income_raises(self) -> None:
        """Test that negative income raises ValueError."""
        with pytest.raises(ValueError, match="Income cannot be negative"):
            calculate_tax(Decimal("-100000"), {})

    def test_zero_income(self) -> None:
        """Test tax calculation with zero income."""
        result = calculate_tax(Decimal("0"), {})
        assert result.total_tax == Decimal("0")
        assert result.net_income == Decimal("0")

    def test_low_income_zero_tax(self) -> None:
        """Test that income under 150k after deductions is tax-free."""
        result = calculate_tax(
            Decimal("200000"),
            {"personal_allowance": Decimal("60000")},
        )
        assert result.expense_deduction == Decimal("100000")
        assert result.net_income == Decimal("40000")
        assert result.total_tax == Decimal("0")

    def test_typical_salary_earner(self) -> None:
        """Test calculation for typical 960k salary earner."""
        deductions = {
            "personal_allowance": Decimal("60000"),
            "social_security": Decimal("9000"),
        }
        result = calculate_tax(Decimal("960000"), deductions)
        assert result.gross_income == Decimal("960000")
        assert result.expense_deduction == Decimal("100000")
        assert result.total_deductions == Decimal("69000")
        assert result.net_income == Decimal("791000")
        assert result.total_tax > Decimal("0")
        assert len(result.tax_breakdown) == 8

    def test_withholding_tax_refund(self) -> None:
        """Test that overpaid withholding results in negative tax_due."""
        result = calculate_tax(
            Decimal("200000"),
            {"personal_allowance": Decimal("60000")},
            withholding_tax_paid=Decimal("10000"),
        )
        assert result.tax_due_or_refund < Decimal("0")

    def test_withholding_tax_additional_due(self) -> None:
        """Test that underpaid withholding results in positive tax_due."""
        result = calculate_tax(
            Decimal("960000"),
            {"personal_allowance": Decimal("60000")},
            withholding_tax_paid=Decimal("1000"),
        )
        assert result.tax_due_or_refund > Decimal("0")

    def test_no_deductions(self) -> None:
        """Test calculation with no deductions at all."""
        result = calculate_tax(Decimal("500000"), {})
        assert result.expense_deduction == Decimal("100000")
        assert result.total_deductions == Decimal("0")
        assert result.net_income == Decimal("400000")

    def test_result_is_pydantic_model(self) -> None:
        """Test that result is a proper Pydantic model."""
        result = calculate_tax(Decimal("500000"), {})
        assert isinstance(result, TaxCalculationResult)
        assert isinstance(result.tax_breakdown[0], TaxBracketResult)

    def test_high_income_top_bracket(self) -> None:
        """Test calculation for 10M income hitting top bracket."""
        result = calculate_tax(Decimal("10000000"), {})
        assert result.total_tax > Decimal("1000000")
        assert result.effective_tax_rate > Decimal("0.20")

    def test_deductions_exceed_income_gives_zero_tax(self) -> None:
        """Test that deductions exceeding income result in zero tax."""
        result = calculate_tax(
            Decimal("50000"),
            {"personal_allowance": Decimal("60000")},
        )
        assert result.expense_deduction == Decimal("25000")
        assert result.net_income == Decimal("0")
        assert result.total_tax == Decimal("0")


class TestExpenseDeduction:
    """Tests for standard expense deduction (50% of income, max 100,000)."""

    def test_fifty_percent_below_cap(self) -> None:
        """Test 50% deduction when income is low (below 200k cap threshold)."""
        result = calculate_expense_deduction(Decimal("100000"))
        assert result == Decimal("50000")

    def test_at_cap(self) -> None:
        """Test deduction capped at 100,000 for higher incomes."""
        result = calculate_expense_deduction(Decimal("500000"))
        assert result == Decimal("100000")

    def test_zero_income(self) -> None:
        """Test zero expense deduction for zero income."""
        result = calculate_expense_deduction(Decimal("0"))
        assert result == Decimal("0")

    def test_result_includes_expense_deduction(self) -> None:
        """Test that TaxCalculationResult includes expense_deduction field."""
        result = calculate_tax(Decimal("600000"), {"personal_allowance": Decimal("60000")})
        assert result.expense_deduction == Decimal("100000")
        assert result.net_income == Decimal("440000")
