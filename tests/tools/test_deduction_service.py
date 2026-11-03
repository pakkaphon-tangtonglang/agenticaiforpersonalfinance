"""Tests for the tax deduction recording service."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session

from finance_ai.database.crud.deduction_crud import DeductionCRUD
from finance_ai.database.models.user import User
from finance_ai.tools.deduction_service import (
    calculate_yearly_social_security,
    detect_tax_saving_fund_type,
    has_deduction_for_year,
    record_deduction,
    remove_latest_deduction,
    validate_claimable_deduction_type,
)


class TestValidateClaimableDeductionType:
    """Tests for validate_claimable_deduction_type."""

    def test_accepts_known_type_case_insensitively(self) -> None:
        """Known types are normalized to lower case."""
        assert validate_claimable_deduction_type(" Life_Insurance ") == "life_insurance"

    def test_accepts_donations(self) -> None:
        """Donations are claimable even though they have no fixed THB limit."""
        assert validate_claimable_deduction_type("donations") == "donations"

    def test_rejects_automatic_allowance(self) -> None:
        """Personal/family allowances come from the profile, not from records."""
        with pytest.raises(ValueError, match="personal_allowance"):
            validate_claimable_deduction_type("personal_allowance")

    def test_rejects_unknown_type(self) -> None:
        """Unknown types list the accepted ones."""
        with pytest.raises(ValueError, match="car_insurance"):
            validate_claimable_deduction_type("car_insurance")


class TestRecordDeduction:
    """Tests for record_deduction."""

    def test_records_new_deduction_with_remaining_room(
        self, test_session: Session, sample_user: User
    ) -> None:
        """A new record reports the year total and the room left under the limit."""
        result = record_deduction(
            test_session, sample_user.id, "life_insurance", Decimal("20000"), 2026, "ประกันชีวิต"
        )
        assert result.status == "recorded"
        assert result.year_total == Decimal("20000")
        assert result.maximum_allowed == Decimal("100000")
        assert result.remaining_room == Decimal("80000")
        assert result.exceeds_limit is False
        assert len(DeductionCRUD().get_by_user_and_year(test_session, sample_user.id, 2026)) == 1

    def test_same_type_amount_and_year_is_duplicate(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Repeating the same message on the same day does not double the deduction."""
        record_deduction(test_session, sample_user.id, "life_insurance", Decimal("20000"), 2026)
        result = record_deduction(
            test_session, sample_user.id, "life_insurance", Decimal("20000"), 2026
        )
        assert result.status == "duplicate"
        assert result.year_total == Decimal("20000")
        assert len(DeductionCRUD().get_by_user_and_year(test_session, sample_user.id, 2026)) == 1

    def test_same_amount_on_another_day_is_new_payment(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Monthly premiums of the same amount are each recorded."""
        record_deduction(test_session, sample_user.id, "life_insurance", Decimal("2000"), 2026)
        for row in DeductionCRUD().get_by_user_and_year(test_session, sample_user.id, 2026):
            row.created_at = datetime.now(timezone.utc) - timedelta(days=31)
        test_session.commit()
        result = record_deduction(
            test_session, sample_user.id, "life_insurance", Decimal("2000"), 2026
        )
        assert result.status == "recorded"
        assert result.year_total == Decimal("4000")

    def test_second_policy_adds_to_total(self, test_session: Session, sample_user: User) -> None:
        """A different amount of the same type is a second policy."""
        record_deduction(test_session, sample_user.id, "life_insurance", Decimal("20000"), 2026)
        result = record_deduction(
            test_session, sample_user.id, "life_insurance", Decimal("90000"), 2026
        )
        assert result.year_total == Decimal("110000")
        assert result.remaining_room == Decimal("0")
        assert result.exceeds_limit is True

    def test_percentage_cap_is_reported(self, test_session: Session, sample_user: User) -> None:
        """RMF reports its 30%-of-income cap."""
        result = record_deduction(test_session, sample_user.id, "rmf", Decimal("50000"), 2026)
        assert result.percentage_of_income_cap == Decimal("0.30")

    def test_donations_have_no_fixed_limit(self, test_session: Session, sample_user: User) -> None:
        """Donations are capped by net income, so no THB room is reported."""
        result = record_deduction(test_session, sample_user.id, "donations", Decimal("5000"), 2026)
        assert result.maximum_allowed is None
        assert result.remaining_room is None

    def test_rejects_non_positive_amount(self, test_session: Session, sample_user: User) -> None:
        """Zero or negative amounts are refused with the received value."""
        with pytest.raises(ValueError, match="-100"):
            record_deduction(test_session, sample_user.id, "rmf", Decimal("-100"), 2026)

    def test_ssf_ended_after_2024(self, test_session: Session, sample_user: User) -> None:
        """SSF is refused from tax year 2025 but still accepted for 2024."""
        with pytest.raises(ValueError, match="2024"):
            record_deduction(test_session, sample_user.id, "ssf", Decimal("10000"), 2026)
        result = record_deduction(test_session, sample_user.id, "ssf", Decimal("10000"), 2024)
        assert result.status == "recorded"


class TestRemoveLatestDeduction:
    """Tests for remove_latest_deduction."""

    def test_removes_most_recent_of_type(self, test_session: Session, sample_user: User) -> None:
        """Undo removes the latest record of that type and returns its amount."""
        record_deduction(test_session, sample_user.id, "rmf", Decimal("10000"), 2026)
        record_deduction(test_session, sample_user.id, "rmf", Decimal("30000"), 2026)
        removed = remove_latest_deduction(test_session, sample_user.id, "rmf", 2026)
        assert removed == Decimal("30000")
        remaining = DeductionCRUD().get_by_user_and_year(test_session, sample_user.id, 2026)
        assert [row.amount for row in remaining] == [Decimal("10000")]

    def test_returns_none_when_nothing_recorded(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Nothing to undo returns None."""
        assert remove_latest_deduction(test_session, sample_user.id, "ssf", 2026) is None


class TestHasDeductionForYear:
    """Tests for has_deduction_for_year."""

    def test_false_then_true(self, test_session: Session, sample_user: User) -> None:
        """Reports whether a type was recorded for the year."""
        assert not has_deduction_for_year(test_session, sample_user.id, "social_security", 2026)
        record_deduction(test_session, sample_user.id, "social_security", Decimal("9000"), 2026)
        assert has_deduction_for_year(test_session, sample_user.id, "social_security", 2026)
        assert not has_deduction_for_year(test_session, sample_user.id, "social_security", 2025)


class TestCalculateYearlySocialSecurity:
    """Tests for calculate_yearly_social_security."""

    @pytest.mark.parametrize(
        ("monthly_salary", "expected"),
        [
            (Decimal("50000"), Decimal("10500.00")),
            (Decimal("17500"), Decimal("10500.00")),
            (Decimal("15000"), Decimal("9000.00")),
            (Decimal("10000"), Decimal("6000.00")),
        ],
    )
    def test_five_percent_of_capped_wage(self, monthly_salary: Decimal, expected: Decimal) -> None:
        """5% of wage up to 17,500/month (from 2569), for 12 months."""
        assert calculate_yearly_social_security(monthly_salary) == expected

    def test_rejects_non_positive_salary(self) -> None:
        """Salary must be positive."""
        with pytest.raises(ValueError, match="0"):
            calculate_yearly_social_security(Decimal("0"))


class TestDetectTaxSavingFundType:
    """Tests for detect_tax_saving_fund_type."""

    @pytest.mark.parametrize(
        ("symbol", "expected"),
        [
            ("K-USARMF", "rmf"),
            ("SCBRMS&P500", None),
            ("scbrmf", "rmf"),
            ("K-CHANGESSF", "ssf"),
            ("KFSDIV", None),
            ("PTT.BK", None),
        ],
    )
    def test_detects_from_fund_code(self, symbol: str, expected: str | None) -> None:
        """RMF/SSF are recognized from the code; everything else is None."""
        assert detect_tax_saving_fund_type(symbol) == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("กองทุนเปิดไทยพาณิชย์ S&P500 เพื่อการเลี้ยงชีพ", "rmf"),
            ("กองทุนเปิดเค Changes เพื่อการออม", "ssf"),
            ("SCB S&P500 Super Savings Fund", "ssf"),
            ("Retirement Mutual Fund", "rmf"),
            ("กองทุนเปิดเค หุ้นปันผล", None),
        ],
    )
    def test_detects_from_fund_name(self, name: str, expected: str | None) -> None:
        """Codes like SCBRMS&P500 do not say RMF, so the fund name is checked too."""
        assert detect_tax_saving_fund_type("SCBRMS&P500", name) == expected
