"""Tests for the tax deduction agent tools."""

from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any

from sqlalchemy.orm import Session

from finance_ai.agents.tax_deduction_tools import (
    calculate_tax_from_records,
    record_social_security_cross,
    record_tax_deduction_cross,
    undo_tax_deduction_cross,
)
from finance_ai.database.crud.deduction_crud import DeductionCRUD
from finance_ai.database.crud.income_crud import IncomeCRUD
from finance_ai.database.models.user import User


def _injected(user: User, factory: Callable[[], Session], **arguments: Any) -> dict[str, Any]:
    """Tool input with the injected state filled in."""
    return {**arguments, "user_id": user.id, "db_session_factory": factory}


class TestRecordTaxDeductionCross:
    """Tests for record_tax_deduction_cross."""

    def test_records_life_insurance(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Amount text with commas and baht is parsed and saved."""
        result = record_tax_deduction_cross.invoke(
            _injected(
                sample_user, db_session_factory,
                deduction_type="life_insurance", amount="20,000 บาท", tax_year="2026",
            )  # fmt: skip
        )
        assert result["status"] == "recorded"
        assert result["year_total"] == "20000.00"
        assert result["remaining_room"] == "80000.00"

    def test_defaults_to_current_year(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """No tax_year means this year."""
        result = record_tax_deduction_cross.invoke(
            _injected(sample_user, db_session_factory, deduction_type="rmf", amount="10000")
        )
        assert result["tax_year"] == date.today().year

    def test_unknown_type_returns_error(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Unknown types come back as an error the agent can explain."""
        result = record_tax_deduction_cross.invoke(
            _injected(sample_user, db_session_factory, deduction_type="car", amount="5000")
        )
        assert result["status"] == "error"
        assert "car" in result["message"]

    def test_ssf_after_2024_returns_error(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """SSF bought in 2026 is not deductible; the agent gets the reason."""
        result = record_tax_deduction_cross.invoke(
            _injected(
                sample_user, db_session_factory,
                deduction_type="ssf", amount="10000", tax_year="2026",
            )  # fmt: skip
        )
        assert result["status"] == "error"
        assert "SSF" in result["message"]

    def test_bad_amount_returns_error(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Unparseable amounts come back as an error."""
        result = record_tax_deduction_cross.invoke(
            _injected(sample_user, db_session_factory, deduction_type="rmf", amount="abc")
        )
        assert result["status"] == "error"


class TestRecordSocialSecurityCross:
    """Tests for record_social_security_cross."""

    def test_records_yearly_contribution(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
        test_session: Session,
    ) -> None:
        """A 50,000 salary records the 10,500 yearly maximum (ceiling 17,500)."""
        result = record_social_security_cross.invoke(
            _injected(sample_user, db_session_factory, monthly_salary="50000", tax_year="2026")
        )
        assert result["status"] == "recorded"
        assert result["amount"] == "10500.00"
        rows = DeductionCRUD().get_by_type(test_session, sample_user.id, 2026, "social_security")
        assert len(rows) == 1

    def test_already_recorded_is_not_added_twice(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Social security is once per year, whatever the salary."""
        arguments = _injected(sample_user, db_session_factory, tax_year="2026")
        record_social_security_cross.invoke(arguments | {"monthly_salary": "50000"})
        result = record_social_security_cross.invoke(arguments | {"monthly_salary": "10000"})
        assert result["status"] == "already_recorded"

    def test_bad_salary_returns_error(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Zero salary is an error."""
        result = record_social_security_cross.invoke(
            _injected(sample_user, db_session_factory, monthly_salary="0")
        )
        assert result["status"] == "error"


class TestUndoTaxDeductionCross:
    """Tests for undo_tax_deduction_cross."""

    def test_removes_latest(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Undo removes the last record of the type."""
        arguments = _injected(
            sample_user, db_session_factory, deduction_type="rmf", tax_year="2026"
        )
        record_tax_deduction_cross.invoke(arguments | {"amount": "30000"})
        result = undo_tax_deduction_cross.invoke(arguments)
        assert result == {
            "status": "removed",
            "deduction_type": "rmf",
            "amount": "30000.00",
            "tax_year": 2026,
        }

    def test_nothing_to_undo(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """No record of the type returns not_found."""
        result = undo_tax_deduction_cross.invoke(
            _injected(sample_user, db_session_factory, deduction_type="rmf")
        )
        assert result["status"] == "not_found"

    def test_invalid_type_returns_error(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Invalid type is an error."""
        result = undo_tax_deduction_cross.invoke(
            _injected(sample_user, db_session_factory, deduction_type="personal_allowance")
        )
        assert result["status"] == "error"


class TestCalculateTaxFromRecords:
    """Tests for calculate_tax_from_records."""

    def test_no_income_asks_for_it(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Without recorded income there is nothing to calculate."""
        result = calculate_tax_from_records.invoke(
            _injected(sample_user, db_session_factory, tax_year="2026")
        )
        assert result["status"] == "no_income"

    def test_uses_recorded_income_and_deductions(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
        test_session: Session,
    ) -> None:
        """Recorded income, recorded deductions and profile allowances are combined."""
        IncomeCRUD().create(
            test_session, user_id=sample_user.id, income_type="salary",
            amount=Decimal("600000"), tax_year=2026, pay_period="yearly",
            withholding_tax=Decimal("0"),
        )  # fmt: skip
        test_session.commit()
        record_tax_deduction_cross.invoke(
            _injected(
                sample_user, db_session_factory,
                deduction_type="life_insurance", amount="20000", tax_year="2026",
            )  # fmt: skip
        )
        result = calculate_tax_from_records.invoke(
            _injected(sample_user, db_session_factory, tax_year="2026")
        )
        assert result["status"] == "calculated"
        assert result["income_record_count"] == 1
        assert result["deductions_used"]["life_insurance"] == "20000.00"
        assert result["deductions_used"]["personal_allowance"] == "60000"
        assert Decimal(result["gross_income"]) == Decimal("600000")
        assert Decimal(result["expense_deduction"]) == Decimal("100000")
        # 600,000 - 100,000 expense - 60,000 personal - 60,000 (2 parents) - 20,000 life
        assert Decimal(result["net_income"]) == Decimal("360000")
