"""Tests for the debt tools the expense agent uses."""

from collections.abc import Callable
from decimal import Decimal
from typing import Any

from sqlalchemy.orm import Session

from finance_ai.agents.debt_tools import (
    list_debts_tool,
    pay_debt,
    record_debt_tool,
    update_debt_balance,
)
from finance_ai.agents.expense_agent import EXPENSE_TOOLS
from finance_ai.agents.planning_agent import PLANNING_TOOLS
from finance_ai.database.crud.transaction_crud import TransactionCRUD
from finance_ai.database.models.user import User


def _injected(user: User, factory: Callable[[], Session], **arguments: Any) -> dict[str, Any]:
    """Tool input with the injected state filled in."""
    return {**arguments, "user_id": user.id, "db_session_factory": factory}


def _owe_friend(user: User, factory: Callable[[], Session], amount: str = "2000") -> Any:
    return record_debt_tool.invoke(_injected(user, factory, creditor="เพื่อน", amount=amount))


class TestRecordDebtTool:
    """Owing money is saved as a debt, never as an expense."""

    def test_records_without_expense(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
        test_session: Session,
    ) -> None:
        """'ติดหนี้เพื่อน 2000' creates a debt and no transaction."""
        result = _owe_friend(sample_user, db_session_factory)
        assert result["status"] == "recorded"
        assert result["remaining_balance"] == "2000.00"
        assert result["debt_type_label"] == "ยืมคนรู้จัก"
        assert TransactionCRUD().get_all(test_session) == []

    def test_bad_amount_is_error(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """A non-number amount comes back as an error the agent can explain."""
        result = _owe_friend(sample_user, db_session_factory, amount="เยอะ")
        assert result["status"] == "error"

    def test_optional_terms_parsed(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Interest and installment strings are saved as numbers."""
        result = record_debt_tool.invoke(
            _injected(
                sample_user, db_session_factory, creditor="ธนาคาร", amount="500,000",
                debt_type="personal_loan", interest_rate_percent="18", monthly_payment="12000",
            )  # fmt: skip
        )
        assert (result["interest_rate_percent"], result["monthly_payment"]) == ("18", "12000")


class TestPayDebt:
    """Paying back is an expense and lowers the debt."""

    def test_pays_down_recorded_debt(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
        test_session: Session,
    ) -> None:
        """2,000 owed, 500 paid: one debt expense and 1,500 left."""
        _owe_friend(sample_user, db_session_factory)
        result = pay_debt.invoke(
            _injected(sample_user, db_session_factory, amount="500", creditor="เพื่อน")
        )
        assert result["status"] == "paid"
        assert result["remaining_balance"] == "1500.00"
        [transaction] = TransactionCRUD().get_all(test_session)
        assert (transaction.category, transaction.amount) == ("debt", Decimal("500"))

    def test_paid_off(self, sample_user: User, db_session_factory: Callable[[], Session]) -> None:
        """Paying the rest says the debt is closed."""
        _owe_friend(sample_user, db_session_factory)
        result = pay_debt.invoke(
            _injected(sample_user, db_session_factory, amount="2000", creditor="เพื่อน")
        )
        assert result["paid_off"] is True

    def test_no_debt_on_file_still_records_expense(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
        test_session: Session,
    ) -> None:
        """'ใช้หนี้เพื่อน 5000' with nothing recorded: expense saved, agent asks what is left."""
        result = pay_debt.invoke(
            _injected(sample_user, db_session_factory, amount="5000", creditor="เพื่อน")
        )
        assert result["status"] == "paid_no_debt_on_file"
        assert result["next_step"] == "ASK_REMAINING_DEBT"
        assert len(TransactionCRUD().get_all(test_session)) == 1

    def test_bad_amount_is_error(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Nothing is saved for an amount that is not a number."""
        result = pay_debt.invoke(_injected(sample_user, db_session_factory, amount="abc"))
        assert result["status"] == "error"


class TestListAndCorrect:
    """Listing debts and correcting a balance."""

    def test_list_totals(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Open debts with the total owed."""
        _owe_friend(sample_user, db_session_factory)
        result = list_debts_tool.invoke(_injected(sample_user, db_session_factory))
        assert (result["debt_count"], result["total_remaining"]) == (1, "2000.00")
        assert result["debts"][0]["creditor"] == "เพื่อน"

    def test_correct_balance(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """The user says only 1,200 is left."""
        _owe_friend(sample_user, db_session_factory)
        result = update_debt_balance.invoke(
            _injected(sample_user, db_session_factory, creditor="เพื่อน", remaining_balance="1200")
        )
        assert (result["status"], result["remaining_balance"]) == ("updated", "1200.00")

    def test_correct_unknown_debt(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """No debt to correct."""
        result = update_debt_balance.invoke(
            _injected(sample_user, db_session_factory, creditor="เพื่อน", remaining_balance="1")
        )
        assert result["status"] == "not_found"

    def test_negative_balance_is_error(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """A negative balance is refused."""
        _owe_friend(sample_user, db_session_factory)
        result = update_debt_balance.invoke(
            _injected(sample_user, db_session_factory, creditor="เพื่อน", remaining_balance="-1")
        )
        assert result["status"] == "error"


def test_agents_have_debt_tools() -> None:
    """Expense records and pays debts; planning can read them."""
    expense_names = {tool.name for tool in EXPENSE_TOOLS}
    assert {"record_debt_tool", "pay_debt", "list_debts_tool", "update_debt_balance"} <= (
        expense_names
    )
    assert "list_debts_tool" in {tool.name for tool in PLANNING_TOOLS}
