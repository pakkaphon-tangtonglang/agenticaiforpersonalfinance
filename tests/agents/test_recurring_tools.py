"""Tests for the recurring income/expense agent tools."""

from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from finance_ai.agents.expense_agent import EXPENSE_TOOLS
from finance_ai.agents.recurring_tools import (
    add_recurring_entry,
    cancel_recurring_entry_tool,
    list_recurring_entries_tool,
)
from finance_ai.database.crud.transaction_crud import TransactionCRUD
from finance_ai.database.models.user import User
from finance_ai.tools.recurring_service import bangkok_today


def _injected(user: User, factory: Callable[[], Session], **arguments: Any) -> dict[str, Any]:
    """Tool input with the injected state filled in."""
    return {**arguments, "user_id": user.id, "db_session_factory": factory}


def _add_rent(user: User, factory: Callable[[], Session], day: int) -> dict[str, Any]:
    result: dict[str, Any] = add_recurring_entry.invoke(
        _injected(
            user, factory, entry_type="income", amount="8,000 บาท",
            category="rental", day_of_month=str(day), description="ค่าเช่าบ้าน",
        )  # fmt: skip
    )
    return result


class TestAddRecurringEntry:
    """add_recurring_entry saves the entry and posts this month when due."""

    def test_due_today_posts_now(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
        test_session: Session,
    ) -> None:
        """Day = today posts this month's income at once."""
        result = _add_rent(sample_user, db_session_factory, bangkok_today().day)
        assert result["status"] == "created"
        assert result["amount"] == "8000.00"
        assert result["posted_now"] == 1
        assert len(TransactionCRUD().get_all(test_session)) == 1

    def test_bad_day_returns_error(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Day 40 is an error the agent can explain."""
        result = _add_rent(sample_user, db_session_factory, 40)
        assert result["status"] == "error"

    def test_non_number_day_returns_error(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """A day that is not a number is an error."""
        result = add_recurring_entry.invoke(
            _injected(
                sample_user, db_session_factory, entry_type="income",
                amount="8000", category="rental", day_of_month="ทุกต้นเดือน",
            )  # fmt: skip
        )
        assert result["status"] == "error"


class TestListAndCancelTools:
    """Listing and cancelling through the tools."""

    def test_list_then_cancel(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """An added entry is listed, then cancelled by keyword."""
        _add_rent(sample_user, db_session_factory, 28)
        listed = list_recurring_entries_tool.invoke(_injected(sample_user, db_session_factory))
        assert listed["count"] == 1
        cancelled = cancel_recurring_entry_tool.invoke(
            _injected(sample_user, db_session_factory, search_term="เช่า")
        )
        assert cancelled["status"] == "cancelled"
        listed = list_recurring_entries_tool.invoke(_injected(sample_user, db_session_factory))
        assert listed["count"] == 0

    def test_cancel_not_found(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Nothing to cancel."""
        result = cancel_recurring_entry_tool.invoke(
            _injected(sample_user, db_session_factory, search_term="เช่า")
        )
        assert result["status"] == "not_found"


def test_expense_agent_has_recurring_tools() -> None:
    """The expense agent can create, list and cancel recurring entries."""
    names = {tool.name for tool in EXPENSE_TOOLS}
    assert {
        "add_recurring_entry",
        "list_recurring_entries_tool",
        "cancel_recurring_entry_tool",
    } <= names


def test_day_already_passed_this_month_is_posted(
    sample_user: User, db_session_factory: Callable[[], Session]
) -> None:
    """Rent on the 1st, entered later in the month: this month's rent is recorded."""
    result = _add_rent(sample_user, db_session_factory, 1)
    assert result["posted_now"] == 1


def test_installment_linked_to_debt(
    sample_user: User, db_session_factory: Callable[[], Session]
) -> None:
    """A monthly installment naming a recorded creditor pays that debt down."""
    from finance_ai.agents.debt_tools import list_debts_tool, record_debt_tool

    record_debt_tool.invoke(
        _injected(sample_user, db_session_factory, creditor="ธนาคารกสิกร", amount="500000")
    )
    result = add_recurring_entry.invoke(
        _injected(
            sample_user, db_session_factory, entry_type="expense", amount="12000",
            category="debt", day_of_month="1", description="ผ่อนสินเชื่อ",
            debt_creditor="กสิกร",
        )  # fmt: skip
    )
    assert result["linked_debt"] == "ธนาคารกสิกร"
    debts = list_debts_tool.invoke(_injected(sample_user, db_session_factory))
    assert debts["total_remaining"] == "488000.00"
