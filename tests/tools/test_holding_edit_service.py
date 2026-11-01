"""Tests for holding_edit_service (edit or delete a portfolio row)."""

# pylint: disable=redefined-outer-name

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session

from finance_ai.database.models.investment_holding import InvestmentHolding
from finance_ai.database.models.transaction import Transaction
from finance_ai.database.models.user import User
from finance_ai.tools.holding_edit_service import (
    HoldingEditForm,
    HoldingNotFoundError,
    delete_holding,
    update_holding,
)
from finance_ai.tools.holding_entry_service import HoldingEntryError
from finance_ai.tools.investment_service import add_investment_holding


@pytest.fixture
def holding(test_session: Session, sample_user: User) -> InvestmentHolding:
    """500 PTT at 42 THB with a stored latest price of 44."""
    saved = add_investment_holding(
        test_session,
        sample_user.id,
        "stock",
        "PTT.BK",
        "PTT",
        Decimal("500"),
        Decimal("42"),
        date(2026, 10, 1),
    )
    saved.current_price_per_unit = Decimal("44")
    test_session.commit()
    return saved


def buy_transactions(session: Session, holding_id: str) -> list[Transaction]:
    """The buy transactions linked to a holding."""
    return session.query(Transaction).filter_by(holding_id=holding_id).all()


class TestUpdateHolding:
    """Quantity and average cost can be corrected."""

    def test_updates_cost_value_and_gain(
        self, test_session: Session, sample_user: User, holding: InvestmentHolding
    ) -> None:
        """600 units at 40 → cost 24,000; at 44 the value is 26,400 (+2,400)."""
        form = HoldingEditForm(quantity="600", price_per_unit="40")

        updated = update_holding(test_session, sample_user.id, holding.id, form)

        assert updated.quantity == Decimal("600")
        assert updated.average_cost_per_unit == Decimal("40")
        assert updated.total_cost == Decimal("24000.00")
        assert updated.current_value == Decimal("26400.00")
        assert updated.unrealized_gain_loss == Decimal("2400.00")

    def test_buy_transaction_follows_the_edit(
        self, test_session: Session, sample_user: User, holding: InvestmentHolding
    ) -> None:
        """The linked buy record keeps the same quantity, price and amount."""
        update_holding(
            test_session,
            sample_user.id,
            holding.id,
            HoldingEditForm(quantity="1,000", price_per_unit="41.5"),
        )

        [buy] = buy_transactions(test_session, holding.id)
        assert (buy.quantity, buy.price_per_unit, buy.amount) == (
            Decimal("1000"),
            Decimal("41.5"),
            Decimal("41500.00"),
        )

    def test_without_a_price_value_stays_empty(
        self, test_session: Session, sample_user: User, holding: InvestmentHolding
    ) -> None:
        """No stored price → no value or gain is invented."""
        holding.current_price_per_unit = None
        holding.current_value = None
        test_session.commit()

        updated = update_holding(
            test_session,
            sample_user.id,
            holding.id,
            HoldingEditForm(quantity="1", price_per_unit="1"),
        )

        assert updated.current_value is None
        assert updated.unrealized_gain_loss is None

    @pytest.mark.parametrize(
        ("quantity", "price", "message"),
        [
            ("0", "40", "จำนวนหุ้น/หน่วย"),
            ("abc", "40", "จำนวนหุ้น/หน่วย"),
            ("10", "-1", "ต้นทุนต่อหน่วย"),
            ("10", "", "ต้นทุนต่อหน่วย"),
        ],
    )
    def test_bad_numbers_are_rejected_in_thai(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        test_session: Session,
        sample_user: User,
        holding: InvestmentHolding,
        quantity: str,
        price: str,
        message: str,
    ) -> None:
        """Zero, negative, blank or unreadable numbers keep the holding unchanged."""
        with pytest.raises(HoldingEntryError, match=message):
            update_holding(
                test_session,
                sample_user.id,
                holding.id,
                HoldingEditForm(quantity=quantity, price_per_unit=price),
            )
        test_session.refresh(holding)
        assert holding.quantity == Decimal("500")

    def test_other_users_holding_is_not_found(
        self, test_session: Session, holding: InvestmentHolding
    ) -> None:
        """Someone else's id behaves like a missing one."""
        with pytest.raises(HoldingNotFoundError):
            update_holding(
                test_session,
                "someone-else",
                holding.id,
                HoldingEditForm(quantity="1", price_per_unit="1"),
            )


class TestDeleteHolding:
    """Deleting removes the row and its buy record."""

    def test_deletes_holding_and_its_transactions(
        self, test_session: Session, sample_user: User, holding: InvestmentHolding
    ) -> None:
        """The only holding can be deleted (the reported bug)."""
        holding_id = holding.id

        delete_holding(test_session, sample_user.id, holding_id)

        assert test_session.get(InvestmentHolding, holding_id) is None
        assert not buy_transactions(test_session, holding_id)

    def test_missing_or_foreign_holding_raises(
        self, test_session: Session, sample_user: User, holding: InvestmentHolding
    ) -> None:
        """Unknown ids and other users' rows are not deleted."""
        with pytest.raises(HoldingNotFoundError):
            delete_holding(test_session, sample_user.id, "no-such-id")
        with pytest.raises(HoldingNotFoundError):
            delete_holding(test_session, "someone-else", holding.id)
        assert test_session.get(InvestmentHolding, holding.id) is not None
