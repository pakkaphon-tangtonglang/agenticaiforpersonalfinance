"""Tests for holding_entry_service (asset page 'add to portfolio' form)."""

# pylint: disable=redefined-outer-name

from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from finance_ai.database.models.investment_holding import InvestmentHolding
from finance_ai.database.models.user import User
from finance_ai.tools.holding_entry_service import (
    HoldingEntry,
    HoldingEntryError,
    HoldingForm,
    add_holding_from_form,
)

_PRICE = "finance_ai.tools.holding_entry_service.fetch_current_price"
_NAV = "finance_ai.tools.holding_entry_service.fetch_latest_nav"


def _add(
    session: Session, user: User, quantity: str, price: str | None, latest: str | None
) -> HoldingEntry:
    """Add PTT.BK with a mocked latest price."""
    latest_price = None if latest is None else Decimal(latest)
    with patch(_PRICE, return_value=latest_price):
        form = HoldingForm(symbol="PTT.BK", name="PTT", quantity=quantity, price_per_unit=price)
        return add_holding_from_form(session, user.id, form)


class TestAddHoldingFromForm:
    """Saving a purchase typed into the asset page."""

    def test_blank_price_uses_latest(self, test_session: Session, sample_user: User) -> None:
        """Blank price = bought at the latest price, so gain is zero."""
        entry = _add(test_session, sample_user, "100", "", "35.25")
        assert entry.used_latest_price is True
        assert entry.price_per_unit == "35.25"
        assert entry.total_cost == "3525.00"
        assert entry.gain_loss == "0.00"
        assert entry.gain_loss_percent == "0.00"

    def test_typed_price_shows_gain(self, test_session: Session, sample_user: User) -> None:
        """Bought at 30, now 36: +600 on 100 shares (+20%)."""
        entry = _add(test_session, sample_user, "1,00", "30 บาท", "36")
        assert entry.used_latest_price is False
        assert entry.quantity == "100"
        assert entry.gain_loss == "600.00"
        assert entry.gain_loss_percent == "20.00"

    def test_typed_price_shows_loss(self, test_session: Session, sample_user: User) -> None:
        """Bought above the latest price: negative gain."""
        entry = _add(test_session, sample_user, "10", "40", "38")
        assert entry.gain_loss == "-20.00"
        assert entry.gain_loss_percent == "-5.00"

    def test_latest_price_is_stored_on_holding(
        self, test_session: Session, sample_user: User
    ) -> None:
        """The portfolio list shows the gain right away."""
        _add(test_session, sample_user, "10", "30", "33")
        holding = test_session.query(InvestmentHolding).one()
        assert holding.current_price_per_unit == Decimal("33")
        assert holding.unrealized_gain_loss == Decimal("30")

    def test_typed_price_without_latest(self, test_session: Session, sample_user: User) -> None:
        """No market price: saved, gain unknown."""
        entry = _add(test_session, sample_user, "10", "30", None)
        assert entry.total_cost == "300.00"
        assert entry.gain_loss is None

    def test_blank_price_without_latest_raises(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Nothing to use as the price: ask the user to type one."""
        with pytest.raises(HoldingEntryError, match="กรอกราคาซื้อเอง"):
            _add(test_session, sample_user, "10", None, None)
        assert test_session.query(InvestmentHolding).count() == 0

    @pytest.mark.parametrize("quantity", ["", "abc", "0", "-5", "nan"])
    def test_bad_quantity_raises(
        self, test_session: Session, sample_user: User, quantity: str
    ) -> None:
        """Quantity must be a positive number."""
        with pytest.raises(HoldingEntryError, match="จำนวนหุ้น"):
            _add(test_session, sample_user, quantity, "", "35")

    @pytest.mark.parametrize("price", ["abc", "0", "-1"])
    def test_bad_price_raises(self, test_session: Session, sample_user: User, price: str) -> None:
        """A typed price must be a positive number."""
        with pytest.raises(HoldingEntryError, match="ราคาซื้อ"):
            _add(test_session, sample_user, "10", price, "35")

    def test_foreign_symbol_rejected(self, test_session: Session, sample_user: User) -> None:
        """The baht portfolio holds Thai stocks only."""
        with pytest.raises(HoldingEntryError, match="หุ้นไทย"):
            add_holding_from_form(
                test_session,
                sample_user.id,
                HoldingForm(symbol="AAPL", quantity="1", price_per_unit="200"),
            )

    def test_tiny_lot_has_no_percent(self, test_session: Session, sample_user: User) -> None:
        """A cost that rounds to 0.00 skips the percentage instead of dividing by zero."""
        entry = _add(test_session, sample_user, "0.001", "1", "2")
        assert entry.total_cost == "0.00"
        assert entry.gain_loss_percent is None


def _add_form(session: Session, user: User, form: HoldingForm, latest: str | None) -> HoldingEntry:
    """Add any form with both price sources mocked to the same latest value."""
    latest_price = None if latest is None else Decimal(latest)
    with patch(_PRICE, return_value=latest_price), patch(_NAV, return_value=latest_price):
        return add_holding_from_form(session, user.id, form)


class TestFundsAndPurchaseDate:
    """Thai mutual funds (SEC NAV) and the optional purchase date."""

    def test_fund_blank_price_uses_latest_nav(
        self, test_session: Session, sample_user: User
    ) -> None:
        """A fund bought at the latest NAV keeps its 4 decimals."""
        form = HoldingForm(symbol="k-usa-a(a)", quantity="100", asset_type="mutual_fund")
        entry = _add_form(test_session, sample_user, form, "16.6925")
        assert entry.symbol == "K-USA-A(A)"
        assert entry.price_per_unit == "16.6925"
        assert entry.total_cost == "1669.25"
        holding = test_session.query(InvestmentHolding).one()
        assert holding.asset_type == "mutual_fund"

    def test_fund_uses_nav_not_market_price(self, test_session: Session, sample_user: User) -> None:
        """Funds never ask Yahoo for a price."""
        form = HoldingForm(symbol="SCBSET50", quantity="10", asset_type="mutual_fund")
        with (
            patch(_NAV, return_value=Decimal("24.044")) as nav,
            patch(_PRICE) as market,
        ):
            add_holding_from_form(test_session, sample_user.id, form)
        nav.assert_called_once_with("SCBSET50")
        market.assert_not_called()

    def test_bad_fund_code(self, test_session: Session, sample_user: User) -> None:
        """A malformed fund code is rejected with examples."""
        form = HoldingForm(symbol="?", quantity="1", asset_type="mutual_fund")
        with pytest.raises(HoldingEntryError, match="รหัสกองทุน"):
            _add_form(test_session, sample_user, form, "10")

    def test_unknown_asset_type(self, test_session: Session, sample_user: User) -> None:
        """Only stock and mutual_fund are accepted."""
        form = HoldingForm(symbol="BTC-USD", quantity="1", asset_type="crypto")
        with pytest.raises(HoldingEntryError, match="ประเภทสินทรัพย์"):
            _add_form(test_session, sample_user, form, "10")

    @pytest.mark.parametrize(
        ("typed", "expected"),
        [("2026-10-01", date(2026, 10, 1)), ("1/10/2569", date(2026, 10, 1))],
    )
    def test_purchase_date_saved(
        self, test_session: Session, sample_user: User, typed: str, expected: date
    ) -> None:
        """AD and Buddhist-era dates are stored as the purchase date."""
        form = HoldingForm(symbol="PTT.BK", quantity="1", price_per_unit="30", purchase_date=typed)
        entry = _add_form(test_session, sample_user, form, "33")
        assert entry.purchase_date == expected.isoformat()
        assert test_session.query(InvestmentHolding).one().purchase_date == expected

    def test_blank_date_is_today(self, test_session: Session, sample_user: User) -> None:
        """No date means bought today."""
        form = HoldingForm(symbol="PTT.BK", quantity="1", price_per_unit="30", purchase_date="")
        entry = _add_form(test_session, sample_user, form, "33")
        assert entry.purchase_date == date.today().isoformat()

    def test_future_date_rejected(self, test_session: Session, sample_user: User) -> None:
        """A purchase cannot be in the future; nothing is saved."""
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        form = HoldingForm(symbol="PTT.BK", quantity="1", purchase_date=tomorrow)
        with pytest.raises(HoldingEntryError, match="ไม่เกินวันนี้"):
            _add_form(test_session, sample_user, form, "33")
        assert test_session.query(InvestmentHolding).count() == 0

    def test_unreadable_date_rejected(self, test_session: Session, sample_user: User) -> None:
        """Dates that cannot be read get a format example."""
        form = HoldingForm(symbol="PTT.BK", quantity="1", purchase_date="last week")
        with pytest.raises(HoldingEntryError, match="2026-10-01"):
            _add_form(test_session, sample_user, form, "33")
