"""Tests for the portfolio (holdings) tool — not the watchlist."""

from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from finance_ai.agents.portfolio_tools import add_portfolio_holding, get_portfolio_holdings
from finance_ai.database.models.investment_holding import InvestmentHolding
from finance_ai.database.crud.watched_asset_crud import WatchedAssetCRUD
from finance_ai.database.models.user import User
from finance_ai.tools.market_data_models import AssetSymbolMatch


def _invoke(user: User, factory: Callable[[], Session]) -> dict[str, Any]:
    result: dict[str, Any] = get_portfolio_holdings.invoke(
        {"user_id": user.id, "db_session_factory": factory}
    )
    return result


def _add_holding(session: Session, user: User, price: Decimal | None) -> None:
    session.add(
        InvestmentHolding(
            user_id=user.id,
            asset_type="stock",
            symbol="PTT.BK",
            name="PTT",
            quantity=Decimal("100"),
            average_cost_per_unit=Decimal("35.50"),
            total_cost=Decimal("3550.00"),
            purchase_date=date(2025, 1, 15),
            current_price_per_unit=price,
            current_value=None if price is None else price * 100,
            unrealized_gain_loss=None if price is None else price * 100 - Decimal("3550"),
        )
    )
    session.commit()


class TestGetPortfolioHoldings:
    """get_portfolio_holdings reads what the user owns."""

    def test_empty_portfolio_says_so(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """No holdings: count 0 and a message (never the watchlist)."""
        result = _invoke(sample_user, db_session_factory)
        assert result["holding_count"] == 0
        assert result["holdings"] == []
        assert "ยังไม่มี" in str(result["message"])

    def test_lists_holdings_with_gain(
        self,
        sample_user: User,
        test_session: Session,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Each holding has quantity, cost, value and gain/loss as strings."""
        _add_holding(test_session, sample_user, Decimal("42"))
        result = _invoke(sample_user, db_session_factory)
        assert result["holding_count"] == 1
        holding = result["holdings"][0]
        assert holding["symbol"] == "PTT.BK"
        assert holding["quantity"] == "100"
        assert holding["total_cost"] == "3550.00"
        assert holding["current_value"] == "4200.00"
        assert holding["unrealized_gain_loss"] == "650.00"
        assert result["total_current_value"] == "4200.00"

    def test_missing_price_is_none(
        self,
        sample_user: User,
        test_session: Session,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """A holding without a stored price reports None, not a made-up value."""
        _add_holding(test_session, sample_user, None)
        holding = _invoke(sample_user, db_session_factory)["holdings"][0]
        assert holding["current_value"] is None
        assert holding["unrealized_gain_loss"] is None


@pytest.fixture
def mock_symbol_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve symbols offline: 'ptt' -> 'PTT.BK', anything else echoed."""

    def fake_search(query: str) -> list[AssetSymbolMatch]:
        upper = query.strip().upper()
        symbol = "PTT.BK" if upper == "PTT" else upper
        return [AssetSymbolMatch(symbol=symbol, name=upper, exchange="MOCK", quote_type="EQUITY")]

    monkeypatch.setattr("finance_ai.tools.symbol_guard.search_asset_symbols", fake_search)


def _add(user: User, factory: Callable[[], Session], **arguments: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"user_id": user.id, "db_session_factory": factory, **arguments}
    result: dict[str, Any] = add_portfolio_holding.invoke(payload)
    return result


@pytest.mark.usefixtures("mock_symbol_search")
class TestAddPortfolioHolding:
    """add_portfolio_holding records what the user bought."""

    def test_adds_thai_stock_by_name(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """'ptt' resolves to PTT.BK and shows up in the portfolio."""
        result = _add(
            sample_user,
            db_session_factory,
            symbol="ptt",
            quantity="100",
            price_per_unit="35.5",
            purchase_date="2026-10-01",
        )
        assert result == {
            "status": "added",
            "symbol": "PTT.BK",
            "total_cost": "3550.00",
            "watchlist": {"symbol": "PTT.BK", "status": "added"},
        }
        holdings = _invoke(sample_user, db_session_factory)["holdings"]
        assert [h["symbol"] for h in holdings] == ["PTT.BK"]

    def test_bought_stock_is_watched_without_asking_alert_time(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
        test_session: Session,
    ) -> None:
        """Bought stock is watched with no follow-up question; buying again keeps one row."""
        first = _add(
            sample_user, db_session_factory, symbol="ptt", quantity="1", price_per_unit="35"
        )
        again = _add(
            sample_user, db_session_factory, symbol="ptt", quantity="1", price_per_unit="36"
        )
        watched = WatchedAssetCRUD().get_by_user(test_session, sample_user.id)
        assert [asset.symbol for asset in watched] == ["PTT.BK"]
        assert "next_step" not in first and "next_step" not in again
        assert again["watchlist"]["status"] == "already_exists"

    def test_adds_mutual_fund(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Funds keep their own code and default to today's date."""
        result = _add(
            sample_user,
            db_session_factory,
            symbol="k-usa-a(a)",
            quantity="10",
            price_per_unit="12",
            asset_type="mutual_fund",
        )
        assert result["symbol"] == "K-USA-A(A)"
        assert "tax_deduction_hint" not in result
        assert "watchlist" not in result

    def test_rmf_purchase_hints_tax_deduction(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """An RMF (named เพื่อการเลี้ยงชีพ) hints the deduction to record."""
        result = _add(
            sample_user,
            db_session_factory,
            symbol="SCBRMS&P500",
            quantity="1000",
            price_per_unit="12.5",
            asset_type="mutual_fund",
            name="กองทุนเปิดไทยพาณิชย์ S&P500 เพื่อการเลี้ยงชีพ",
        )
        assert result["tax_deduction_hint"] == {"deduction_type": "rmf", "amount": "12500.00"}

    def test_foreign_stock_added_in_baht(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """AAPL at 200 USD is stored in baht at today's rate, and watched for news."""
        foreign = "finance_ai.tools.foreign_stock_service"
        with (
            patch(f"{foreign}.fetch_currency", return_value="USD"),
            patch(f"{foreign}.fetch_current_price", return_value=Decimal("35")),
        ):
            result = _add(
                sample_user, db_session_factory, symbol="AAPL", quantity="2", price_per_unit="200"
            )
        assert result["status"] == "added"
        assert result["total_cost"] == "14000.00"
        assert (result["currency"], result["price_in_currency"]) == ("USD", "200.00")
        assert result["watchlist"]["symbol"] == "AAPL"
        holdings = _invoke(sample_user, db_session_factory)["holdings"]
        assert holdings[0]["asset_type"] == "หุ้นต่างประเทศ"

    def test_past_foreign_purchase_uses_that_days_rate(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Bought on 1 Oct at 33 THB/USD (today's rate 35): cost uses 33."""
        foreign = "finance_ai.tools.foreign_stock_service"
        with (
            patch(f"{foreign}.fetch_currency", return_value="USD"),
            patch(f"{foreign}.fetch_current_price", return_value=Decimal("35")),
            patch(f"{foreign}.fetch_close_on", return_value=Decimal("33")),
        ):
            result = _add(
                sample_user, db_session_factory, symbol="AAPL", quantity="1",
                price_per_unit="100", purchase_date="2026-10-01",
            )  # fmt: skip
        assert (result["total_cost"], result["rate_date"]) == ("3300.00", "2026-10-01")

    def test_foreign_stock_without_rate(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """No exchange rate → error, nothing saved."""
        foreign = "finance_ai.tools.foreign_stock_service"
        with (
            patch(f"{foreign}.fetch_currency", return_value="USD"),
            patch(f"{foreign}.fetch_current_price", return_value=None),
        ):
            result = _add(
                sample_user, db_session_factory, symbol="AAPL", quantity="1", price_per_unit="200"
            )
        assert "USD" in result["error"]
        assert _invoke(sample_user, db_session_factory)["holding_count"] == 0

    def test_crypto_rejected(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Crypto cannot go in the portfolio; the watchlist is suggested."""
        result = _add(
            sample_user, db_session_factory, symbol="BTC-USD", quantity="1", price_per_unit="200"
        )
        assert "watchlist" in result["error"]

    def test_bad_quantity_rejected(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Non-numeric quantity returns an error and saves nothing."""
        result = _add(
            sample_user, db_session_factory, symbol="PTT", quantity="lots", price_per_unit="35"
        )
        assert "quantity" in result["error"]
        assert _invoke(sample_user, db_session_factory)["holding_count"] == 0

    def test_zero_price_rejected(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Price must be positive."""
        result = _add(
            sample_user, db_session_factory, symbol="PTT", quantity="10", price_per_unit="0"
        )
        assert "error" in result


@pytest.mark.usefixtures("mock_symbol_search")
class TestAddPortfolioHoldingEdgeCases:
    """Unclear input becomes a question; obvious formatting is fixed."""

    def test_thai_digits_commas_and_buddhist_year(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """'๑,๐๐๐' units at '35 บาท' bought '1/10/2569' are understood."""
        result = _add(
            sample_user,
            db_session_factory,
            symbol="$ptt",
            quantity="๑,๐๐๐",
            price_per_unit="35 บาท",
            purchase_date="1/10/2569",
        )
        assert result["status"] == "added"
        assert result["symbol"] == "PTT.BK"
        assert result["total_cost"] == "35000.00"

    def test_thai_asset_type_alias(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """asset_type 'กองทุน' means mutual_fund."""
        result = _add(
            sample_user,
            db_session_factory,
            symbol="KFSDIV",
            quantity="10",
            price_per_unit="12",
            asset_type="กองทุน",
        )
        assert result["symbol"] == "KFSDIV"

    def test_fund_code_sent_as_stock_asks_if_fund(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stock search miss on a fund-shaped code asks 'is it a fund?'."""
        monkeypatch.setattr("finance_ai.tools.symbol_guard.search_asset_symbols", lambda q: [])
        result = _add(
            sample_user, db_session_factory, symbol="K-USA-A(A)", quantity="1", price_per_unit="10"
        )
        assert result["needs_clarification"] is True
        assert "K-USA-A(A)" in result["ask_user"]
        assert result["options"] == ["หุ้นไทย", "กองทุนรวม"]

    def test_unknown_stock_asks_for_name(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A miss on something that is not fund-shaped asks for the name again."""
        monkeypatch.setattr("finance_ai.tools.symbol_guard.search_asset_symbols", lambda q: [])
        result = _add(
            sample_user, db_session_factory, symbol="???", quantity="1", price_per_unit="10"
        )
        assert result["needs_clarification"] is True
        assert "options" not in result

    def test_unknown_asset_type_asks(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """An unknown type offers stock or fund."""
        result = _add(
            sample_user,
            db_session_factory,
            symbol="PTT",
            quantity="1",
            price_per_unit="10",
            asset_type="bond",
        )
        assert result["options"] == ["หุ้นไทย", "กองทุนรวม"]

    def test_bad_fund_code_asks(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """A malformed fund code asks for it again with examples."""
        result = _add(
            sample_user,
            db_session_factory,
            symbol="?",
            quantity="1",
            price_per_unit="10",
            asset_type="mutual_fund",
        )
        assert "KFSDIV" in result["ask_user"]

    def test_future_date_asks(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """A purchase date after today is questioned, not saved."""
        result = _add(
            sample_user,
            db_session_factory,
            symbol="PTT",
            quantity="1",
            price_per_unit="10",
            purchase_date="2999-01-01",
        )
        assert result["needs_clarification"] is True
        assert _invoke(sample_user, db_session_factory)["holding_count"] == 0

    def test_unreadable_date_asks(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """'last week' cannot be parsed: ask for the date."""
        result = _add(
            sample_user,
            db_session_factory,
            symbol="PTT",
            quantity="1",
            price_per_unit="10",
            purchase_date="last week",
        )
        assert "วัน" in result["ask_user"]

    def test_bad_price_asks_for_price(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """The question names the unclear value."""
        result = _add(
            sample_user, db_session_factory, symbol="PTT", quantity="1", price_per_unit="cheap"
        )
        assert "ราคา" in result["ask_user"]

    def test_negative_quantity_asks(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Validation failures ask for positive values."""
        result = _add(
            sample_user, db_session_factory, symbol="PTT", quantity="-5", price_per_unit="10"
        )
        assert result["needs_clarification"] is True
