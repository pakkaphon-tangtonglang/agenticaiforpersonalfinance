"""Tests for foreign stocks held in a baht portfolio."""

from decimal import Decimal
from unittest.mock import patch

import pytest

from finance_ai.tools.foreign_stock_service import (
    is_foreign_stock_symbol,
    latest_price_in_thb,
    price_in_thb,
)

SERVICE = "finance_ai.tools.foreign_stock_service"


class TestIsForeignStockSymbol:
    """Which symbols count as foreign stocks."""

    @pytest.mark.parametrize("symbol", ["NVDA", "aapl", "BRK-B", "7203.T", "0700.HK"])
    def test_foreign_stocks(self, symbol: str) -> None:
        """US and other exchanges' tickers are foreign stocks."""
        assert is_foreign_stock_symbol(symbol)

    @pytest.mark.parametrize("symbol", ["PTT.BK", "BTC-USD", "GC=F", "^SET", "USDTHB=X", ""])
    def test_not_foreign_stocks(self, symbol: str) -> None:
        """Thai stocks, crypto, futures, indexes and FX are not."""
        assert not is_foreign_stock_symbol(symbol)


class TestPriceInThb:
    """Converting a price in the stock's currency to baht."""

    def test_usd_price_converted(self) -> None:
        """180 USD at 35 THB/USD is 6,300 THB."""
        with (
            patch(f"{SERVICE}.fetch_currency", return_value="USD"),
            patch(f"{SERVICE}.fetch_current_price", return_value=Decimal("35")),
        ):
            converted = price_in_thb("NVDA", Decimal("180"))
        assert converted.amount_thb == Decimal("6300.0000")
        assert converted.currency == "USD"
        assert converted.exchange_rate == Decimal("35")

    def test_unknown_currency_defaults_to_usd(self) -> None:
        """Yahoo sometimes reports no currency; US listings are assumed."""
        with (
            patch(f"{SERVICE}.fetch_currency", return_value=None),
            patch(f"{SERVICE}.fetch_current_price", return_value=Decimal("35")),
        ):
            assert price_in_thb("NVDA", Decimal("1")).currency == "USD"

    def test_thb_listing_needs_no_rate(self) -> None:
        """A THB-quoted symbol keeps its price."""
        with patch(f"{SERVICE}.fetch_currency", return_value="THB"):
            converted = price_in_thb("XYZ", Decimal("10"))
        assert converted.amount_thb == Decimal("10")
        assert converted.exchange_rate == Decimal("1")

    def test_missing_rate_raises(self) -> None:
        """No exchange rate → a clear error."""
        with (
            patch(f"{SERVICE}.fetch_currency", return_value="USD"),
            patch(f"{SERVICE}.fetch_current_price", return_value=None),
        ):
            with pytest.raises(ValueError, match="USD"):
                price_in_thb("NVDA", Decimal("1"))


class TestLatestPriceInThb:
    """The latest market price, in baht."""

    def test_latest_converted(self) -> None:
        """Latest USD price times today's rate."""
        prices = {"NVDA": Decimal("200"), "USDTHB=X": Decimal("34")}
        with (
            patch(f"{SERVICE}.fetch_currency", return_value="USD"),
            patch(f"{SERVICE}.fetch_current_price", side_effect=prices.get),
        ):
            assert latest_price_in_thb("NVDA") == Decimal("6800.0000")

    def test_no_price_is_none(self) -> None:
        """No market price → None (the holding keeps its stored value)."""
        with patch(f"{SERVICE}.fetch_current_price", return_value=None):
            assert latest_price_in_thb("NVDA") is None

    def test_no_rate_is_none(self) -> None:
        """A price without a rate is not usable in baht."""
        prices = {"NVDA": Decimal("200")}
        with (
            patch(f"{SERVICE}.fetch_currency", return_value="USD"),
            patch(f"{SERVICE}.fetch_current_price", side_effect=prices.get),
        ):
            assert latest_price_in_thb("NVDA") is None
