"""Tests for price client (yfinance-backed, free — no API key).

Mocks yfinance.Ticker at the boundary so no real network calls are made.
"""

from decimal import Decimal
from unittest.mock import MagicMock, PropertyMock, patch

from finance_ai.tools.price_client import (
    fetch_currency,
    fetch_current_price,
    fetch_multiple_prices,
    is_valid_ticker,
)

MODULE_PATH = "finance_ai.tools.price_client"
YFINANCE_TICKER_PATH = "yfinance.Ticker"


def make_ticker(price: object) -> MagicMock:
    """Build a MagicMock yfinance Ticker with the given fast_info.last_price.

    Args:
        price: Value for fast_info.last_price (Decimal-convertible or None).

    Returns:
        Mocked yfinance Ticker instance.
    """
    ticker = MagicMock()
    ticker.fast_info.last_price = price
    return ticker


class TestFetchCurrentPrice:
    """Tests for fetch_current_price."""

    @patch(YFINANCE_TICKER_PATH)
    def test_returns_decimal_price(self, mock_ticker_cls: MagicMock) -> None:
        """Converts yfinance last_price float to Decimal."""
        mock_ticker_cls.return_value = make_ticker(35.50)

        result = fetch_current_price("PTT.BK")

        assert result == Decimal("35.50")
        mock_ticker_cls.assert_called_once_with("PTT.BK")

    @patch(YFINANCE_TICKER_PATH)
    def test_converts_string_price(self, mock_ticker_cls: MagicMock) -> None:
        """Converts string last_price values via Decimal(str(price))."""
        mock_ticker_cls.return_value = make_ticker("42.25")

        assert fetch_current_price("AAPL") == Decimal("42.25")

    @patch(YFINANCE_TICKER_PATH)
    def test_returns_none_when_price_missing(self, mock_ticker_cls: MagicMock) -> None:
        """Returns None when yfinance gives no price (None)."""
        mock_ticker_cls.return_value = make_ticker(None)

        assert fetch_current_price("BADSIGNAL") is None

    @patch(YFINANCE_TICKER_PATH)
    def test_returns_none_on_ticker_exception(self, mock_ticker_cls: MagicMock) -> None:
        """Returns None when yfinance.Ticker() itself raises."""
        mock_ticker_cls.side_effect = RuntimeError("network down")

        assert fetch_current_price("PTT.BK") is None

    @patch(YFINANCE_TICKER_PATH)
    def test_returns_none_when_fast_info_raises(self, mock_ticker_cls: MagicMock) -> None:
        """Returns None when accessing fast_info.last_price raises."""
        ticker = MagicMock()
        type(ticker).fast_info = PropertyMock(side_effect=RuntimeError("boom"))
        mock_ticker_cls.return_value = ticker

        assert fetch_current_price("PTT.BK") is None


class TestFetchMultiplePrices:
    """Tests for fetch_multiple_prices."""

    @patch(f"{MODULE_PATH}.fetch_current_price")
    def test_fetches_all_symbols(self, mock_fetch: MagicMock) -> None:
        """Returns dict mapping each symbol to its price."""
        mock_fetch.side_effect = [Decimal("42.50"), Decimal("155.00")]

        result = fetch_multiple_prices(["PTT.BK", "KBANK.BK"])

        assert result == {
            "PTT.BK": Decimal("42.50"),
            "KBANK.BK": Decimal("155.00"),
        }

    @patch(f"{MODULE_PATH}.fetch_current_price")
    def test_handles_failed_symbols(self, mock_fetch: MagicMock) -> None:
        """Returns None for symbols that fail."""
        mock_fetch.side_effect = [Decimal("42.50"), None]

        result = fetch_multiple_prices(["PTT.BK", "INVALID"])

        assert result["PTT.BK"] == Decimal("42.50")
        assert result["INVALID"] is None

    @patch(f"{MODULE_PATH}.fetch_current_price")
    def test_empty_list(self, mock_fetch: MagicMock) -> None:
        """Returns empty dict for empty symbol list."""
        assert fetch_multiple_prices([]) == {}


class TestIsValidTicker:
    """Tests for is_valid_ticker."""

    @patch(f"{MODULE_PATH}.fetch_current_price")
    def test_valid_ticker(self, mock_fetch: MagicMock) -> None:
        """Returns True when price is available."""
        mock_fetch.return_value = Decimal("42.50")

        assert is_valid_ticker("PTT.BK") is True

    @patch(f"{MODULE_PATH}.fetch_current_price")
    def test_invalid_ticker(self, mock_fetch: MagicMock) -> None:
        """Returns False when price is None."""
        mock_fetch.return_value = None

        assert is_valid_ticker("NOTREAL") is False


class TestFetchCurrency:
    """Tests for fetch_currency."""

    @patch(YFINANCE_TICKER_PATH)
    def test_returns_uppercase_currency(self, mock_ticker_cls: MagicMock) -> None:
        """Currency codes are uppercased (e.g. "thb" -> "THB")."""
        ticker = MagicMock()
        ticker.info = {"currency": "thb"}
        mock_ticker_cls.return_value = ticker

        assert fetch_currency("PTT.BK") == "THB"

    @patch(YFINANCE_TICKER_PATH)
    def test_returns_none_when_missing(self, mock_ticker_cls: MagicMock) -> None:
        """A ticker without currency info returns None."""
        ticker = MagicMock()
        ticker.info = {}
        mock_ticker_cls.return_value = ticker

        assert fetch_currency("NOTREAL") is None

    @patch(YFINANCE_TICKER_PATH)
    def test_returns_none_on_exception(self, mock_ticker_cls: MagicMock) -> None:
        """A failing yfinance call degrades to None."""
        mock_ticker_cls.side_effect = RuntimeError("boom")

        assert fetch_currency("AAPL") is None
