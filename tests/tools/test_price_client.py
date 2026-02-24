"""Tests for price client (yfinance wrapper)."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from finance_ai.tools.price_client import (
    fetch_current_price,
    fetch_multiple_prices,
    is_valid_ticker,
)

MODULE_PATH = "finance_ai.tools.price_client"


@pytest.fixture
def mock_yfinance() -> MagicMock:
    """Create a mock yfinance module."""
    mock_yf = MagicMock()
    return mock_yf


class TestFetchCurrentPrice:
    """Tests for fetch_current_price."""

    @patch(f"{MODULE_PATH}._import_yfinance")
    def test_returns_price_from_current_price(self, mock_import: MagicMock) -> None:
        """Returns Decimal from currentPrice field."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.info = {"currentPrice": 42.50}
        mock_yf.Ticker.return_value = mock_ticker
        mock_import.return_value = mock_yf
        result = fetch_current_price("PTT.BK")
        assert result == Decimal("42.5")
        mock_yf.Ticker.assert_called_once_with("PTT.BK")

    @patch(f"{MODULE_PATH}._import_yfinance")
    def test_falls_back_to_regular_market_price(self, mock_import: MagicMock) -> None:
        """Uses regularMarketPrice when currentPrice is missing."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.info = {"regularMarketPrice": 35.00}
        mock_yf.Ticker.return_value = mock_ticker
        mock_import.return_value = mock_yf
        result = fetch_current_price("AOT.BK")
        assert result == Decimal("35")

    @patch(f"{MODULE_PATH}._import_yfinance")
    def test_returns_none_when_no_price(self, mock_import: MagicMock) -> None:
        """Returns None when no price fields available."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_yf.Ticker.return_value = mock_ticker
        mock_import.return_value = mock_yf
        result = fetch_current_price("INVALID")
        assert result is None

    @patch(f"{MODULE_PATH}._import_yfinance")
    def test_returns_none_on_exception(self, mock_import: MagicMock) -> None:
        """Returns None on unexpected error."""
        mock_yf = MagicMock()
        mock_yf.Ticker.side_effect = RuntimeError("network error")
        mock_import.return_value = mock_yf
        result = fetch_current_price("PTT.BK")
        assert result is None

    @patch(f"{MODULE_PATH}._import_yfinance")
    def test_returns_none_when_info_is_none(self, mock_import: MagicMock) -> None:
        """Returns None when ticker.info is None."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.info = None
        mock_yf.Ticker.return_value = mock_ticker
        mock_import.return_value = mock_yf
        result = fetch_current_price("PTT.BK")
        assert result is None


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
        result = fetch_multiple_prices([])
        assert result == {}


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
