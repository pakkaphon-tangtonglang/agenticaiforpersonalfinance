"""Tests for price client (yfinance-backed, free — no API key).

Mocks yfinance.Ticker at the boundary so no real network calls are made.
The module-level price cache is cleared around every test so cached
results from one test never leak into another.
"""

from decimal import Decimal
from typing import Iterator
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from pandas import DataFrame

from finance_ai.tools.price_client import (
    _PRICE_CACHE,
    fetch_currency,
    fetch_current_price,
    fetch_multiple_prices,
    is_valid_ticker,
)

MODULE_PATH = "finance_ai.tools.price_client"
YFINANCE_TICKER_PATH = "yfinance.Ticker"


@pytest.fixture(autouse=True)
def clear_price_cache() -> Iterator[None]:
    """Isolate the module-level price cache between tests."""
    _PRICE_CACHE.clear()
    yield
    _PRICE_CACHE.clear()


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

    @patch(f"{MODULE_PATH}._retry_sleep")
    @patch(YFINANCE_TICKER_PATH)
    def test_returns_none_when_price_missing(
        self, mock_ticker_cls: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Returns None when yfinance gives no price from any source."""
        mock_ticker_cls.return_value = make_ticker(None)
        mock_ticker_cls.return_value.history.return_value = DataFrame({"Close": []})

        assert fetch_current_price("BADSIGNAL") is None

    @patch(f"{MODULE_PATH}._retry_sleep")
    @patch(YFINANCE_TICKER_PATH)
    def test_returns_none_on_ticker_exception(
        self, mock_ticker_cls: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Returns None when yfinance.Ticker() itself raises."""
        mock_ticker_cls.side_effect = RuntimeError("network down")

        assert fetch_current_price("PTT.BK") is None

    @patch(f"{MODULE_PATH}._retry_sleep")
    @patch(YFINANCE_TICKER_PATH)
    def test_returns_none_when_fast_info_raises(
        self, mock_ticker_cls: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Returns None when accessing fast_info.last_price raises."""
        ticker = MagicMock()
        type(ticker).fast_info = PropertyMock(side_effect=RuntimeError("boom"))
        ticker.history.return_value = DataFrame({"Close": []})
        mock_ticker_cls.return_value = ticker

        assert fetch_current_price("PTT.BK") is None

    @patch(f"{MODULE_PATH}._retry_sleep")
    @patch(YFINANCE_TICKER_PATH)
    def test_falls_back_to_history_close(
        self, mock_ticker_cls: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Uses the last chart close when the quote endpoint returns None."""
        ticker = make_ticker(None)
        ticker.history.return_value = DataFrame({"Close": [30.0, 42.25]})
        mock_ticker_cls.return_value = ticker

        assert fetch_current_price("PTT.BK") == Decimal("42.25")

    @patch(f"{MODULE_PATH}._retry_sleep")
    @patch(YFINANCE_TICKER_PATH)
    def test_history_empty_dataframe_returns_none(
        self, mock_ticker_cls: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """An empty chart history still degrades to None."""
        ticker = make_ticker(None)
        ticker.history.return_value = DataFrame({"Close": []})
        mock_ticker_cls.return_value = ticker

        assert fetch_current_price("PTT.BK") is None


class TestPriceRetry:
    """Tests for the retry-with-backoff loop."""

    @patch(f"{MODULE_PATH}._retry_sleep")
    @patch(f"{MODULE_PATH}._fetch_price_any_source")
    def test_retries_then_succeeds(self, mock_source: MagicMock, mock_sleep: MagicMock) -> None:
        """A transient failure on round 1 succeeds on round 2."""
        mock_source.side_effect = [None, Decimal("35.50")]

        assert fetch_current_price("PTT.BK") == Decimal("35.50")
        assert mock_source.call_count == 2
        mock_sleep.assert_called_once()

    @patch(f"{MODULE_PATH}._retry_sleep")
    @patch(f"{MODULE_PATH}._fetch_price_any_source")
    def test_gives_up_after_max_attempts(
        self, mock_source: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Exhausts PRICE_RETRY_ATTEMPTS rounds before returning None."""
        mock_source.return_value = None

        assert fetch_current_price("PTT.BK") is None
        assert mock_source.call_count == 2
        mock_sleep.assert_called_once()

    @patch(f"{MODULE_PATH}._retry_sleep")
    @patch(f"{MODULE_PATH}._fetch_price_any_source")
    def test_no_sleep_after_success(self, mock_source: MagicMock, mock_sleep: MagicMock) -> None:
        """A first-round success never sleeps."""
        mock_source.return_value = Decimal("35.50")

        assert fetch_current_price("PTT.BK") == Decimal("35.50")
        mock_sleep.assert_not_called()


class TestPriceCaching:
    """Tests for the module-level TTL cache behaviour."""

    @patch(f"{MODULE_PATH}._fetch_price_with_retry")
    def test_cached_price_skips_refetch(self, mock_retry: MagicMock) -> None:
        """A second call within the TTL is served from cache."""
        mock_retry.return_value = Decimal("35.50")
        first = fetch_current_price("PTT.BK")
        mock_retry.return_value = Decimal("99.00")
        second = fetch_current_price("PTT.BK")

        assert first == second == Decimal("35.50")
        mock_retry.assert_called_once_with("PTT.BK")

    @patch(f"{MODULE_PATH}._fetch_price_with_retry")
    def test_failed_fetch_is_not_cached(self, mock_retry: MagicMock) -> None:
        """A None result is not cached, so the next call refetches."""
        mock_retry.return_value = None
        assert fetch_current_price("PTT.BK") is None
        mock_retry.return_value = Decimal("35.50")

        assert fetch_current_price("PTT.BK") == Decimal("35.50")
        assert mock_retry.call_count == 2


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
