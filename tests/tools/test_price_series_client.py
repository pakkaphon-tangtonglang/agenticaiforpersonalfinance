"""Tests for price_series_client (daily closes from yfinance for trend charts).

yfinance.Ticker is mocked at the boundary; the module cache is cleared
around every test so one test's series never leaks into another.
"""

from datetime import date
from decimal import Decimal
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest
from pandas import DataFrame, DatetimeIndex

from finance_ai.tools.price_series_client import PricePoint, clear_series_cache, fetch_price_series

YFINANCE_TICKER_PATH = "yfinance.Ticker"


@pytest.fixture(autouse=True)
def isolated_cache() -> Iterator[None]:
    """Start and end every test with an empty series cache."""
    clear_series_cache()
    yield
    clear_series_cache()


def history_frame(closes: dict[str, object]) -> DataFrame:
    """A yfinance-style history frame indexed by Bangkok timestamps."""
    index = DatetimeIndex(list(closes.keys())).tz_localize("Asia/Bangkok")
    return DataFrame({"Close": list(closes.values())}, index=index)


def ticker_returning(frame: object) -> MagicMock:
    """A mocked Ticker whose history() returns the given frame."""
    ticker = MagicMock()
    ticker.history.return_value = frame
    return ticker


class TestFetchPriceSeries:
    """fetch_price_series: closes as Decimal points, oldest first."""

    @patch(YFINANCE_TICKER_PATH)
    def test_returns_daily_closes_as_points(self, mock_ticker: MagicMock) -> None:
        """Each close becomes a dated Decimal point (Thai stock, THB)."""
        mock_ticker.return_value = ticker_returning(
            history_frame({"2026-10-29": 42.25, "2026-10-30": 42.0})
        )

        series = fetch_price_series("PTT.BK", days=30)

        assert series == [
            PricePoint(day=date(2026, 10, 29), value=Decimal("42.2500")),
            PricePoint(day=date(2026, 10, 30), value=Decimal("42.0000")),
        ]
        mock_ticker.assert_called_once_with("PTT.BK")

    @patch(YFINANCE_TICKER_PATH)
    def test_holiday_gaps_are_skipped_not_filled(self, mock_ticker: MagicMock) -> None:
        """Missing and NaN days (SET holidays) are simply absent."""
        mock_ticker.return_value = ticker_returning(
            history_frame({"2026-04-10": 30.0, "2026-04-13": float("nan"), "2026-04-16": 31.0})
        )

        days = [point.day for point in fetch_price_series("PTT.BK")]

        assert days == [date(2026, 4, 10), date(2026, 4, 16)]

    @patch(YFINANCE_TICKER_PATH)
    def test_empty_history_is_empty_list(self, mock_ticker: MagicMock) -> None:
        """No rows → [] (caller hides the line)."""
        mock_ticker.return_value = ticker_returning(DataFrame())

        assert not fetch_price_series("NOPE.BK")

    @patch(YFINANCE_TICKER_PATH)
    def test_network_error_is_empty_list(self, mock_ticker: MagicMock) -> None:
        """A yfinance failure (rate limit on Render) degrades to []."""
        mock_ticker.return_value.history.side_effect = RuntimeError("429 Too Many Requests")

        assert not fetch_price_series("PTT.BK")

    @patch(YFINANCE_TICKER_PATH)
    def test_series_is_cached(self, mock_ticker: MagicMock) -> None:
        """A second call within the TTL does not hit yfinance again."""
        mock_ticker.return_value = ticker_returning(history_frame({"2026-10-30": 42.0}))

        fetch_price_series("PTT.BK")
        fetch_price_series("PTT.BK")

        assert mock_ticker.call_count == 1

    @patch(YFINANCE_TICKER_PATH)
    def test_failures_are_not_cached(self, mock_ticker: MagicMock) -> None:
        """An empty answer is retried on the next call."""
        mock_ticker.return_value = ticker_returning(DataFrame())
        fetch_price_series("PTT.BK")
        mock_ticker.return_value = ticker_returning(history_frame({"2026-10-30": 42.0}))

        assert len(fetch_price_series("PTT.BK")) == 1

    def test_rejects_non_positive_days(self) -> None:
        """days must be at least 1."""
        with pytest.raises(ValueError, match="days must be at least 1. Received: 0"):
            fetch_price_series("PTT.BK", days=0)
