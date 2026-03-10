"""Tests for market data service functions."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from finance_ai.tools.market_data_models import (
    CurrencyConversionResult,
    FinanceNewsResult,
    StockDashboardResult,
)
from finance_ai.tools.market_data_service import (
    _calculate_dividend_yield,
    _extract_price,
    _to_decimal,
    _validate_currency_code,
    convert_currency,
    fetch_finance_news,
    fetch_stock_dashboard,
)

# ---------------------------------------------------------------------------
# Helper: _to_decimal
# ---------------------------------------------------------------------------


class TestToDecimal:
    """Tests for _to_decimal helper."""

    def test_converts_float(self) -> None:
        """Should convert float to Decimal."""
        assert _to_decimal(35.5) == Decimal("35.5")

    def test_converts_int(self) -> None:
        """Should convert int to Decimal."""
        assert _to_decimal(100) == Decimal("100")

    def test_returns_none_for_none(self) -> None:
        """Should return None for None input."""
        assert _to_decimal(None) is None

    def test_returns_none_for_invalid(self) -> None:
        """Should return None for non-numeric strings."""
        assert _to_decimal("not_a_number") is None


# ---------------------------------------------------------------------------
# Helper: _extract_price
# ---------------------------------------------------------------------------


class TestExtractPrice:
    """Tests for _extract_price helper."""

    def test_uses_current_price(self) -> None:
        """Should prefer currentPrice field."""
        info = {"currentPrice": 35.5, "regularMarketPrice": 34.0}
        assert _extract_price(info) == Decimal("35.5")

    def test_falls_back_to_regular_market_price(self) -> None:
        """Should use regularMarketPrice when currentPrice is absent."""
        info = {"regularMarketPrice": 34.0}
        assert _extract_price(info) == Decimal("34")

    def test_returns_none_when_no_price(self) -> None:
        """Should return None when no price fields exist."""
        assert _extract_price({}) is None


# ---------------------------------------------------------------------------
# Helper: _calculate_dividend_yield
# ---------------------------------------------------------------------------


class TestCalculateDividendYield:
    """Tests for _calculate_dividend_yield helper."""

    def test_converts_fraction_to_percent(self) -> None:
        """0.035 should become 3.5%."""
        info = {"dividendYield": 0.035}
        assert _calculate_dividend_yield(info) == Decimal("3.5")

    def test_returns_zero_when_missing(self) -> None:
        """Should return 0 when dividendYield is absent."""
        assert _calculate_dividend_yield({}) == Decimal("0")

    def test_returns_zero_for_none(self) -> None:
        """Should return 0 when dividendYield is None."""
        info = {"dividendYield": None}
        assert _calculate_dividend_yield(info) == Decimal("0")


# ---------------------------------------------------------------------------
# fetch_stock_dashboard
# ---------------------------------------------------------------------------


class TestFetchStockDashboard:
    """Tests for fetch_stock_dashboard function."""

    @patch("finance_ai.tools.market_data_service._import_yfinance")
    def test_returns_full_dashboard(self, mock_yf: MagicMock) -> None:
        """Should return populated StockDashboardResult."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "longName": "PTT Public Company Limited",
            "currentPrice": 35.5,
            "currency": "THB",
            "fiftyTwoWeekHigh": 42.0,
            "fiftyTwoWeekLow": 28.0,
            "trailingPE": 12.5,
            "marketCap": 1000000000,
            "dividendYield": 0.035,
            "targetMeanPrice": 40.0,
            "recommendationKey": "buy",
        }
        mock_yf.return_value.Ticker.return_value = mock_ticker

        result = fetch_stock_dashboard("PTT.BK")

        assert isinstance(result, StockDashboardResult)
        assert result.name == "PTT Public Company Limited"
        assert result.current_price == Decimal("35.5")
        assert result.pe_ratio == Decimal("12.5")
        assert result.dividend_yield_percent == Decimal("3.5")
        assert result.recommendation == "buy"

    @patch("finance_ai.tools.market_data_service._import_yfinance")
    def test_handles_empty_info(self, mock_yf: MagicMock) -> None:
        """Should return empty model when info is empty."""
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_yf.return_value.Ticker.return_value = mock_ticker

        result = fetch_stock_dashboard("INVALID")

        assert result.name is None
        assert result.current_price is None
        assert result.dividend_yield_percent == Decimal("0")

    @patch("finance_ai.tools.market_data_service._import_yfinance")
    def test_handles_exception(self, mock_yf: MagicMock) -> None:
        """Should return empty model when yfinance raises."""
        mock_yf.return_value.Ticker.side_effect = RuntimeError("Network")

        result = fetch_stock_dashboard("FAIL")

        assert isinstance(result, StockDashboardResult)
        assert result.name is None

    @patch("finance_ai.tools.market_data_service._import_yfinance")
    def test_uses_regular_market_price_fallback(self, mock_yf: MagicMock) -> None:
        """Should fallback to regularMarketPrice."""
        mock_ticker = MagicMock()
        mock_ticker.info = {"regularMarketPrice": 34.0}
        mock_yf.return_value.Ticker.return_value = mock_ticker

        result = fetch_stock_dashboard("PTT.BK")
        assert result.current_price == Decimal("34")


# ---------------------------------------------------------------------------
# _validate_currency_code
# ---------------------------------------------------------------------------


class TestValidateCurrencyCode:
    """Tests for _validate_currency_code helper."""

    def test_valid_code(self) -> None:
        """Should normalize valid codes to uppercase."""
        assert _validate_currency_code("usd", "test") == "USD"

    def test_strips_whitespace(self) -> None:
        """Should strip whitespace."""
        assert _validate_currency_code(" THB ", "test") == "THB"

    def test_rejects_too_short(self) -> None:
        """Should reject codes shorter than 3 chars."""
        with pytest.raises(ValueError, match="Invalid test"):
            _validate_currency_code("US", "test")

    def test_rejects_numeric(self) -> None:
        """Should reject codes with numbers."""
        with pytest.raises(ValueError, match="Invalid test"):
            _validate_currency_code("U2D", "test")


# ---------------------------------------------------------------------------
# convert_currency
# ---------------------------------------------------------------------------


class TestConvertCurrency:
    """Tests for convert_currency function."""

    @patch("finance_ai.tools.market_data_service._import_yfinance")
    def test_converts_usd_to_thb(self, mock_yf: MagicMock) -> None:
        """Should return correct conversion result."""
        mock_ticker = MagicMock()
        mock_ticker.info = {"regularMarketPrice": 34.5}
        mock_yf.return_value.Ticker.return_value = mock_ticker

        result = convert_currency("USD", "THB", Decimal("100"))

        assert isinstance(result, CurrencyConversionResult)
        assert result.from_currency == "USD"
        assert result.to_currency == "THB"
        assert result.exchange_rate == Decimal("34.5")
        assert result.converted_amount == Decimal("3450.0")

    @patch("finance_ai.tools.market_data_service._import_yfinance")
    def test_uses_previous_close_fallback(self, mock_yf: MagicMock) -> None:
        """Should fallback to previousClose when no regularMarketPrice."""
        mock_ticker = MagicMock()
        mock_ticker.info = {"previousClose": 34.0}
        mock_yf.return_value.Ticker.return_value = mock_ticker

        result = convert_currency("USD", "THB", Decimal("1"))
        assert result.exchange_rate == Decimal("34")

    @patch("finance_ai.tools.market_data_service._import_yfinance")
    def test_raises_when_no_rate(self, mock_yf: MagicMock) -> None:
        """Should raise ValueError when rate is unavailable."""
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_yf.return_value.Ticker.return_value = mock_ticker

        with pytest.raises(ValueError, match="ไม่พบอัตราแลกเปลี่ยน"):
            convert_currency("USD", "THB", Decimal("100"))

    def test_raises_for_invalid_currency(self) -> None:
        """Should raise ValueError for invalid currency codes."""
        with pytest.raises(ValueError, match="Invalid from_currency"):
            convert_currency("XX", "THB", Decimal("100"))

    @patch("finance_ai.tools.market_data_service._import_yfinance")
    def test_decimal_precision(self, mock_yf: MagicMock) -> None:
        """Should maintain Decimal precision in calculation."""
        mock_ticker = MagicMock()
        mock_ticker.info = {"regularMarketPrice": 0.2345}
        mock_yf.return_value.Ticker.return_value = mock_ticker

        result = convert_currency("JPY", "THB", Decimal("10000"))
        assert result.exchange_rate == Decimal("0.2345")


# ---------------------------------------------------------------------------
# fetch_finance_news
# ---------------------------------------------------------------------------


class TestFetchFinanceNews:
    """Tests for fetch_finance_news function."""

    @patch("finance_ai.tools.market_data_service._import_yahoo_news_tool")
    def test_returns_news_content(self, mock_import: MagicMock) -> None:
        """Should return news when available."""
        mock_tool = MagicMock()
        mock_tool.return_value.run.return_value = "Apple reports Q4 earnings"
        mock_import.return_value = mock_tool

        result = fetch_finance_news("AAPL")

        assert isinstance(result, FinanceNewsResult)
        assert result.has_news is True
        assert "Apple" in result.news_content

    @patch("finance_ai.tools.market_data_service._import_yahoo_news_tool")
    def test_handles_no_news(self, mock_import: MagicMock) -> None:
        """Should set has_news=False when no news found."""
        mock_tool = MagicMock()
        mock_tool.return_value.run.return_value = "No news found for XYZ"
        mock_import.return_value = mock_tool

        result = fetch_finance_news("XYZ")
        assert result.has_news is False

    @patch("finance_ai.tools.market_data_service._import_yahoo_news_tool")
    def test_handles_empty_content(self, mock_import: MagicMock) -> None:
        """Should set has_news=False for empty content."""
        mock_tool = MagicMock()
        mock_tool.return_value.run.return_value = ""
        mock_import.return_value = mock_tool

        result = fetch_finance_news("EMPTY")
        assert result.has_news is False

    @patch(
        "finance_ai.tools.market_data_service._import_yahoo_news_tool",
        side_effect=ImportError("langchain-community not installed"),
    )
    def test_handles_import_error(self, mock_import: MagicMock) -> None:
        """Should return graceful result if dependency is missing."""
        result = fetch_finance_news("AAPL")

        assert result.has_news is False
        assert "langchain-community" in result.news_content

    @patch("finance_ai.tools.market_data_service._import_yahoo_news_tool")
    def test_handles_runtime_error(self, mock_import: MagicMock) -> None:
        """Should catch unexpected exceptions gracefully."""
        mock_tool = MagicMock()
        mock_tool.return_value.run.side_effect = RuntimeError("Timeout")
        mock_import.return_value = mock_tool

        result = fetch_finance_news("FAIL")
        assert result.has_news is False
        assert "ข้อผิดพลาด" in result.news_content
