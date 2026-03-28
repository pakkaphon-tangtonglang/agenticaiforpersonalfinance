"""Tests for market data service functions (Bright Data SERP API)."""

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
    _format_news_from_organic,
    _to_decimal,
    _validate_currency_code,
    convert_currency,
    fetch_finance_news,
    fetch_stock_dashboard,
)

SERVICE_PATH = "finance_ai.tools.market_data_service"

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

    def test_uses_price_field(self) -> None:
        """Should extract from 'price' key."""
        info = {"price": "35.50"}
        assert _extract_price(info) == Decimal("35.50")

    def test_uses_value_field(self) -> None:
        """Should use 'value' key as fallback."""
        info = {"value": "$34.00"}
        assert _extract_price(info) == Decimal("34")

    def test_returns_none_when_no_price(self) -> None:
        """Should return None when no price fields exist."""
        assert _extract_price({}) is None


# ---------------------------------------------------------------------------
# Helper: _calculate_dividend_yield
# ---------------------------------------------------------------------------


class TestCalculateDividendYield:
    """Tests for _calculate_dividend_yield helper."""

    def test_parses_percentage_string(self) -> None:
        """'3.50%' should become Decimal('3.50')."""
        info = {"dividend_yield": "3.50%"}
        assert _calculate_dividend_yield(info) == Decimal("3.50")

    def test_parses_plain_number(self) -> None:
        """'2.5' should become Decimal('2.5')."""
        info = {"dividend_yield": "2.5"}
        assert _calculate_dividend_yield(info) == Decimal("2.5")

    def test_returns_zero_when_missing(self) -> None:
        """Should return 0 when dividend_yield is absent."""
        assert _calculate_dividend_yield({}) == Decimal("0")

    def test_returns_zero_for_none(self) -> None:
        """Should return 0 when dividend_yield is None."""
        info = {"dividend_yield": None}
        assert _calculate_dividend_yield(info) == Decimal("0")


# ---------------------------------------------------------------------------
# fetch_stock_dashboard
# ---------------------------------------------------------------------------


class TestFetchStockDashboard:
    """Tests for fetch_stock_dashboard function."""

    @patch(f"{SERVICE_PATH}._serp_request")
    def test_returns_full_dashboard(self, mock_serp: MagicMock) -> None:
        """Should return populated StockDashboardResult."""
        mock_serp.return_value = {
            "knowledge": {
                "title": "PTT Public Company Limited",
                "price": "35.50",
                "currency": "THB",
                "52_week_high": "42.00",
                "52_week_low": "28.00",
                "pe_ratio": "12.5",
                "market_cap": "1,000,000,000",
                "dividend_yield": "3.5%",
                "target_price": "40.00",
                "recommendation": "buy",
            },
            "organic": [],
        }

        result = fetch_stock_dashboard("PTT.BK")

        assert isinstance(result, StockDashboardResult)
        assert result.name == "PTT Public Company Limited"
        assert result.current_price == Decimal("35.50")
        assert result.pe_ratio == Decimal("12.5")
        assert result.dividend_yield_percent == Decimal("3.5")
        assert result.recommendation == "buy"

    @patch(f"{SERVICE_PATH}._serp_request")
    def test_handles_empty_knowledge(self, mock_serp: MagicMock) -> None:
        """Should return empty model when knowledge is empty."""
        mock_serp.return_value = {"knowledge": {}, "organic": []}

        result = fetch_stock_dashboard("INVALID")

        assert result.name is None
        assert result.current_price is None
        assert result.dividend_yield_percent == Decimal("0")

    @patch(f"{SERVICE_PATH}._serp_request", return_value=None)
    def test_handles_serp_failure(self, mock_serp: MagicMock) -> None:
        """Should return empty model when SERP returns None."""
        result = fetch_stock_dashboard("FAIL")

        assert isinstance(result, StockDashboardResult)
        assert result.name is None


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

    @patch(f"{SERVICE_PATH}._fetch_exchange_rate")
    def test_converts_usd_to_thb(self, mock_rate: MagicMock) -> None:
        """Should return correct conversion result."""
        mock_rate.return_value = Decimal("34.5")

        result = convert_currency("USD", "THB", Decimal("100"))

        assert isinstance(result, CurrencyConversionResult)
        assert result.from_currency == "USD"
        assert result.to_currency == "THB"
        assert result.exchange_rate == Decimal("34.5")
        assert result.converted_amount == Decimal("3450.0")

    @patch(f"{SERVICE_PATH}._fetch_exchange_rate")
    def test_decimal_precision(self, mock_rate: MagicMock) -> None:
        """Should maintain Decimal precision in calculation."""
        mock_rate.return_value = Decimal("0.2345")

        result = convert_currency("JPY", "THB", Decimal("10000"))
        assert result.exchange_rate == Decimal("0.2345")

    @patch(
        f"{SERVICE_PATH}._fetch_exchange_rate",
        side_effect=ValueError("ไม่พบอัตราแลกเปลี่ยน"),
    )
    def test_raises_when_no_rate(self, mock_rate: MagicMock) -> None:
        """Should raise ValueError when rate is unavailable."""
        with pytest.raises(ValueError, match="ไม่พบอัตราแลกเปลี่ยน"):
            convert_currency("USD", "THB", Decimal("100"))

    def test_raises_for_invalid_currency(self) -> None:
        """Should raise ValueError for invalid currency codes."""
        with pytest.raises(ValueError, match="Invalid from_currency"):
            convert_currency("XX", "THB", Decimal("100"))


# ---------------------------------------------------------------------------
# _format_news_from_organic
# ---------------------------------------------------------------------------


class TestFormatNewsFromOrganic:
    """Tests for _format_news_from_organic helper."""

    def test_formats_articles(self) -> None:
        """Should format organic results into readable news."""
        organic = [
            {
                "title": "AAPL Reports Record Earnings",
                "source": "Reuters",
                "description": "Apple Inc reported...",
                "link": "https://example.com/article",
            },
        ]
        result = _format_news_from_organic(organic, "AAPL")
        assert "AAPL Reports Record Earnings" in result
        assert "Reuters" in result
        assert "AAPL" in result

    def test_returns_empty_for_no_results(self) -> None:
        """Should return empty string for empty organic list."""
        assert _format_news_from_organic([], "AAPL") == ""

    def test_limits_to_five_articles(self) -> None:
        """Should only include up to 5 articles."""
        organic = [{"title": f"Article {i}", "description": f"Desc {i}"} for i in range(10)]
        result = _format_news_from_organic(organic, "TEST")
        assert result.count("**Article") == 5


# ---------------------------------------------------------------------------
# fetch_finance_news
# ---------------------------------------------------------------------------


class TestFetchFinanceNews:
    """Tests for fetch_finance_news function."""

    @patch(f"{SERVICE_PATH}._serp_request")
    def test_returns_news_content(self, mock_serp: MagicMock) -> None:
        """Should return news when organic results available."""
        mock_serp.return_value = {
            "knowledge": {},
            "organic": [
                {
                    "title": "Apple reports Q4 earnings",
                    "source": "CNBC",
                    "description": "Apple Inc reported record...",
                },
            ],
        }

        result = fetch_finance_news("AAPL")

        assert isinstance(result, FinanceNewsResult)
        assert result.has_news is True
        assert "Apple" in result.news_content

    @patch(f"{SERVICE_PATH}._serp_request")
    def test_handles_no_news(self, mock_serp: MagicMock) -> None:
        """Should set has_news=False when no organic results."""
        mock_serp.return_value = {"knowledge": {}, "organic": []}

        result = fetch_finance_news("XYZ")
        assert result.has_news is False

    @patch(f"{SERVICE_PATH}._serp_request", return_value=None)
    def test_handles_serp_failure(self, mock_serp: MagicMock) -> None:
        """Should return graceful result when SERP fails."""
        result = fetch_finance_news("AAPL")

        assert result.has_news is False
        assert "ข้อผิดพลาด" in result.news_content

    @patch(f"{SERVICE_PATH}._serp_request")
    def test_handles_empty_articles(self, mock_serp: MagicMock) -> None:
        """Should handle organic results with empty content."""
        mock_serp.return_value = {
            "knowledge": {},
            "organic": [{}],
        }
        result = fetch_finance_news("FAIL")
        assert result.has_news is False
