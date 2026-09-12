"""Tests for market data service functions (yfinance + Google News RSS)."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import httpx
import pytest

from finance_ai.tools.market_data_constants import NEWS_NOT_FOUND_MESSAGE
from finance_ai.tools.market_data_models import (
    CurrencyConversionResult,
    FinanceNewsResult,
    StockDashboardResult,
)
from finance_ai.tools.market_data_service import (
    _fetch_exchange_rate,
    _parse_rss_items,
    _to_decimal,
    _validate_currency_code,
    convert_currency,
    fetch_finance_news,
    fetch_stock_dashboard,
)

SERVICE_PATH = "finance_ai.tools.market_data_service"

RSS_FEED_TWO_ITEMS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<item>
    <title>PTT reports strong Q4 earnings</title>
    <source url="https://example.com">Bangkok Post</source>
    <link>https://example.com/ptt-q4</link>
    <pubDate>Mon, 12 Sep 2026 09:00:00 GMT</pubDate>
</item>
<item>
    <title>Oil prices rise on supply concerns</title>
    <source>Reuters</source>
    <link>https://example.com/oil</link>
    <pubDate>Mon, 12 Sep 2026 10:30:00 GMT</pubDate>
</item>
</channel></rss>"""

RSS_FEED_EMPTY = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel></channel></rss>"""


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
# fetch_stock_dashboard (yfinance only)
# ---------------------------------------------------------------------------


class TestFetchStockDashboard:
    """Tests for fetch_stock_dashboard function."""

    @patch(f"{SERVICE_PATH}._fetch_dashboard_from_yfinance")
    def test_returns_full_dashboard(self, mock_yf: MagicMock) -> None:
        """Should return the populated yfinance StockDashboardResult."""
        mock_yf.return_value = StockDashboardResult(
            name="PTT Public Company Limited",
            current_price=Decimal("35.50"),
            currency="THB",
            fifty_two_week_high=Decimal("42.00"),
            fifty_two_week_low=Decimal("28.00"),
            pe_ratio=Decimal("12.5"),
            market_cap=Decimal("1000000000"),
            dividend_yield_percent=Decimal("3.50"),
            analyst_target_price=Decimal("40.00"),
            recommendation="buy",
        )

        result = fetch_stock_dashboard("PTT.BK")

        assert isinstance(result, StockDashboardResult)
        assert result.name == "PTT Public Company Limited"
        assert result.current_price == Decimal("35.50")
        assert result.pe_ratio == Decimal("12.5")
        assert result.dividend_yield_percent == Decimal("3.50")
        assert result.recommendation == "buy"

    @patch(f"{SERVICE_PATH}._fetch_dashboard_from_yfinance")
    def test_returns_empty_when_yfinance_fails(self, mock_yf: MagicMock) -> None:
        """Should return an empty model when yfinance yields nothing."""
        mock_yf.return_value = StockDashboardResult()

        result = fetch_stock_dashboard("INVALID")

        assert isinstance(result, StockDashboardResult)
        assert result.name is None
        assert result.current_price is None
        assert result.dividend_yield_percent == Decimal("0")


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
# _fetch_exchange_rate (yfinance FX tickers like "USDTHB=X")
# ---------------------------------------------------------------------------


class TestFetchExchangeRate:
    """Tests for _fetch_exchange_rate function."""

    @patch("yfinance.Ticker")
    def test_returns_rate_from_yfinance(self, mock_ticker_cls: MagicMock) -> None:
        """Should convert yfinance last_price to a Decimal rate."""
        ticker = MagicMock()
        ticker.fast_info.last_price = 34.5
        mock_ticker_cls.return_value = ticker

        rate = _fetch_exchange_rate("USD", "THB")

        assert rate == Decimal("34.5")
        mock_ticker_cls.assert_called_once_with("USDTHB=X")

    @patch("yfinance.Ticker")
    def test_raises_fetch_error_on_exception(self, mock_ticker_cls: MagicMock) -> None:
        """Should raise Thai ValueError when yfinance fails."""
        mock_ticker_cls.side_effect = RuntimeError("network down")

        with pytest.raises(ValueError, match="ไม่สามารถดึงอัตราแลกเปลี่ยน USD/THB"):
            _fetch_exchange_rate("USD", "THB")

    @patch("yfinance.Ticker")
    def test_raises_not_found_when_rate_missing(self, mock_ticker_cls: MagicMock) -> None:
        """Should raise Thai ValueError when no rate is returned."""
        ticker = MagicMock()
        ticker.fast_info.last_price = None
        mock_ticker_cls.return_value = ticker

        with pytest.raises(ValueError, match="ไม่พบอัตราแลกเปลี่ยน"):
            _fetch_exchange_rate("USD", "THB")


# ---------------------------------------------------------------------------
# _parse_rss_items (Google News RSS)
# ---------------------------------------------------------------------------


class TestParseRssItems:
    """Tests for _parse_rss_items helper."""

    def test_parses_two_items(self) -> None:
        """Should extract title/source/link/pubDate for each item."""
        items = _parse_rss_items(RSS_FEED_TWO_ITEMS)

        assert len(items) == 2
        assert items[0]["title"] == "PTT reports strong Q4 earnings"
        assert items[0]["source"] == "Bangkok Post"
        assert items[0]["link"] == "https://example.com/ptt-q4"
        assert items[0]["pubDate"] == "Mon, 12 Sep 2026 09:00:00 GMT"
        assert items[1]["title"] == "Oil prices rise on supply concerns"
        assert items[1]["source"] == "Reuters"

    def test_empty_feed_returns_no_items(self) -> None:
        """Should return an empty list for a feed without items."""
        assert _parse_rss_items(RSS_FEED_EMPTY) == []


# ---------------------------------------------------------------------------
# fetch_finance_news (Google News RSS)
# ---------------------------------------------------------------------------


class TestFetchFinanceNews:
    """Tests for fetch_finance_news function."""

    @patch(f"{SERVICE_PATH}._fetch_news_rss")
    def test_formats_rss_items(self, mock_rss: MagicMock) -> None:
        """Should format RSS items into the Thai news template."""
        mock_rss.return_value = RSS_FEED_TWO_ITEMS

        result = fetch_finance_news("PTT.BK")

        assert isinstance(result, FinanceNewsResult)
        assert result.has_news is True
        assert result.news_content.startswith("ข่าวล่าสุดสำหรับ PTT.BK:")
        assert "**PTT reports strong Q4 earnings**" in result.news_content
        assert "แหล่งที่มา: Bangkok Post" in result.news_content
        assert "https://example.com/ptt-q4" in result.news_content
        assert "Mon, 12 Sep 2026 09:00:00 GMT" in result.news_content
        assert "\n\n---\n\n" in result.news_content

    @patch(f"{SERVICE_PATH}._fetch_news_rss")
    def test_limits_to_five_articles(self, mock_rss: MagicMock) -> None:
        """Should include at most 5 articles."""
        items = "".join(
            f"<item><title>Article {i}</title><source>S</source>"
            f"<link>https://example.com/{i}</link>"
            f"<pubDate>Mon, 12 Sep 2026 0{i}:00:00 GMT</pubDate></item>"
            for i in range(7)
        )
        mock_rss.return_value = f"<rss><channel>{items}</channel></rss>"

        result = fetch_finance_news("TEST.BK")

        assert result.news_content.count("**Article") == 5

    @patch(f"{SERVICE_PATH}._fetch_news_rss")
    def test_empty_feed_returns_not_found(self, mock_rss: MagicMock) -> None:
        """Should return the Thai not-found message for an empty feed."""
        mock_rss.return_value = RSS_FEED_EMPTY

        result = fetch_finance_news("XYZ")

        assert result.has_news is False
        assert result.news_content == NEWS_NOT_FOUND_MESSAGE.format(symbol="XYZ")

    @patch(f"{SERVICE_PATH}._fetch_news_rss")
    def test_fetch_failure_returns_not_found(self, mock_rss: MagicMock) -> None:
        """Should never raise — fetch errors degrade to the not-found message."""
        mock_rss.side_effect = httpx.HTTPError("network down")

        result = fetch_finance_news("AAPL")

        assert result.has_news is False
        assert result.news_content == NEWS_NOT_FOUND_MESSAGE.format(symbol="AAPL")

    @patch(f"{SERVICE_PATH}._fetch_news_rss")
    def test_invalid_xml_returns_not_found(self, mock_rss: MagicMock) -> None:
        """Should return the not-found message when XML parsing fails."""
        mock_rss.return_value = "this is not xml"

        result = fetch_finance_news("AAPL")

        assert result.has_news is False
        assert result.news_content == NEWS_NOT_FOUND_MESSAGE.format(symbol="AAPL")
