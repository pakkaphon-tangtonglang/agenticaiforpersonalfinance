"""Tests for market data tool wrappers."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from finance_ai.agents.market_data_tools import (
    MARKET_DATA_TOOLS,
    _parse_amount,
    _serialize_conversion,
    _serialize_dashboard,
    convert_currency_tool,
    get_finance_news,
    get_stock_dashboard,
)


class TestMarketDataToolsList:
    """Tests for MARKET_DATA_TOOLS export."""

    def test_contains_three_tools(self) -> None:
        """Should export exactly 3 tools."""
        assert len(MARKET_DATA_TOOLS) == 3

    def test_contains_expected_tools(self) -> None:
        """Should contain all 3 market data tools."""
        tool_names = [t.name for t in MARKET_DATA_TOOLS]
        assert "get_stock_dashboard" in tool_names
        assert "convert_currency_tool" in tool_names
        assert "get_finance_news" in tool_names


class TestParseAmount:
    """Tests for _parse_amount helper."""

    def test_parses_valid_number(self) -> None:
        """Should parse valid number string."""
        assert _parse_amount("100") == Decimal("100")

    def test_parses_decimal_number(self) -> None:
        """Should parse decimal strings."""
        assert _parse_amount("1234.56") == Decimal("1234.56")

    def test_strips_whitespace(self) -> None:
        """Should strip whitespace before parsing."""
        assert _parse_amount(" 50 ") == Decimal("50")

    def test_raises_for_invalid(self) -> None:
        """Should raise ValueError for non-numeric strings."""
        with pytest.raises(ValueError, match="Invalid amount"):
            _parse_amount("abc")


class TestSerializeDashboard:
    """Tests for _serialize_dashboard helper."""

    def test_serializes_all_fields(self) -> None:
        """Should convert model to dict with string values."""
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "name": "PTT",
            "current_price": Decimal("35.50"),
            "pe_ratio": None,
        }

        output = _serialize_dashboard(mock_result, "PTT.BK")

        assert output["action"] == "stock_dashboard"
        assert output["symbol"] == "PTT.BK"
        assert output["name"] == "PTT"
        assert output["current_price"] == "35.50"
        assert output["pe_ratio"] is None


class TestSerializeConversion:
    """Tests for _serialize_conversion helper."""

    def test_serializes_all_fields(self) -> None:
        """Should convert model to dict with string values."""
        mock_result = MagicMock()
        mock_result.from_currency = "USD"
        mock_result.to_currency = "THB"
        mock_result.amount = Decimal("100")
        mock_result.exchange_rate = Decimal("34.5")
        mock_result.converted_amount = Decimal("3450")

        output = _serialize_conversion(mock_result)

        assert output["action"] == "convert_currency"
        assert output["from_currency"] == "USD"
        assert output["converted_amount"] == "3450"


class TestGetStockDashboardTool:
    """Tests for get_stock_dashboard @tool wrapper."""

    @patch(
        "finance_ai.agents.market_data_tools.fetch_stock_dashboard",
        create=True,
    )
    def test_calls_service_and_returns_dict(self, mock_fetch: MagicMock) -> None:
        """Should call service and return serialized dict."""
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "name": "AAPL",
            "current_price": Decimal("180"),
        }

        with patch(
            "finance_ai.tools.market_data_service.fetch_stock_dashboard",
            return_value=mock_result,
        ):
            result = get_stock_dashboard.invoke({"symbol": "AAPL"})

        assert result["action"] == "stock_dashboard"
        assert result["symbol"] == "AAPL"


class TestConvertCurrencyTool:
    """Tests for convert_currency_tool @tool wrapper."""

    @patch("finance_ai.tools.market_data_service.convert_currency")
    def test_parses_amount_and_calls_service(self, mock_convert: MagicMock) -> None:
        """Should parse string amount and call service."""
        mock_result = MagicMock()
        mock_result.from_currency = "USD"
        mock_result.to_currency = "THB"
        mock_result.amount = Decimal("100")
        mock_result.exchange_rate = Decimal("34.5")
        mock_result.converted_amount = Decimal("3450")
        mock_convert.return_value = mock_result

        result = convert_currency_tool.invoke(
            {
                "from_currency": "USD",
                "to_currency": "THB",
                "amount": "100",
            }
        )

        assert result["action"] == "convert_currency"
        assert result["converted_amount"] == "3450"
        mock_convert.assert_called_once_with("USD", "THB", Decimal("100"))


class TestGetFinanceNewsTool:
    """Tests for get_finance_news @tool wrapper."""

    @patch("finance_ai.tools.market_data_service.fetch_finance_news")
    def test_returns_news_dict(self, mock_fetch: MagicMock) -> None:
        """Should return dict with news content."""
        mock_result = MagicMock()
        mock_result.symbol = "AAPL"
        mock_result.news_content = "Apple earnings..."
        mock_result.has_news = True
        mock_fetch.return_value = mock_result

        result = get_finance_news.invoke({"symbol": "AAPL"})

        assert result["action"] == "finance_news"
        assert result["symbol"] == "AAPL"
        assert result["has_news"] is True
