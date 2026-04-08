"""Tests for market data tool wrappers."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

from finance_ai.agents.market_data_tools import (
    MARKET_DATA_TOOLS,
    get_stock_price,
    search_finance_news,
)


class TestMarketDataToolsList:
    """Tests for MARKET_DATA_TOOLS export."""

    def test_contains_two_tools(self) -> None:
        """Should export exactly 2 tools."""
        assert len(MARKET_DATA_TOOLS) == 2

    def test_contains_expected_tools(self) -> None:
        """Should contain get_stock_price and search_finance_news."""
        tool_names = [t.name for t in MARKET_DATA_TOOLS]
        assert "get_stock_price" in tool_names
        assert "search_finance_news" in tool_names


class TestGetStockPriceTool:
    """Tests for get_stock_price @tool wrapper."""

    def test_calls_service_and_returns_dict(self) -> None:
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
            result = get_stock_price.invoke({"symbol": "AAPL"})

        assert result["action"] == "stock_price"
        assert result["symbol"] == "AAPL"

    def test_converts_values_to_strings(self) -> None:
        """Should convert non-None values to strings."""
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "name": "PTT",
            "current_price": Decimal("35.50"),
            "pe_ratio": None,
        }

        with patch(
            "finance_ai.tools.market_data_service.fetch_stock_dashboard",
            return_value=mock_result,
        ):
            result = get_stock_price.invoke({"symbol": "PTT.BK"})

        assert result["name"] == "PTT"
        assert result["current_price"] == "35.50"
        assert result["pe_ratio"] is None


class TestSearchFinanceNewsTool:
    """Tests for search_finance_news @tool wrapper."""

    def test_returns_news_dict(self) -> None:
        """Should return dict with news content."""
        mock_result = MagicMock()
        mock_result.symbol = "AAPL"
        mock_result.news_content = "Apple earnings..."
        mock_result.has_news = True

        with patch(
            "finance_ai.tools.market_data_service.fetch_finance_news",
            return_value=mock_result,
        ):
            result = search_finance_news.invoke({"symbol": "AAPL"})

        assert result["action"] == "finance_news"
        assert result["symbol"] == "AAPL"
        assert result["has_news"] is True

    def test_returns_no_news_flag(self) -> None:
        """Should return has_news=False when no news available."""
        mock_result = MagicMock()
        mock_result.symbol = "UNKNOWN"
        mock_result.news_content = ""
        mock_result.has_news = False

        with patch(
            "finance_ai.tools.market_data_service.fetch_finance_news",
            return_value=mock_result,
        ):
            result = search_finance_news.invoke({"symbol": "UNKNOWN"})

        assert result["has_news"] is False
