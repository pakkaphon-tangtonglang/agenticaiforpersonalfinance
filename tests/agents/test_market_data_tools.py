"""Tests for market data tool wrappers."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

from finance_ai.agents.market_data_tools import (
    MARKET_DATA_TOOLS,
    get_stock_price,
    resolve_asset_symbol,
    search_finance_news,
)
from finance_ai.tools.market_data_models import (
    AssetSymbolMatch,
    StockDashboardResult,
)

SYMBOL_SEARCH_PATH = "finance_ai.tools.symbol_search_service.search_asset_symbols"


class TestMarketDataToolsList:
    """Tests for MARKET_DATA_TOOLS export."""

    def test_contains_three_tools(self) -> None:
        """Should export exactly 3 tools."""
        assert len(MARKET_DATA_TOOLS) == 3

    def test_contains_expected_tools(self) -> None:
        """Should contain the price, news, and symbol search tools."""
        tool_names = [t.name for t in MARKET_DATA_TOOLS]
        assert "get_stock_price" in tool_names
        assert "search_finance_news" in tool_names
        assert "resolve_asset_symbol" in tool_names


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

    def test_includes_price_display_with_baht_unit(self) -> None:
        """Should include a formatted price with หน่วย for SET symbols."""
        dashboard = StockDashboardResult(
            name="PTT Public Company Limited",
            current_price=Decimal("35.50"),
            currency="THB",
        )

        with patch(
            "finance_ai.tools.market_data_service.fetch_stock_dashboard",
            return_value=dashboard,
        ):
            result = get_stock_price.invoke({"symbol": "PTT.BK"})

        assert result["price_display"] == "35.50 บาท"
        assert result["current_price"] == "35.50"

    def test_omits_price_display_without_price(self) -> None:
        """Should omit price_display when no price is available."""
        dashboard = StockDashboardResult(name="PTT")

        with patch(
            "finance_ai.tools.market_data_service.fetch_stock_dashboard",
            return_value=dashboard,
        ):
            result = get_stock_price.invoke({"symbol": "PTT.BK"})

        assert "price_display" not in result

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


class TestResolveAssetSymbolTool:
    """Tests for resolve_asset_symbol @tool wrapper."""

    def test_returns_candidates_dict(self) -> None:
        """Should map search matches into candidate dicts."""
        matches = [
            AssetSymbolMatch(
                symbol="PTT.BK",
                name="PTT Public Company Limited",
                exchange="SET",
                quote_type="EQUITY",
            )
        ]

        with patch(SYMBOL_SEARCH_PATH, return_value=matches):
            result = resolve_asset_symbol.invoke({"query": "หุ้นปตท"})

        assert result["action"] == "resolve_symbol"
        assert result["candidates"] == [
            {
                "symbol": "PTT.BK",
                "name": "PTT Public Company Limited",
                "exchange": "SET",
                "type": "EQUITY",
            }
        ]

    def test_limits_candidates_to_top_five(self) -> None:
        """Should return at most 5 candidates to keep tool output small."""
        matches = [
            AssetSymbolMatch(
                symbol=f"S{i}.BK",
                name=f"Stock Number {i}",
                exchange="SET",
                quote_type="EQUITY",
            )
            for i in range(7)
        ]

        with patch(SYMBOL_SEARCH_PATH, return_value=matches):
            result = resolve_asset_symbol.invoke({"query": "stock"})

        assert len(result["candidates"]) == 5

    def test_empty_candidates_case(self) -> None:
        """Should return an empty candidates list when search finds nothing."""

        with patch(SYMBOL_SEARCH_PATH, return_value=[]):
            result = resolve_asset_symbol.invoke({"query": "zzzz"})

        assert result["action"] == "resolve_symbol"
        assert result["candidates"] == []
