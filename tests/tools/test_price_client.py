"""Tests for price client (Bright Data SERP API wrapper)."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from finance_ai.tools.price_client import (
    _extract_price_from_knowledge,
    _extract_price_from_organic,
    _parse_price_string,
    _serp_request,
    fetch_current_price,
    fetch_multiple_prices,
    is_valid_ticker,
)

MODULE_PATH = "finance_ai.tools.price_client"


class TestParsepriceString:
    """Tests for _parse_price_string helper."""

    def test_simple_number(self) -> None:
        """Parses a simple decimal number."""
        assert _parse_price_string("178.25") == Decimal("178.25")

    def test_with_dollar_sign(self) -> None:
        """Strips dollar sign and parses."""
        assert _parse_price_string("$178.25") == Decimal("178.25")

    def test_with_thousands_separator(self) -> None:
        """Handles comma thousands separators."""
        assert _parse_price_string("1,234.56") == Decimal("1234.56")

    def test_with_currency_suffix(self) -> None:
        """Strips currency text suffix."""
        assert _parse_price_string("35.50 THB") == Decimal("35.50")

    def test_empty_string(self) -> None:
        """Returns None for empty string."""
        assert _parse_price_string("") is None

    def test_no_number(self) -> None:
        """Returns None for non-numeric text."""
        assert _parse_price_string("no price here") is None


class TestExtractPriceFromKnowledge:
    """Tests for _extract_price_from_knowledge."""

    def test_extracts_from_price_field(self) -> None:
        """Extracts price from 'price' key."""
        knowledge = {"price": "178.25"}
        assert _extract_price_from_knowledge(knowledge) == Decimal("178.25")

    def test_extracts_from_value_field(self) -> None:
        """Falls back to 'value' key."""
        knowledge = {"value": "$42.50"}
        assert _extract_price_from_knowledge(knowledge) == Decimal("42.50")

    def test_extracts_from_title(self) -> None:
        """Tries title field as last resort."""
        knowledge = {"title": "178.25 USD"}
        assert _extract_price_from_knowledge(knowledge) == Decimal("178.25")

    def test_returns_none_for_empty(self) -> None:
        """Returns None when no price data."""
        assert _extract_price_from_knowledge({}) is None


class TestExtractPriceFromOrganic:
    """Tests for _extract_price_from_organic."""

    def test_finds_price_in_snippet(self) -> None:
        """Extracts price from search result snippet."""
        organic = [
            {"description": "AMZN stock price is USD 178.25 today"},
        ]
        assert _extract_price_from_organic(organic) == Decimal("178.25")

    def test_returns_none_for_empty(self) -> None:
        """Returns None for empty results."""
        assert _extract_price_from_organic([]) is None

    def test_returns_none_for_no_price(self) -> None:
        """Returns None when snippets have no price."""
        organic = [{"description": "Company news and updates"}]
        assert _extract_price_from_organic(organic) is None


class TestSerpRequest:
    """Tests for _serp_request."""

    @patch(f"{MODULE_PATH}._get_zone", return_value="ai_agent")
    @patch(f"{MODULE_PATH}._get_api_token", return_value="test-token")
    @patch(f"{MODULE_PATH}.httpx.Client")
    def test_successful_request(
        self,
        mock_client_cls: MagicMock,
        mock_token: MagicMock,
        mock_zone: MagicMock,
    ) -> None:
        """Returns parsed JSON on success."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"knowledge": {}, "organic": []}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__ = MagicMock(
            return_value=mock_client,
        )
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = _serp_request("AMZN stock price")
        assert result == {"knowledge": {}, "organic": []}
        mock_client.post.assert_called_once()

    @patch(f"{MODULE_PATH}._get_api_token", side_effect=ValueError("No token"))
    def test_returns_none_on_config_error(
        self,
        mock_token: MagicMock,
    ) -> None:
        """Returns None when token is missing."""
        assert _serp_request("test") is None


class TestFetchCurrentPrice:
    """Tests for fetch_current_price."""

    @patch(f"{MODULE_PATH}._serp_request")
    def test_returns_price_from_knowledge(
        self,
        mock_serp: MagicMock,
    ) -> None:
        """Returns Decimal from knowledge panel price field."""
        mock_serp.return_value = {
            "knowledge": {"price": "42.50"},
            "organic": [],
        }
        result = fetch_current_price("PTT.BK")
        assert result == Decimal("42.50")
        mock_serp.assert_called_once_with("PTT.BK stock price")

    @patch(f"{MODULE_PATH}._serp_request")
    def test_falls_back_to_organic(self, mock_serp: MagicMock) -> None:
        """Uses organic results when knowledge has no price."""
        mock_serp.return_value = {
            "knowledge": {},
            "organic": [
                {"description": "Stock price is USD 35.00 per share"},
            ],
        }
        result = fetch_current_price("AOT.BK")
        assert result == Decimal("35.00")

    @patch(f"{MODULE_PATH}._serp_request")
    def test_returns_none_when_no_data(self, mock_serp: MagicMock) -> None:
        """Returns None when no price in knowledge or organic."""
        mock_serp.return_value = {"knowledge": {}, "organic": []}
        result = fetch_current_price("INVALID")
        assert result is None

    @patch(f"{MODULE_PATH}._serp_request")
    def test_returns_none_when_serp_fails(
        self,
        mock_serp: MagicMock,
    ) -> None:
        """Returns None when SERP API returns None."""
        mock_serp.return_value = None
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
