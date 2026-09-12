"""Tests for free-text symbol search service (Yahoo Finance, no API key).

Mocks httpx at the transport boundary (monkeypatches httpx.Client.get) so
no real network calls are made.
"""

from typing import Any, Optional

import httpx

from finance_ai.tools.symbol_search_service import search_asset_symbols

SERVICE_PATH = "finance_ai.tools.symbol_search_service"

VALID_PAYLOAD: dict[str, Any] = {
    "quotes": [
        {
            "symbol": "PTT.BK",
            "shortname": "PTT Public Company Limited",
            "exchDisp": "SET",
            "quoteType": "EQUITY",
        },
        {
            "symbol": "PTTGC.BK",
            "longname": "PTT Global Chemical Public Company Limited",
            "exchange": "SET",
            "quoteType": "EQUITY",
        },
    ]
}


class FakeResponse:
    """Minimal httpx response stand-in for search service tests."""

    def __init__(
        self,
        json_data: Optional[dict[str, Any]] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Store either a JSON payload or an error to raise on access.

        Args:
            json_data: Payload returned by json(); None simulates bad JSON.
            error: Exception raised by raise_for_status()/json().
        """
        self._json_data = json_data
        self._error = error

    def raise_for_status(self) -> None:
        """Raise the stored error to simulate HTTP failures."""
        if self._error is not None:
            raise self._error

    def json(self) -> dict[str, Any]:
        """Return the stored payload or raise to simulate invalid JSON."""
        if self._error is not None:
            raise self._error
        if self._json_data is None:
            raise ValueError("Expecting value: invalid JSON body")
        return self._json_data


def install_fake_get(
    monkeypatch: Any,
    response: FakeResponse,
) -> dict[str, Any]:
    """Monkeypatch httpx.Client.get and capture call arguments.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        response: FakeResponse to return (or raise) for every GET.

    Returns:
        Dict capturing "url" and "params" of the first call (empty if none).
    """
    captured: dict[str, Any] = {}

    def fake_get(self: httpx.Client, url: str, **kwargs: Any) -> FakeResponse:
        captured["url"] = url
        captured["params"] = kwargs.get("params")
        return response

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    return captured


class TestSearchAssetSymbols:
    """Tests for search_asset_symbols."""

    def test_parses_valid_response(self, monkeypatch: Any) -> None:
        """Parses quotes into AssetSymbolMatch with name/exchange fallbacks."""
        install_fake_get(monkeypatch, FakeResponse(json_data=VALID_PAYLOAD))

        matches = search_asset_symbols("PTT")

        assert len(matches) == 2
        assert matches[0].symbol == "PTT.BK"
        assert matches[0].name == "PTT Public Company Limited"
        assert matches[0].exchange == "SET"
        assert matches[0].quote_type == "EQUITY"
        assert matches[1].symbol == "PTTGC.BK"
        assert matches[1].name == "PTT Global Chemical Public Company Limited"
        assert matches[1].exchange == "SET"

    def test_sends_expected_request_params(self, monkeypatch: Any) -> None:
        """GETs the Yahoo search endpoint with q/quotesCount/newsCount."""
        captured = install_fake_get(monkeypatch, FakeResponse(json_data={"quotes": []}))

        search_asset_symbols("Apple")

        assert captured["url"].endswith("/v1/finance/search")
        assert captured["params"]["q"] == "Apple"
        assert captured["params"]["quotesCount"] == 8
        assert captured["params"]["newsCount"] == 0

    def test_http_error_returns_empty(self, monkeypatch: Any) -> None:
        """Returns [] when the HTTP response is an error status."""
        error = httpx.HTTPError("500 Server Error")
        install_fake_get(monkeypatch, FakeResponse(error=error))

        assert search_asset_symbols("PTT") == []

    def test_connection_error_returns_empty(self, monkeypatch: Any) -> None:
        """Returns [] when the network connection fails."""
        error = httpx.ConnectError("connection refused")
        install_fake_get(monkeypatch, FakeResponse(error=error))

        assert search_asset_symbols("PTT") == []

    def test_invalid_json_returns_empty(self, monkeypatch: Any) -> None:
        """Returns [] when the response body is not valid JSON."""
        install_fake_get(monkeypatch, FakeResponse(json_data=None))

        assert search_asset_symbols("PTT") == []

    def test_missing_quotes_key_returns_empty(self, monkeypatch: Any) -> None:
        """Returns [] when the payload has no 'quotes' key."""
        install_fake_get(monkeypatch, FakeResponse(json_data={"news": []}))

        assert search_asset_symbols("PTT") == []

    def test_empty_query_returns_empty_without_http_call(self, monkeypatch: Any) -> None:
        """Blank/whitespace queries return [] without any HTTP request."""
        captured = install_fake_get(monkeypatch, FakeResponse(json_data=VALID_PAYLOAD))

        assert search_asset_symbols("   ") == []
        assert captured == {}

    def test_strips_empty_name_entries(self, monkeypatch: Any) -> None:
        """Quote entries without a resolvable name are dropped."""
        payload: dict[str, Any] = {
            "quotes": [
                {"symbol": "MYSTERY.BK", "quoteType": "EQUITY"},
                VALID_PAYLOAD["quotes"][0],
            ]
        }
        install_fake_get(monkeypatch, FakeResponse(json_data=payload))

        matches = search_asset_symbols("PTT")

        assert len(matches) == 1
        assert matches[0].symbol == "PTT.BK"

    def test_caps_at_eight_results(self, monkeypatch: Any) -> None:
        """Never returns more than 8 candidates."""
        payload = {
            "quotes": [
                {
                    "symbol": f"S{i}.BK",
                    "shortname": f"Stock Number {i}",
                    "quoteType": "EQUITY",
                }
                for i in range(12)
            ]
        }
        install_fake_get(monkeypatch, FakeResponse(json_data=payload))

        matches = search_asset_symbols("stock")

        assert len(matches) == 8

    def test_skips_non_dict_quote_entries(self, monkeypatch: Any) -> None:
        """Non-dict entries inside 'quotes' are ignored safely."""
        payload: dict[str, Any] = {"quotes": ["garbage", VALID_PAYLOAD["quotes"][0]]}
        install_fake_get(monkeypatch, FakeResponse(json_data=payload))

        matches = search_asset_symbols("PTT")

        assert len(matches) == 1
        assert matches[0].symbol == "PTT.BK"
