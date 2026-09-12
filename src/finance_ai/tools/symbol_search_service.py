"""Free-text asset symbol search via Yahoo Finance (free, no API key).

Powers the resolve_asset_symbol tool: the agent can resolve user phrases
like "PTT", "Apple", or "gold" into concrete tradable symbols (PTT.BK,
AAPL, GC=F) before calling price/news tools.
"""

from typing import Any, Optional

import httpx

from finance_ai.core.logging import get_logger
from finance_ai.tools.market_data_models import AssetSymbolMatch

logger = get_logger(__name__)

# Yahoo Finance public search endpoint (no API key required)
_YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"

# Timeout for search requests (seconds)
_REQUEST_TIMEOUT = 15

# Maximum number of candidates returned
_MAX_RESULTS = 8

# Yahoo rejects bare HTTP clients without a User-Agent
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def search_asset_symbols(query: str) -> list[AssetSymbolMatch]:
    """Search Yahoo Finance (free, no API key) for asset symbols matching free text.

    Works with English names, tickers, and transliterations (e.g. "PTT",
    "Apple", "gold"). Thai names should be translated by the LLM first.
    Never raises — returns [] on any failure (LLM-facing tool boundary).

    Args:
        query: Free-text asset name or ticker (e.g. "PTT", "Apple").

    Returns:
        Up to 8 AssetSymbolMatch candidates, [] on any failure.

    Example:
        >>> matches = search_asset_symbols("PTT")
        >>> matches[0].symbol
        'PTT.BK'
    """
    stripped_query = query.strip()
    if not stripped_query:
        return []
    try:
        payload = _fetch_search_payload(stripped_query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Symbol search failed for %r: %s", stripped_query, exc)
        return []
    return _parse_quotes(payload)


def _fetch_search_payload(query: str) -> dict[str, Any]:
    """Fetch the raw search payload from Yahoo Finance.

    Args:
        query: Free-text search query.

    Returns:
        Parsed JSON payload dict.

    Raises:
        httpx.HTTPError: On network or HTTP failure.
        ValueError: If the response body is not valid JSON.
    """
    params: dict[str, Any] = {
        "q": query,
        "quotesCount": _MAX_RESULTS,
        "newsCount": 0,
    }
    with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
        response = client.get(_YAHOO_SEARCH_URL, params=params, headers=_HEADERS)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return payload


def _parse_quotes(payload: dict[str, Any]) -> list[AssetSymbolMatch]:
    """Parse a Yahoo search payload into asset symbol matches.

    Args:
        payload: Raw JSON payload from the Yahoo search endpoint.

    Returns:
        Up to _MAX_RESULTS matches with non-empty names.
    """
    if not isinstance(payload, dict):
        return []
    quotes = payload.get("quotes")
    if not isinstance(quotes, list):
        return []
    matches = (_quote_to_match(quote) for quote in quotes if isinstance(quote, dict))
    return [match for match in matches if match is not None][:_MAX_RESULTS]


def _quote_to_match(quote: dict[str, Any]) -> Optional[AssetSymbolMatch]:
    """Convert a single Yahoo quote entry to a match.

    Args:
        quote: One entry from the payload's "quotes" list.

    Returns:
        AssetSymbolMatch, or None when no name can be resolved.
    """
    name = str(quote.get("shortname") or quote.get("longname") or "").strip()
    if not name:
        return None
    return AssetSymbolMatch(
        symbol=str(quote.get("symbol") or ""),
        name=name,
        exchange=str(quote.get("exchDisp") or quote.get("exchange") or ""),
        quote_type=str(quote.get("quoteType") or ""),
    )
