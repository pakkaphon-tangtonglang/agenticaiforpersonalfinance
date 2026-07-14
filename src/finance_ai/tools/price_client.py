"""Price client for fetching stock prices via Bright Data SERP API.

Uses Google Search SERP API to retrieve stock price data from Google Finance
knowledge panels. All prices are returned as Decimal for financial precision.
"""

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Optional
from urllib.parse import quote_plus

import httpx

from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

# Timeout for Bright Data API requests (seconds)
_REQUEST_TIMEOUT = 30

# Bright Data SERP API endpoint
_SERP_API_URL = "https://api.brightdata.com/request"


def get_api_token() -> str:
    """Get Bright Data API token from settings.

    Returns:
        API token string.

    Raises:
        ValueError: If token is not configured.
    """
    from finance_ai.core.config import get_settings  # noqa: PLC0415

    settings = get_settings()
    if not settings.bright_data_api_token:
        raise ValueError(
            "BRIGHT_DATA_API_TOKEN is required for market data. " "Set it in your .env file."
        )
    return settings.bright_data_api_token


def get_zone() -> str:
    """Get Bright Data zone name from settings.

    Returns:
        Zone name string.
    """
    from finance_ai.core.config import get_settings  # noqa: PLC0415

    return get_settings().bright_data_zone


def build_headers(token: str) -> dict[str, str]:
    """Build HTTP headers for Bright Data SERP API requests.

    Args:
        token: Bright Data API token.

    Returns:
        Headers dict with authorization.
    """
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def serp_request(query: str) -> Optional[dict[str, Any]]:
    """Send a SERP search request to Bright Data API.

    Args:
        query: Search query string.

    Returns:
        Parsed JSON response, or None on failure.
    """
    try:
        token = get_api_token()
        zone = get_zone()
        headers = build_headers(token)
        search_url = f"https://www.google.com/search" f"?q={quote_plus(query)}&brd_json=1"
        payload = {
            "zone": zone,
            "url": search_url,
            "format": "raw",
        }

        with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
            response = client.post(
                _SERP_API_URL,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
    except ValueError as exc:
        logger.warning("Bright Data config error: %s", exc)
        return None
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "SERP API error for '%s': %s %s",
            query,
            exc.response.status_code,
            exc.response.text,
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected SERP error for '%s': %s", query, exc)
        return None


def _extract_price_from_knowledge(
    knowledge: dict[str, Any],
) -> Optional[Decimal]:
    """Extract stock price from Google knowledge panel data.

    Args:
        knowledge: Knowledge graph data from SERP response.

    Returns:
        Price as Decimal, or None if not found.
    """
    # Try common knowledge panel fields
    for key in ("price", "current_price", "value"):
        raw = knowledge.get(key)
        if raw is not None:
            return parse_price_string(str(raw))

    # Try title/description that may contain price
    title = knowledge.get("title", "")
    if title:
        price = parse_price_string(title)
        if price is not None:
            return price

    return None


def parse_price_string(text: str) -> Optional[Decimal]:
    """Parse a price value from a text string.

    Handles formats like "178.25", "$178.25", "1,234.56", "35.50 THB".

    Args:
        text: Text containing a price value.

    Returns:
        Decimal price, or None if parsing fails.
    """
    # Remove currency symbols and whitespace
    cleaned = re.sub(r"[^\d.,]", "", text.strip())
    # Remove thousands separator commas
    cleaned = cleaned.replace(",", "")
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None


def _extract_price_from_organic(
    organic: list[dict[str, Any]],
) -> Optional[Decimal]:
    """Try to extract price from organic search results snippets.

    Args:
        organic: List of organic search results.

    Returns:
        First price found in snippets, or None.
    """
    price_pattern = re.compile(
        r"(?:USD|THB|\$|฿)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,4})?)",
    )
    for result in organic[:3]:
        snippet = result.get("description", "") or result.get("snippet", "")
        match = price_pattern.search(snippet)
        if match:
            return parse_price_string(match.group(1))
    return None


def fetch_current_price(symbol: str) -> Optional[Decimal]:
    """Fetch the current market price for a single symbol via SERP API.

    Searches Google for stock price using Bright Data SERP API
    and extracts price from the knowledge panel or search results.

    Args:
        symbol: Ticker symbol (e.g., "PTT.BK", "AAPL").

    Returns:
        Current price as Decimal, or None if unavailable.

    Example:
        >>> price = fetch_current_price("PTT.BK")
    """
    query = f"{symbol} stock price"
    data = serp_request(query)
    if data is None:
        return None

    knowledge = data.get("knowledge", {})
    if knowledge:
        price = _extract_price_from_knowledge(knowledge)
        if price is not None:
            return price

    organic = data.get("organic", [])
    if organic:
        price = _extract_price_from_organic(organic)
        if price is not None:
            return price

    logger.warning("No price found in SERP results for: %s", symbol)
    return None


def fetch_multiple_prices(
    symbols: list[str],
) -> dict[str, Optional[Decimal]]:
    """Fetch current prices for multiple symbols.

    Args:
        symbols: List of ticker symbols.

    Returns:
        Dict mapping symbol to price (or None if unavailable).

    Example:
        >>> prices = fetch_multiple_prices(["PTT.BK", "AAPL"])
    """
    return {symbol: fetch_current_price(symbol) for symbol in symbols}


def is_valid_ticker(symbol: str) -> bool:
    """Check whether a symbol returns valid data from SERP API.

    Args:
        symbol: Ticker symbol to validate.

    Returns:
        True if the ticker has price data, False otherwise.

    Example:
        >>> is_valid_ticker("PTT.BK")
        True
    """
    price = fetch_current_price(symbol)
    return price is not None
