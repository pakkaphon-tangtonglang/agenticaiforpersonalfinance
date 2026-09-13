"""Price client for fetching stock prices via yfinance (free, no API key).

Hardened for datacenter hosting (Render), where Yahoo Finance often
rate-limits the quote endpoint on shared egress IPs:

1. fast_info quote fetch first,
2. chart-history last close as fallback (different Yahoo endpoint),
3. one retry round with backoff,
4. a short TTL cache so repeat queries never refetch.

All prices are returned as Decimal for financial precision; any failure
returns None so callers (scheduler, services) can degrade gracefully.
"""

import time
from decimal import Decimal
from typing import Optional

from finance_ai.core.logging import get_logger
from finance_ai.tools.price_cache import PriceCache

logger = get_logger(__name__)

PRICE_RETRY_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 1.5
PRICE_CACHE_TTL_SECONDS = 60.0

_PRICE_CACHE = PriceCache(ttl_seconds=PRICE_CACHE_TTL_SECONDS)


def _retry_sleep() -> None:
    """Wait briefly between price fetch retry rounds."""
    time.sleep(RETRY_DELAY_SECONDS)


def fetch_current_price(symbol: str) -> Optional[Decimal]:
    """Fetch the current market price for a symbol via yfinance (free, no key).

    Serves from the TTL cache when fresh, otherwise fetches with the
    history fallback and one retry round. Returns None on any failure.

    Args:
        symbol: Ticker symbol (e.g., "PTT.BK", "AAPL", "BTC-USD").

    Returns:
        Current price as Decimal, or None if unavailable.

    Example:
        >>> price = fetch_current_price("PTT.BK")
    """
    cached_price = _PRICE_CACHE.get(symbol)
    if cached_price is not None:
        return cached_price
    price = _fetch_price_with_retry(symbol)
    if price is not None:
        _PRICE_CACHE.store(symbol, price)
    return price


def _fetch_price_with_retry(symbol: str) -> Optional[Decimal]:
    """Fetch a price, retrying up to PRICE_RETRY_ATTEMPTS rounds.

    Args:
        symbol: Ticker symbol to fetch.

    Returns:
        Price as Decimal, or None when every round fails.
    """
    for attempt in range(PRICE_RETRY_ATTEMPTS):
        price = _fetch_price_any_source(symbol)
        if price is not None:
            return price
        if attempt < PRICE_RETRY_ATTEMPTS - 1:
            _retry_sleep()
    return None


def _fetch_price_any_source(symbol: str) -> Optional[Decimal]:
    """Try the quote endpoint, then the chart-history fallback.

    Args:
        symbol: Ticker symbol to fetch.

    Returns:
        Price as Decimal, or None when both sources fail.
    """
    quote_price = _fetch_price_fast_info(symbol)
    if quote_price is not None:
        return quote_price
    return _fetch_price_history_close(symbol)


def _fetch_price_fast_info(symbol: str) -> Optional[Decimal]:
    """Fetch last_price via the yfinance quote endpoint; None on failure.

    Args:
        symbol: Ticker symbol to fetch.

    Returns:
        Price as Decimal, or None.
    """
    try:
        import yfinance as yf  # noqa: PLC0415

        price = yf.Ticker(symbol).fast_info.last_price
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance price fetch failed for %s: %s", symbol, exc)
        return None
    if price is None:
        logger.warning("No price returned by yfinance for: %s", symbol)
        return None
    return Decimal(str(price))


def _fetch_price_history_close(symbol: str) -> Optional[Decimal]:
    """Fetch the last close via the yfinance chart endpoint; None on failure.

    The chart endpoint often still works when the quote endpoint is
    rate-limited (observed on Render's shared egress IPs).

    Args:
        symbol: Ticker symbol to fetch.

    Returns:
        Last close as Decimal, or None.
    """
    try:
        import yfinance as yf  # noqa: PLC0415

        history = yf.Ticker(symbol).history(period="5d")
        if history is None or history.empty or "Close" not in history:
            return None
        closes = history["Close"].dropna()
        if closes.empty:
            return None
        return Decimal(str(closes.iloc[-1]))
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance history fetch failed for %s: %s", symbol, exc)
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
    """Check whether a symbol returns valid price data from yfinance.

    Args:
        symbol: Ticker symbol to validate.

    Returns:
        True if the ticker has price data, False otherwise.

    Example:
        >>> is_valid_ticker("PTT.BK")
        True
    """
    return fetch_current_price(symbol) is not None


def fetch_currency(symbol: str) -> Optional[str]:
    """Fetch the trading currency code for a symbol via yfinance (free).

    Args:
        symbol: Ticker symbol (e.g., "PTT.BK", "AAPL").

    Returns:
        Uppercased currency code (e.g., "THB"), or None when yfinance
        does not report a currency or the lookup fails.

    Example:
        >>> fetch_currency("PTT.BK")
        'THB'
    """
    try:
        import yfinance as yf  # noqa: PLC0415

        info = yf.Ticker(symbol).info or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance currency fetch failed for %s: %s", symbol, exc)
        return None
    currency = info.get("currency")
    if not currency:
        logger.warning("No currency reported by yfinance for: %s", symbol)
        return None
    return str(currency).upper()
