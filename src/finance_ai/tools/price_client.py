"""Price client for fetching stock prices via yfinance (free, no API key).

Uses yfinance fast_info.last_price for real-time prices. All prices are
returned as Decimal for financial precision; any failure returns None so
callers (scheduler, services) can degrade gracefully.
"""

from decimal import Decimal
from typing import Optional

from finance_ai.core.logging import get_logger

logger = get_logger(__name__)


def fetch_current_price(symbol: str) -> Optional[Decimal]:
    """Fetch the current market price for a symbol via yfinance (free, no key).

    Uses fast_info.last_price. Returns None on any failure.

    Args:
        symbol: Ticker symbol (e.g., "PTT.BK", "AAPL", "BTC-USD").

    Returns:
        Current price as Decimal, or None if unavailable.

    Example:
        >>> price = fetch_current_price("PTT.BK")
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
        symbol: Ticker symbol (e.g., "PTT.BK", "AAPL", "BTC-USD").

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
