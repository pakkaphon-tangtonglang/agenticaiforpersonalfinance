"""Price client for fetching stock and fund prices from yfinance.

Supports Thai stocks (.BK suffix) and international tickers.
All prices are returned as Decimal for financial precision.
"""

from decimal import Decimal, InvalidOperation

from finance_ai.core.logging import get_logger

logger = get_logger(__name__)


def _import_yfinance() -> object:
    """Lazy-import yfinance to keep it an optional dependency.

    Returns:
        The yfinance module.

    Raises:
        ImportError: If yfinance is not installed.
    """
    try:
        import yfinance  # noqa: PLC0415

        return yfinance
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for price fetching. " "Install it with: pip install yfinance"
        ) from exc


def fetch_current_price(symbol: str) -> Decimal | None:
    """Fetch the current market price for a single symbol.

    Uses yfinance to retrieve the latest closing/regular market price.
    Supports Thai stocks (e.g., PTT.BK) and international tickers (e.g., AAPL).

    Args:
        symbol: Ticker symbol (e.g., "PTT.BK", "AAPL").

    Returns:
        Current price as Decimal, or None if unavailable.

    Example:
        >>> price = fetch_current_price("PTT.BK")
    """
    yf = _import_yfinance()
    try:
        ticker = yf.Ticker(symbol)  # type: ignore[attr-defined]
        info = ticker.info or {}
        price_value = info.get("currentPrice") or info.get("regularMarketPrice")
        if price_value is None:
            logger.warning("No price data for symbol: %s", symbol)
            return None
        return Decimal(str(price_value))
    except (ValueError, InvalidOperation, KeyError) as exc:
        logger.warning("Failed to fetch price for %s: %s", symbol, exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error fetching price for %s: %s", symbol, exc)
        return None


def fetch_multiple_prices(
    symbols: list[str],
) -> dict[str, Decimal | None]:
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
    """Check whether a symbol exists in yfinance.

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
