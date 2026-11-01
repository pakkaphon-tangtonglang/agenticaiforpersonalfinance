"""Daily closing-price series for trend charts (yfinance, free, no key).

Used by the dashboard insight cards. Thai stocks (`PTT.BK`) and ETFs
(`TDEX.BK`) come back in THB; SET holidays are simply missing days.
Series are cached for an hour because Yahoo rate-limits Render's shared
egress IPs; failures return [] and are never cached.
"""

import time
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel

from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

SERIES_TTL_SECONDS = 3600.0
PRICE_PLACES = Decimal("0.0001")

_series_cache: dict[tuple[str, int], tuple[float, list["PricePoint"]]] = {}


class PricePoint(BaseModel):
    """One day's closing price (or fund NAV).

    Example:
        >>> PricePoint(day=date(2026, 10, 30), value=Decimal("42.00")).value
        Decimal('42.00')
    """

    day: date
    value: Decimal


def clear_series_cache() -> None:
    """Forget cached series (used by tests)."""
    _series_cache.clear()


def fetch_price_series(symbol: str, days: int = 30) -> list[PricePoint]:
    """Daily closes for the last `days` calendar days, oldest first.

    Args:
        symbol: Ticker symbol, e.g. "PTT.BK".
        days: Calendar days to look back (at least 1).

    Returns:
        Points with a positive close, or [] when yfinance has nothing.

    Raises:
        ValueError: If days is less than 1.

    Example:
        >>> fetch_price_series("PTT.BK", days=30)  # doctest: +SKIP
        [PricePoint(day=datetime.date(2026, 10, 1), value=Decimal('41.7500')), ...]
    """
    if days < 1:
        raise ValueError(f"days must be at least 1. Received: {days}")
    key = (symbol, days)
    cached = _series_cache.get(key)
    if cached and time.monotonic() - cached[0] < SERIES_TTL_SECONDS:
        return list(cached[1])
    series = _download_closes(symbol, days)
    if series:
        _series_cache[key] = (time.monotonic(), series)
    return list(series)


def _download_closes(symbol: str, days: int) -> list[PricePoint]:
    """Ask yfinance for daily history; [] on any failure."""
    start = (date.today() - timedelta(days=days)).isoformat()
    try:
        import yfinance as yf  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        history = yf.Ticker(symbol).history(start=start)
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        logger.warning("yfinance series fetch failed for %s: %s", symbol, exc)
        return []
    return _closes_to_points(history)


def _closes_to_points(history: Any) -> list[PricePoint]:
    """Turn a yfinance history frame into positive Decimal points."""
    if history is None or history.empty or "Close" not in history:
        return []
    series: list[PricePoint] = []
    for timestamp, close in history["Close"].dropna().items():
        try:
            value = Decimal(str(close)).quantize(PRICE_PLACES)
        except InvalidOperation:
            continue
        if value > 0:
            series.append(PricePoint(day=timestamp.date(), value=value))
    return series
