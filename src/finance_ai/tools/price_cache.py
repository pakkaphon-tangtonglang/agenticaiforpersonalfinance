"""Short-lived TTL cache for market prices.

Beta testers ask the same price queries back-to-back; serving from cache
avoids hammering Yahoo Finance (the suspected cause of Render's empty
price responses) and keeps answers fast. Thai market prices move within
seconds anyway, so a short TTL loses nothing.
"""

import time
from decimal import Decimal
from typing import Optional


class PriceCache:
    """Time-based cache mapping ticker symbols to Decimal prices."""

    def __init__(self, ttl_seconds: float = 60.0) -> None:
        """Create a cache whose entries expire after ttl_seconds.

        Args:
            ttl_seconds: Seconds until a stored entry is stale.

        Example:
            >>> cache = PriceCache(ttl_seconds=60.0)
        """
        self._ttl_seconds = ttl_seconds
        self._entries: dict[str, tuple[float, Decimal]] = {}

    def get(self, symbol: str) -> Optional[Decimal]:
        """Return the cached price for a symbol, or None when stale.

        Args:
            symbol: Ticker symbol (e.g., "PTT.BK").

        Returns:
            Cached price, or None when absent or expired.
        """
        entry = self._entries.get(symbol)
        if entry is None:
            return None
        stored_at, price = entry
        if time.monotonic() - stored_at >= self._ttl_seconds:
            return None
        return price

    def store(self, symbol: str, price: Decimal) -> None:
        """Store a price for a symbol, overwriting any previous entry.

        Args:
            symbol: Ticker symbol (e.g., "PTT.BK").
            price: Fetched price as Decimal.
        """
        self._entries[symbol] = (time.monotonic(), price)

    def clear(self) -> None:
        """Remove all entries (used by tests and manual invalidation)."""
        self._entries.clear()
