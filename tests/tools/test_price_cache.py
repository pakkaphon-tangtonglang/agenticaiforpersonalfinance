"""Tests for the short-lived TTL price cache."""

from decimal import Decimal

from finance_ai.tools.price_cache import PriceCache


class TestPriceCache:
    """Tests for PriceCache get/store/clear."""

    def test_missing_symbol_returns_none(self) -> None:
        """A symbol never stored returns None."""
        cache = PriceCache(ttl_seconds=60.0)

        assert cache.get("PTT.BK") is None

    def test_stored_price_is_returned(self) -> None:
        """A freshly stored price is returned unchanged."""
        cache = PriceCache(ttl_seconds=60.0)
        cache.store("PTT.BK", Decimal("35.50"))

        assert cache.get("PTT.BK") == Decimal("35.50")

    def test_entry_expires_after_ttl(self) -> None:
        """A zero TTL expires entries immediately."""
        cache = PriceCache(ttl_seconds=0.0)
        cache.store("PTT.BK", Decimal("35.50"))

        assert cache.get("PTT.BK") is None

    def test_store_overwrites_previous_price(self) -> None:
        """Storing again replaces the earlier price."""
        cache = PriceCache(ttl_seconds=60.0)
        cache.store("PTT.BK", Decimal("35.50"))
        cache.store("PTT.BK", Decimal("42.00"))

        assert cache.get("PTT.BK") == Decimal("42.00")

    def test_clear_removes_all_entries(self) -> None:
        """clear() empties the cache."""
        cache = PriceCache(ttl_seconds=60.0)
        cache.store("PTT.BK", Decimal("35.50"))
        cache.store("KBANK.BK", Decimal("155.00"))
        cache.clear()

        assert cache.get("PTT.BK") is None
        assert cache.get("KBANK.BK") is None

    def test_symbols_are_isolated(self) -> None:
        """One symbol's entry does not affect another's."""
        cache = PriceCache(ttl_seconds=60.0)
        cache.store("PTT.BK", Decimal("35.50"))

        assert cache.get("KBANK.BK") is None
        assert cache.get("PTT.BK") == Decimal("35.50")
