"""Tests for world_gold_service (world gold straight from the price API, USD per ounce)."""

from decimal import Decimal
from unittest.mock import patch

import pytest

from finance_ai.tools.world_gold_service import is_world_gold_symbol, world_gold_search_result

_PRICE = "finance_ai.tools.world_gold_service.fetch_current_price"


@pytest.mark.parametrize(("symbol", "expected"), [("GC=F", True), (" gc=f ", True), ("GLD", False)])
def test_is_world_gold_symbol(symbol: str, expected: bool) -> None:
    """GC=F in any case; ETFs are not world gold."""
    assert is_world_gold_symbol(symbol) is expected


def test_card_has_usd_price() -> None:
    """Type WORLD_GOLD in the gold category, price fetched from the API."""
    with patch(_PRICE, return_value=Decimal("4012.5")):
        result = world_gold_search_result()
    assert (result["symbol"], result["type"], result["category"], result["price"]) == (
        "GC=F",
        "WORLD_GOLD",
        "gold",
        "4012.5",
    )


def test_card_without_price() -> None:
    """API down: the card still shows, price None."""
    with patch(_PRICE, return_value=None):
        assert world_gold_search_result()["price"] is None
