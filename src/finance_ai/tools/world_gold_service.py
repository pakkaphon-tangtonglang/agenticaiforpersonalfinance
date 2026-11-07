"""World gold as the price API quotes it (Yahoo GC=F, USD per troy ounce).

The counterpart of thai_gold_service: that one *calculates* a baht-weight price
for Thai gold bars, which can be held. This one shows the API price as is.
Yahoo has no spot gold quote, only the COMEX futures contract, so this card is
for viewing and tracking — futures cannot be added to the portfolio.
"""

from typing import Any

from finance_ai.tools.price_client import fetch_current_price
from finance_ai.tools.thai_gold_service import GOLD, WORLD_GOLD_SYMBOL

__all__ = [
    "WORLD_GOLD_NAME",
    "WORLD_GOLD_SYMBOL",
    "is_world_gold_symbol",
    "world_gold_search_result",
]

WORLD_GOLD_NAME: str = "ทองคำโลก (สัญญาล่วงหน้า COMEX)"


def is_world_gold_symbol(symbol: str) -> bool:
    """Whether a symbol is the API's world gold price.

    Example:
        >>> is_world_gold_symbol("gc=f"), is_world_gold_symbol("GLD")
        (True, False)
    """
    return "".join(symbol.split()).upper() == WORLD_GOLD_SYMBOL


def world_gold_search_result() -> dict[str, Any]:
    """Search card for world gold with the API's latest USD-per-ounce price.

    Example:
        >>> world_gold_search_result()["symbol"]  # doctest: +SKIP
        'GC=F'
    """
    price = fetch_current_price(WORLD_GOLD_SYMBOL)
    return {
        "symbol": WORLD_GOLD_SYMBOL,
        "name": WORLD_GOLD_NAME,
        "exchange": "ราคาจาก API (Yahoo Finance)",
        "type": "WORLD_GOLD",
        "category": GOLD,
        "price": None if price is None else format(price, "f"),
    }
