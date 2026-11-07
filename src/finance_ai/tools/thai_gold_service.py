"""Thai gold bar (96.5%) in the portfolio, priced in baht per baht-weight.

Thai shops quote gold per baht-weight (15.244 g of 96.5% gold), not per
ounce. There is no free API for the Gold Traders Association price, so the
latest price is estimated from world gold (COMEX GC=F, USD per troy ounce)
and the USD→THB rate. It lands within a few hundred baht of the shop price;
a user who knows their real buy price types it.
"""

import re
from decimal import Decimal
from typing import Any, Optional

from finance_ai.tools.price_client import fetch_current_price

GOLD = "gold"
THAI_GOLD_SYMBOL: str = "THGOLD965"
THAI_GOLD_NAME: str = "ทองคำแท่ง 96.5%"
WORLD_GOLD_SYMBOL: str = "GC=F"
USD_THB_SYMBOL: str = "USDTHB=X"
GRAMS_PER_BAHT_WEIGHT = Decimal("15.244")
GRAMS_PER_TROY_OUNCE = Decimal("31.1035")
THAI_GOLD_PURITY = Decimal("0.965")
_MONEY = Decimal("0.01")
_GOLD_QUERY = re.compile(r"ทอง|\bgold\b|\bxau\b", re.IGNORECASE)


def thai_gold_price_per_baht_weight() -> Optional[Decimal]:
    """Estimated baht price of one baht-weight of 96.5% gold, None when unknown.

    Example:
        >>> thai_gold_price_per_baht_weight()  # doctest: +SKIP
        Decimal('61483.75')
    """
    ounce_usd = fetch_current_price(WORLD_GOLD_SYMBOL)
    usd_thb = fetch_current_price(USD_THB_SYMBOL)
    if ounce_usd is None or usd_thb is None:
        return None
    gold_grams = GRAMS_PER_BAHT_WEIGHT * THAI_GOLD_PURITY
    return (ounce_usd * usd_thb * gold_grams / GRAMS_PER_TROY_OUNCE).quantize(_MONEY)


def is_thai_gold_query(query: str) -> bool:
    """Whether a search is about gold (ทอง, gold, XAU) — not 'goldman'.

    Example:
        >>> is_thai_gold_query("ทองคำ"), is_thai_gold_query("goldman")
        (True, False)
    """
    return bool(_GOLD_QUERY.search(query or ""))


def validate_thai_gold_symbol(symbol: str) -> str:
    """The gold bar symbol, normalized.

    Raises:
        ValueError: For any other symbol (futures, ETFs...).

    Example:
        >>> validate_thai_gold_symbol("thgold965")
        'THGOLD965'
    """
    normalized = "".join(symbol.split()).upper()
    if normalized != THAI_GOLD_SYMBOL:
        raise ValueError(f"Gold bars use the symbol {THAI_GOLD_SYMBOL}. Received: '{symbol}'")
    return normalized


def thai_gold_search_result(price: Optional[Decimal]) -> dict[str, Any]:
    """Search card for the gold bar, with the estimated price when known.

    Example:
        >>> thai_gold_search_result(None)["symbol"]
        'THGOLD965'
    """
    return {
        "symbol": THAI_GOLD_SYMBOL,
        "name": THAI_GOLD_NAME,
        "exchange": "ราคาประมาณจากทองคำโลก × USD/THB",
        "type": "GOLD",
        "category": GOLD,
        "price": None if price is None else format(price, "f"),
    }
