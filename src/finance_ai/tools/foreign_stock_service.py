"""Foreign stocks (NVDA, AAPL...) in the baht portfolio.

Holdings are stored in THB. A price the user types is in the stock's own
currency (usually USD) and is converted at today's rate; refreshed prices
use the latest rate, so gain/loss includes currency moves, as a Thai
investor actually sees it.
"""

import re
from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel

from finance_ai.tools.investment_constants import STOCK_SYMBOL_SUFFIX
from finance_ai.tools.price_client import fetch_close_on, fetch_current_price, fetch_currency
from finance_ai.tools.recurring_service import bangkok_today

FOREIGN_STOCK: str = "foreign_stock"
BASE_CURRENCY: str = "THB"
DEFAULT_FOREIGN_CURRENCY: str = "USD"
_PRICE_PLACES = Decimal("0.0001")
# Tickers like NVDA, BRK-B, 7203.T, 0700.HK. Crypto (-USD), futures (=F),
# indexes (^) and FX (=X) are not stocks.
FOREIGN_STOCK_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")
NON_STOCK_SUFFIXES: tuple[str, ...] = ("-USD", "-USDT", "=F", "=X")


class ThbPrice(BaseModel):
    """A price converted to baht, with the rate used.

    Example:
        >>> ThbPrice(amount_thb=Decimal("6300"), currency="USD",
        ...     exchange_rate=Decimal("35")).currency
        'USD'
    """

    amount_thb: Decimal
    currency: str
    exchange_rate: Decimal
    rate_date: Optional[date] = None


def is_foreign_stock_symbol(symbol: str) -> bool:
    """Whether a ticker is a stock listed outside Thailand.

    Example:
        >>> is_foreign_stock_symbol("NVDA"), is_foreign_stock_symbol("PTT.BK")
        (True, False)
    """
    upper = symbol.strip().upper()
    if not FOREIGN_STOCK_PATTERN.match(upper) or upper.endswith(STOCK_SYMBOL_SUFFIX):
        return False
    return not upper.endswith(NON_STOCK_SUFFIXES)


def _currency_of(symbol: str) -> str:
    """Trading currency (USD when Yahoo does not say)."""
    return (fetch_currency(symbol) or DEFAULT_FOREIGN_CURRENCY).upper()


def _rate_to_thb(currency: str) -> Optional[Decimal]:
    """THB per unit of the currency (1 for THB), None when unavailable."""
    if currency == BASE_CURRENCY:
        return Decimal("1")
    return _rounded(fetch_current_price(f"{currency}{BASE_CURRENCY}=X"))


def _rounded(rate: Optional[Decimal]) -> Optional[Decimal]:
    """Yahoo rates are floats (33.59000015...); keep 4 decimal places."""
    return None if rate is None else rate.quantize(_PRICE_PLACES)


def price_in_thb(symbol: str, price: Decimal, on_date: Optional[date] = None) -> ThbPrice:
    """Convert a price in the stock's currency to baht at the purchase day's rate.

    A past purchase uses that day's closing rate (the last trading day on or
    before it); today, or no history, uses the live rate. rate_date says which.

    Args:
        symbol: Foreign ticker (e.g. "NVDA").
        price: Price per share in the stock's currency (e.g. 180 USD).
        on_date: Purchase day; None means today.

    Returns:
        Baht price with the currency and rate used.

    Raises:
        ValueError: If no exchange rate is available.

    Example:
        >>> price_in_thb("NVDA", Decimal("180")).amount_thb  # doctest: +SKIP
        Decimal('6300.0000')
    """
    rate = exchange_rate_to_thb(symbol, on_date)
    return ThbPrice(
        amount_thb=(price * rate.exchange_rate).quantize(_PRICE_PLACES),
        currency=rate.currency,
        exchange_rate=rate.exchange_rate,
        rate_date=rate.rate_date,
    )


def exchange_rate_to_thb(symbol: str, on_date: Optional[date] = None) -> ThbPrice:
    """Rate from the stock's trading currency to baht: that day's close for a past
    day, otherwise live (Yahoo, 60 s cache).

    Args:
        symbol: Foreign ticker (e.g. "NVDA").
        on_date: Day the rate is for; None or today means live.

    Returns:
        ThbPrice for one unit of the currency (amount_thb equals the rate).

    Raises:
        ValueError: If no exchange rate is available.

    Example:
        >>> exchange_rate_to_thb("NVDA").exchange_rate  # doctest: +SKIP
        Decimal('33.7612')
    """
    currency = _currency_of(symbol)
    past = _rate_on(currency, on_date)
    if past is not None:
        return past
    rate = _rate_to_thb(currency)
    if rate is None:
        raise ValueError(
            f"No exchange rate for {currency} → THB right now. Please try again later."
        )
    return ThbPrice(
        amount_thb=rate, currency=currency, exchange_rate=rate, rate_date=bangkok_today()
    )


def _rate_on(currency: str, on_date: Optional[date]) -> Optional[ThbPrice]:
    """That day's closing rate for a past day; None for today or when Yahoo has none."""
    if on_date is None or currency == BASE_CURRENCY or on_date >= bangkok_today():
        return None
    rate = _rounded(fetch_close_on(f"{currency}{BASE_CURRENCY}=X", on_date))
    if rate is None:
        return None
    return ThbPrice(amount_thb=rate, currency=currency, exchange_rate=rate, rate_date=on_date)


def latest_price_in_thb(symbol: str) -> Optional[Decimal]:
    """Latest market price in baht, or None when the price or rate is missing.

    Example:
        >>> latest_price_in_thb("NVDA")  # doctest: +SKIP
        Decimal('6800.0000')
    """
    price = fetch_current_price(symbol)
    if price is None:
        return None
    rate = _rate_to_thb(_currency_of(symbol))
    return None if rate is None else (price * rate).quantize(_PRICE_PLACES)
