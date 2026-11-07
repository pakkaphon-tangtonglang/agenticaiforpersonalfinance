"""Crypto (Yahoo pairs like BTC-USD) in the baht portfolio.

Prices are in USD, so they are converted to baht the same way as foreign
stocks (foreign_stock_service.price_in_thb / latest_price_in_thb).
"""

import re

CRYPTO = "crypto"
CRYPTO_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{2,15}-USDT?$")
CRYPTO_SYMBOL_EXAMPLES = "BTC-USD, ETH-USD"


def is_crypto_symbol(symbol: str) -> bool:
    """Whether a symbol is a Yahoo crypto pair priced in USD.

    Example:
        >>> is_crypto_symbol("BTC-USD"), is_crypto_symbol("AAPL")
        (True, False)
    """
    return bool(CRYPTO_SYMBOL_PATTERN.match(symbol.strip().upper()))


def validate_crypto_symbol(symbol: str) -> str:
    """Normalize a crypto symbol.

    Raises:
        ValueError: When it is not a coin-USD pair.

    Example:
        >>> validate_crypto_symbol("eth-usd")
        'ETH-USD'
    """
    if not is_crypto_symbol(symbol):
        raise ValueError(f"'{symbol}' is not a crypto symbol. Examples: {CRYPTO_SYMBOL_EXAMPLES}")
    return symbol.strip().upper()
