"""Constants for market data tools: stock dashboard, currency conversion, news.

Used by market_data_service.py and market_data_tools.py for consistent
configuration across all market data operations.
"""

# Currency codes commonly used by Thai investors
SUPPORTED_CURRENCIES: tuple[str, ...] = (
    "THB",
    "USD",
    "EUR",
    "JPY",
    "GBP",
    "CNY",
    "SGD",
    "HKD",
    "AUD",
    "KRW",
)

# Default base currency for Thai users
DEFAULT_CURRENCY: str = "THB"

# Minimum currency code length
CURRENCY_CODE_LENGTH: int = 3

# News tool: message when no results found
NEWS_NOT_FOUND_MESSAGE: str = "ไม่พบข่าวสารล่าสุดสำหรับ {symbol} ในขณะนี้"

# Dividend yield is returned as a fraction (0.03 = 3%); multiply to get percent
DIVIDEND_YIELD_TO_PERCENT: int = 100
