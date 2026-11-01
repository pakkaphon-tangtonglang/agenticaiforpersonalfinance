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

# News quality: only recent articles from established publishers reach users
# and the AI. No ranking formula: items keep Google News order.
NEWS_MAX_ARTICLES: int = 5
# Project-defined: market news older than a month rarely reflects the
# current price; undated articles fail this check.
NEWS_MAX_AGE_DAYS: int = 30
# Project-defined allowlist: Thai and international financial news outlets,
# exchanges and regulators. Matched against the publisher site in the feed
# (<source url>), ignoring "www." and subdomains.
NEWS_SOURCE_DOMAINS: frozenset[str] = frozenset(
    {
        # International
        "reuters.com",
        "bloomberg.com",
        "cnbc.com",
        "ft.com",
        "wsj.com",
        "marketwatch.com",
        "finance.yahoo.com",
        "apnews.com",
        "bbc.com",
        "nikkei.com",
        "economist.com",
        "barrons.com",
        "investing.com",
        # Thai
        "bangkokbiznews.com",
        "prachachat.net",
        "thansettakij.com",
        "kaohoon.com",
        "hoonsmart.com",
        "thunhoon.com",
        "infoquest.co.th",
        "bangkokpost.com",
        "nationthailand.com",
        "posttoday.com",
        "thairath.co.th",
        "efinancethai.com",
        "moneyandbanking.co.th",
        # Exchanges and regulators
        "set.or.th",
        "sec.or.th",
        "bot.or.th",
        "rd.go.th",
    }
)
# Publisher names accepted when a feed item has no <source url> (lowercase).
NEWS_SOURCE_NAMES: frozenset[str] = frozenset(
    {
        "reuters",
        "bloomberg",
        "cnbc",
        "bangkok post",
        "bangkokbiznews",
        "prachachat",
        "kaohoon",
        "set",
    }
)

# Dividend yield is returned as a fraction (0.03 = 3%); multiply to get percent
DIVIDEND_YIELD_TO_PERCENT: int = 100
