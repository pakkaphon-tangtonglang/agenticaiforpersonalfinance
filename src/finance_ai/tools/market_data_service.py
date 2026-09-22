"""Pure functions for market data retrieval: dashboard, forex, news.

No database dependency. Uses free, no-API-key sources only: yfinance for
stock dashboards and FX rates, Google News RSS for financial news.
"""

from datetime import timezone
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from typing import Any, Optional
from xml.etree import ElementTree

import httpx

from finance_ai.core.logging import get_logger
from finance_ai.tools.market_data_constants import (
    CURRENCY_CODE_LENGTH,
    NEWS_NOT_FOUND_MESSAGE,
)
from finance_ai.tools.market_data_models import (
    CurrencyConversionResult,
    FinanceNewsResult,
    NewsItem,
    StockDashboardResult,
)
from finance_ai.tools.price_client import (
    fetch_current_price,
    fetch_currency,
)

logger = get_logger(__name__)

# Google News RSS search endpoint (free, no API key)
_GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"

# Timeout for news RSS requests (seconds)
_REQUEST_TIMEOUT = 30


def _to_decimal(value: Any) -> Optional[Decimal]:
    """Safely convert a value to Decimal, returning None on failure.

    Args:
        value: Any value from API response.

    Returns:
        Decimal representation, or None if conversion fails.

    Example:
        >>> _to_decimal(35.5)
        Decimal('35.5')
        >>> _to_decimal(None) is None
        True
    """
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def fetch_stock_dashboard(symbol: str) -> StockDashboardResult:
    """Fetch comprehensive stock/asset overview via yfinance (free, no key).

    Args:
        symbol: Ticker symbol (e.g., "PTT.BK", "AAPL", "BTC-USD").

    Returns:
        StockDashboardResult with all available fields populated.

    Example:
        >>> result = fetch_stock_dashboard("PTT.BK")
        >>> result.name
        'PTT Public Company Limited'
    """
    return _fetch_dashboard_from_yfinance(symbol)


def _yfinance_dividend_yield(info: dict[str, Any]) -> Decimal:
    """Convert yfinance dividendYield (fraction) to percentage Decimal.

    yfinance returns 0.035 for 3.5% yield; we store as 3.50.

    Args:
        info: yfinance ticker info dict.

    Returns:
        Dividend yield as percentage (e.g., Decimal('3.50')), or Decimal('0').
    """
    raw = info.get("dividendYield")
    if raw is None:
        return Decimal("0")
    converted = _to_decimal(raw)
    if converted is None:
        return Decimal("0")
    return (converted * 100).quantize(Decimal("0.01"))


def _fetch_dashboard_from_yfinance(symbol: str) -> StockDashboardResult:
    """Fetch stock dashboard from yfinance.

    The full info block (quoteSummary endpoint) is often blocked on
    datacenter IPs (Render); in that case still return the current price
    via the hardened price client (quote → history fallback → cache).

    Args:
        symbol: Ticker symbol (e.g., "PTT.BK", "AAPL").

    Returns:
        StockDashboardResult populated from yfinance, or empty on failure.
    """
    try:
        import yfinance as yf  # noqa: PLC0415

        info = yf.Ticker(symbol).info or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance fetch failed for %s: %s", symbol, exc)
        return StockDashboardResult(current_price=fetch_current_price(symbol))
    if not info:
        return StockDashboardResult(current_price=fetch_current_price(symbol))
    return StockDashboardResult(
        name=info.get("longName") or info.get("shortName"),
        current_price=_price_with_fallback(info, symbol),
        currency=info.get("currency"),
        fifty_two_week_high=_to_decimal(info.get("fiftyTwoWeekHigh")),
        fifty_two_week_low=_to_decimal(info.get("fiftyTwoWeekLow")),
        pe_ratio=_to_decimal(info.get("trailingPE")),
        market_cap=_to_decimal(info.get("marketCap")),
        dividend_yield_percent=_yfinance_dividend_yield(info),
        analyst_target_price=_to_decimal(info.get("targetMeanPrice")),
        recommendation=info.get("recommendationKey"),
    )


def _price_with_fallback(info: dict[str, Any], symbol: str) -> Optional[Decimal]:
    """Current price from info, falling back to the price client.

    Args:
        info: yfinance ticker info dict.
        symbol: Ticker symbol (used by the fallback fetch).

    Returns:
        Price as Decimal, or None when no source has a price.
    """
    price = _to_decimal(
        info.get("currentPrice") or info.get("regularMarketPrice") or info.get("price")
    )
    if price is not None:
        return price
    return fetch_current_price(symbol)


def format_price_with_unit(price: Decimal, symbol: str) -> str:
    """Format a price with its currency unit (หน่วย) for display.

    Args:
        price: Price as Decimal.
        symbol: Ticker symbol, used to resolve the unit.

    Returns:
        Formatted price, e.g. "42.00 บาท" for SET symbols,
        "190.50 USD" for US symbols, or "190.50" when the currency
        is unknown.

    Example:
        >>> format_price_with_unit(Decimal("42.00"), "PTT.BK")
        '42.00 บาท'
    """
    unit = _price_unit(symbol)
    if unit is None:
        return f"{price:,.2f}"
    return f"{price:,.2f} {unit}"


def _price_unit(symbol: str) -> Optional[str]:
    """Resolve the display unit for a symbol's price.

    SET symbols (".BK" suffix) always trade in THB — returned as the
    Thai "บาท" without a network call. Other symbols resolve via the
    yfinance currency code (e.g., "USD"), or None when unavailable.

    Args:
        symbol: Ticker symbol.

    Returns:
        Display unit string, or None when unknown.
    """
    if symbol.upper().endswith(".BK"):
        return "บาท"
    return fetch_currency(symbol)


def _validate_currency_code(code: str, label: str) -> str:
    """Validate and normalize a currency code.

    Args:
        code: Currency code to validate.
        label: Label for error messages (e.g., "from_currency").

    Returns:
        Uppercased, stripped currency code.

    Raises:
        ValueError: If code is not a 3-character alphabetic string.
    """
    normalized = code.strip().upper()
    if len(normalized) != CURRENCY_CODE_LENGTH or not normalized.isalpha():
        raise ValueError(
            f"Invalid {label}: '{code}'. "
            f"Must be a {CURRENCY_CODE_LENGTH}-character currency code "
            f"(e.g., 'USD', 'THB')."
        )
    return normalized


def convert_currency(
    from_currency: str,
    to_currency: str,
    amount: Decimal,
) -> CurrencyConversionResult:
    """Convert currency using exchange rates from yfinance (free, no key).

    Args:
        from_currency: Source currency code (e.g., "USD").
        to_currency: Target currency code (e.g., "THB").
        amount: Amount to convert.

    Returns:
        CurrencyConversionResult with rate and converted amount.

    Raises:
        ValueError: If currency codes are invalid or rate unavailable.

    Example:
        >>> result = convert_currency("USD", "THB", Decimal("100"))
        >>> result.converted_amount
        Decimal('3450.00')
    """
    norm_from = _validate_currency_code(from_currency, "from_currency")
    norm_to = _validate_currency_code(to_currency, "to_currency")
    rate = _fetch_exchange_rate(norm_from, norm_to)
    converted = amount * rate

    return CurrencyConversionResult(
        from_currency=norm_from,
        to_currency=norm_to,
        amount=amount,
        exchange_rate=rate,
        converted_amount=converted,
    )


def _fetch_exchange_rate(from_currency: str, to_currency: str) -> Decimal:
    """Fetch exchange rate via the hardened price client (e.g., "USDTHB=X").

    Reuses the retry + history-fallback + cache logic, so FX survives
    transient Yahoo rate limiting on Render.

    Args:
        from_currency: Normalized source currency code.
        to_currency: Normalized target currency code.

    Returns:
        Exchange rate as Decimal.

    Raises:
        ValueError: If exchange rate is unavailable.
    """
    fx_symbol = f"{from_currency}{to_currency}=X"
    rate = fetch_current_price(fx_symbol)
    if rate is None:
        raise ValueError(
            f"ไม่พบอัตราแลกเปลี่ยนสำหรับ {from_currency} → {to_currency} "
            "(โปรดลองใหม่อีกครั้งในภายหลัง)"
        )
    return rate


def fetch_finance_news(symbol: str) -> FinanceNewsResult:
    """Fetch latest finance news for a symbol via Google News RSS (free).

    Args:
        symbol: Ticker symbol (e.g., "AAPL", "PTT.BK", "BTC-USD").

    Returns:
        FinanceNewsResult with news content and availability flag.
        Never raises — failures degrade to the Thai not-found message.

    Example:
        >>> result = fetch_finance_news("AAPL")
        >>> result.has_news
        True
    """
    not_found_message = NEWS_NOT_FOUND_MESSAGE.format(symbol=symbol)
    try:
        xml_text = _fetch_news_rss(symbol)
        items = _parse_rss_items(xml_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("News fetch failed for %s: %s", symbol, exc)
        return FinanceNewsResult(symbol=symbol, news_content=not_found_message, has_news=False)
    content = _format_news_items(items, symbol)
    if not content:
        return FinanceNewsResult(symbol=symbol, news_content=not_found_message, has_news=False)
    return FinanceNewsResult(symbol=symbol, news_content=content, has_news=True)


def _fetch_news_rss(symbol: str) -> str:
    """Fetch the raw Google News RSS feed for a symbol.

    Args:
        symbol: Ticker symbol to search news for.

    Returns:
        Raw RSS/XML response body.

    Raises:
        httpx.HTTPError: On network or HTTP failure.
    """
    params = {"q": f"{symbol} stock", "hl": "th", "gl": "TH", "ceid": "TH:th"}
    with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
        response = client.get(_GOOGLE_NEWS_RSS_URL, params=params)
        response.raise_for_status()
        return response.text


def _parse_rss_items(xml_text: str) -> list[dict[str, str]]:
    """Parse Google News RSS XML into a list of item dicts.

    Args:
        xml_text: Raw RSS/XML response body.

    Returns:
        List of dicts with title, source, link, and pubDate keys.

    Raises:
        ElementTree.ParseError: If the body is not valid XML.
    """
    # RSS comes from Google News over HTTPS; stdlib expat is acceptable here
    # (defusedxml would require a new dependency) — see bandit B314.
    root = ElementTree.fromstring(xml_text)  # nosec B314
    return [
        {
            "title": (item.findtext("title") or "").strip(),
            "source": (item.findtext("source") or "").strip(),
            "link": (item.findtext("link") or "").strip(),
            "pubDate": (item.findtext("pubDate") or "").strip(),
        }
        for item in root.findall(".//item")
    ]


def _format_news_items(items: list[dict[str, str]], symbol: str) -> str:
    """Format parsed RSS items into a Thai news digest.

    Args:
        items: Parsed RSS item dicts (title/source/link/pubDate).
        symbol: The ticker symbol queried.

    Returns:
        Formatted news string, or "" when there are no items.
    """
    articles = []
    for item in items[:5]:
        parts = [f"**{item.get('title', '')}**"]
        if item.get("source"):
            parts.append(f"แหล่งที่มา: {item['source']}")
        if item.get("link"):
            parts.append(item["link"])
        if item.get("pubDate"):
            parts.append(item["pubDate"])
        articles.append("\n".join(parts))
    if not articles:
        return ""
    return f"ข่าวล่าสุดสำหรับ {symbol}:\n\n" + "\n\n---\n\n".join(articles)


def fetch_news_items(symbol: str) -> list[NewsItem]:
    """Fetch parsed news items for a symbol via Google News RSS (free).

    Structured counterpart of fetch_finance_news: returns validated article
    models instead of a markdown string. Never raises — failures degrade
    to an empty list.

    Args:
        symbol: Ticker symbol (e.g., "AAPL", "PTT.BK").

    Returns:
        List of NewsItem (at most 5), [] on any failure.

    Example:
        >>> items = fetch_news_items("AAPL")
        >>> len(items) <= 5
        True
    """
    try:
        xml_text = _fetch_news_rss(symbol)
        raw_items = _parse_rss_items(xml_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("News items fetch failed for %s: %s", symbol, exc)
        return []
    return [_raw_item_to_news_item(item) for item in raw_items[:5]]


def _raw_item_to_news_item(raw_item: dict[str, str]) -> NewsItem:
    """Convert a parsed RSS item dict into a NewsItem model.

    Args:
        raw_item: Dict with title, source, link, and pubDate keys.

    Returns:
        NewsItem with an ISO 8601 published_at (None when unparsable).
    """
    return NewsItem(
        title=raw_item.get("title", ""),
        source=raw_item.get("source", ""),
        link=raw_item.get("link") or None,
        published_at=_parse_pub_date(raw_item.get("pubDate", "")),
    )


def _parse_pub_date(raw_date: str) -> Optional[str]:
    """Convert an RFC 2822 pubDate into an ISO 8601 timestamp string.

    Args:
        raw_date: Raw pubDate string from the RSS feed.

    Returns:
        ISO 8601 string (e.g., "2026-09-12T09:00:00+00:00"), or None when
        the date is empty or unparsable.

    Example:
        >>> _parse_pub_date("Mon, 12 Sep 2026 09:00:00 GMT")
        '2026-09-12T09:00:00+00:00'
    """
    if not raw_date:
        return None
    try:
        parsed_date = parsedate_to_datetime(raw_date)
    except (TypeError, ValueError):
        return None
    if parsed_date.tzinfo is None:
        parsed_date = parsed_date.replace(tzinfo=timezone.utc)
    return parsed_date.isoformat()
