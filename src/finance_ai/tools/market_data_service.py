"""Pure functions for market data retrieval: dashboard, forex, news.

No database dependency. Uses Bright Data SERP API (Google Search) for
stock data, exchange rates, and financial news.
"""

import json
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from finance_ai.core.logging import get_logger
from finance_ai.tools.market_data_constants import (
    CURRENCY_CODE_LENGTH,
    NEWS_NOT_FOUND_MESSAGE,
)
from finance_ai.tools.market_data_models import (
    CurrencyConversionResult,
    FinanceNewsResult,
    StockDashboardResult,
)
from finance_ai.tools.price_client import (
    _build_headers,
    _get_api_token,
    _get_zone,
    _parse_price_string,
    _serp_request,
)

logger = get_logger(__name__)


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


def _extract_price(info: dict[str, Any]) -> Optional[Decimal]:
    """Extract current price from knowledge panel data.

    Args:
        info: Knowledge graph data from SERP response.

    Returns:
        Current price as Decimal, or None if unavailable.

    Example:
        >>> _extract_price({"price": "35.50"})
        Decimal('35.50')
    """
    for key in ("price", "current_price", "value"):
        raw = info.get(key)
        if raw is not None:
            return _parse_price_string(str(raw))
    return None


def _calculate_dividend_yield(info: dict[str, Any]) -> Decimal:
    """Extract dividend yield from knowledge panel as a percentage.

    Args:
        info: Knowledge graph data from SERP response.

    Returns:
        Dividend yield as percentage (e.g., 3.50 for 3.5%).

    Example:
        >>> _calculate_dividend_yield({"dividend_yield": "3.50%"})
        Decimal('3.50')
    """
    raw_yield = info.get("dividend_yield") or info.get("dividendYield")
    if raw_yield is None:
        return Decimal("0")
    cleaned = str(raw_yield).replace("%", "").strip()
    result = _to_decimal(cleaned)
    return result if result is not None else Decimal("0")


def fetch_stock_dashboard(symbol: str) -> StockDashboardResult:
    """Fetch comprehensive stock/asset overview via yfinance (with SERP fallback).

    Tries yfinance first for real-time data, falls back to SERP API.

    Args:
        symbol: Ticker symbol (e.g., "PTT.BK", "AAPL", "BTC-USD").

    Returns:
        StockDashboardResult with all available fields populated.

    Example:
        >>> result = fetch_stock_dashboard("PTT.BK")
        >>> result.name
        'PTT Public Company Limited'
    """
    result = _fetch_dashboard_from_yfinance(symbol)
    if result.current_price is not None:
        return result
    return _fetch_dashboard_from_serp(symbol)


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

    Args:
        symbol: Ticker symbol (e.g., "PTT.BK", "AAPL").

    Returns:
        StockDashboardResult populated from yfinance, or empty on failure.
    """
    try:
        import yfinance as yf  # noqa: PLC0415

        ticker = yf.Ticker(symbol)
        info = ticker.info
        if not info:
            return StockDashboardResult()

        price = _to_decimal(
            info.get("currentPrice") or info.get("regularMarketPrice") or info.get("price")
        )
        return StockDashboardResult(
            name=info.get("longName") or info.get("shortName"),
            current_price=price,
            currency=info.get("currency"),
            fifty_two_week_high=_to_decimal(info.get("fiftyTwoWeekHigh")),
            fifty_two_week_low=_to_decimal(info.get("fiftyTwoWeekLow")),
            pe_ratio=_to_decimal(info.get("trailingPE")),
            market_cap=_to_decimal(info.get("marketCap")),
            dividend_yield_percent=_yfinance_dividend_yield(info),
            analyst_target_price=_to_decimal(info.get("targetMeanPrice")),
            recommendation=info.get("recommendationKey"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance fetch failed for %s: %s", symbol, exc)
        return StockDashboardResult()


def _fetch_dashboard_from_serp(symbol: str) -> StockDashboardResult:
    """Fetch stock dashboard from SERP API (Google knowledge panel).

    Args:
        symbol: Ticker symbol.

    Returns:
        StockDashboardResult populated from SERP, or empty on failure.
    """
    query = f"{symbol} stock"
    data = _serp_request(query)
    if data is None:
        return StockDashboardResult()

    knowledge = data.get("knowledge", {})
    if not knowledge:
        return StockDashboardResult()

    return _build_dashboard_from_knowledge(knowledge)


def _build_dashboard_from_knowledge(
    info: dict[str, Any],
) -> StockDashboardResult:
    """Build StockDashboardResult from SERP knowledge panel.

    Args:
        info: Knowledge graph data from Google SERP response.

    Returns:
        Populated StockDashboardResult.
    """
    return StockDashboardResult(
        name=info.get("title") or info.get("name"),
        current_price=_extract_price(info),
        currency=info.get("currency"),
        fifty_two_week_high=_parse_price_string(
            str(info.get("52_week_high", "")),
        ),
        fifty_two_week_low=_parse_price_string(
            str(info.get("52_week_low", "")),
        ),
        pe_ratio=_to_decimal(info.get("pe_ratio") or info.get("trailingPE")),
        market_cap=_parse_price_string(
            str(info.get("market_cap", "")),
        ),
        dividend_yield_percent=_calculate_dividend_yield(info),
        analyst_target_price=_parse_price_string(
            str(info.get("target_price", "")),
        ),
        recommendation=info.get("recommendation"),
    )


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
    """Convert currency using exchange rates from SERP API.

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
    """Fetch exchange rate via SERP API (Google Search).

    Args:
        from_currency: Normalized source currency code.
        to_currency: Normalized target currency code.

    Returns:
        Exchange rate as Decimal.

    Raises:
        ValueError: If exchange rate is unavailable.
    """
    query = f"1 {from_currency} to {to_currency}"
    data = _serp_request(query)
    if data is None:
        raise ValueError(f"ไม่สามารถดึงอัตราแลกเปลี่ยน {from_currency}/{to_currency}")

    knowledge = data.get("knowledge", {})
    # Google typically shows "1 USD = 34.50 THB" in knowledge panel
    for key in ("price", "value", "result", "conversion"):
        raw = knowledge.get(key)
        if raw is not None:
            rate = _parse_price_string(str(raw))
            if rate is not None:
                return rate

    # Try parsing from title (e.g., "34.50 Thai Baht")
    title = knowledge.get("title", "")
    if title:
        rate = _parse_price_string(title)
        if rate is not None:
            return rate

    raise ValueError(f"ไม่พบอัตราแลกเปลี่ยนสำหรับ {from_currency} → {to_currency}")


def _format_news_from_organic(
    organic: list[dict[str, Any]],
    symbol: str,
) -> str:
    """Format news content from organic search results.

    Args:
        organic: List of organic search results from SERP.
        symbol: The ticker symbol queried.

    Returns:
        Formatted news string with titles, sources, and snippets.
    """
    articles = []
    for item in organic[:5]:
        title = item.get("title", "")
        source = item.get("source", "") or item.get("displayed_link", "")
        snippet = item.get("description", "") or item.get("snippet", "")
        link = item.get("link", "")

        parts = []
        if title:
            parts.append(f"**{title}**")
        if source:
            parts.append(f"แหล่งที่มา: {source}")
        if snippet:
            parts.append(snippet)
        if link:
            parts.append(f"ลิงก์: {link}")

        if parts:
            articles.append("\n".join(parts))

    if not articles:
        return ""
    return f"ข่าวล่าสุดสำหรับ {symbol}:\n\n" + "\n\n---\n\n".join(articles)


def fetch_finance_news(symbol: str) -> FinanceNewsResult:
    """Fetch latest finance news for a given asset symbol via SERP API.

    Searches Google News for the symbol and formats results.

    Args:
        symbol: Ticker symbol (e.g., "AAPL", "PTT.BK", "BTC-USD").

    Returns:
        FinanceNewsResult with news content and availability flag.

    Example:
        >>> result = fetch_finance_news("AAPL")
        >>> result.has_news
        True
    """
    query = f"{symbol} stock news"
    try:
        data = _serp_request(query)
    except ValueError as exc:
        logger.warning("Bright Data config error for news: %s", exc)
        return FinanceNewsResult(
            symbol=symbol,
            news_content=str(exc),
            has_news=False,
        )

    if data is None:
        return FinanceNewsResult(
            symbol=symbol,
            news_content=f"เกิดข้อผิดพลาดในการดึงข่าว {symbol}",
            has_news=False,
        )

    organic = data.get("organic", [])
    if not organic:
        message = NEWS_NOT_FOUND_MESSAGE.format(symbol=symbol)
        return FinanceNewsResult(
            symbol=symbol,
            news_content=message,
            has_news=False,
        )

    content = _format_news_from_organic(organic, symbol)
    if not content:
        message = NEWS_NOT_FOUND_MESSAGE.format(symbol=symbol)
        return FinanceNewsResult(
            symbol=symbol,
            news_content=message,
            has_news=False,
        )

    return FinanceNewsResult(symbol=symbol, news_content=content, has_news=True)
