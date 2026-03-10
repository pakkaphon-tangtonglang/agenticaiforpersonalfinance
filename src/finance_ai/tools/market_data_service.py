"""Pure functions for market data retrieval: dashboard, forex, news.

No database dependency. Uses yfinance for stock/forex data and
langchain_community for Yahoo Finance news (optional dependency).
"""

from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from finance_ai.core.logging import get_logger
from finance_ai.tools.market_data_constants import (
    CURRENCY_CODE_LENGTH,
    DIVIDEND_YIELD_TO_PERCENT,
    FOREX_SYMBOL_TEMPLATE,
    NEWS_NOT_FOUND_MESSAGE,
    NEWS_USER_AGENT,
)
from finance_ai.tools.market_data_models import (
    CurrencyConversionResult,
    FinanceNewsResult,
    StockDashboardResult,
)
from finance_ai.tools.price_client import _import_yfinance

logger = get_logger(__name__)


def _to_decimal(value: Any) -> Optional[Decimal]:
    """Safely convert a value to Decimal, returning None on failure.

    Args:
        value: Any value from yfinance info dict.

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
    """Extract current price from yfinance info dict with fallback.

    Args:
        info: Ticker info dictionary from yfinance.

    Returns:
        Current price as Decimal, or None if unavailable.

    Example:
        >>> _extract_price({"currentPrice": 35.5})
        Decimal('35.5')
    """
    raw = info.get("currentPrice") or info.get("regularMarketPrice")
    return _to_decimal(raw)


def _calculate_dividend_yield(info: dict[str, Any]) -> Decimal:
    """Calculate dividend yield as a percentage from yfinance data.

    Args:
        info: Ticker info dictionary from yfinance.

    Returns:
        Dividend yield as percentage (e.g., 3.50 for 3.5%).

    Example:
        >>> _calculate_dividend_yield({"dividendYield": 0.035})
        Decimal('3.5')
    """
    raw_yield = info.get("dividendYield")
    if raw_yield is None:
        return Decimal("0")
    decimal_value = _to_decimal(raw_yield)
    if decimal_value is None:
        return Decimal("0")
    return decimal_value * DIVIDEND_YIELD_TO_PERCENT


def fetch_stock_dashboard(symbol: str) -> StockDashboardResult:
    """Fetch comprehensive stock/asset overview from yfinance.

    Retrieves name, price, P/E ratio, market cap, 52-week range,
    dividend yield, analyst target, and recommendation for a given symbol.

    Args:
        symbol: Ticker symbol (e.g., "PTT.BK", "AAPL", "BTC-USD").

    Returns:
        StockDashboardResult with all available fields populated.

    Raises:
        No exceptions raised; returns empty model on failure.

    Example:
        >>> result = fetch_stock_dashboard("PTT.BK")
        >>> result.name
        'PTT Public Company Limited'
    """
    yf = _import_yfinance()
    try:
        ticker = yf.Ticker(symbol)  # type: ignore[attr-defined]
        info: dict[str, Any] = ticker.info or {}
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fetch dashboard for %s: %s", symbol, exc)
        return StockDashboardResult()

    return _build_dashboard_from_info(info)


def _build_dashboard_from_info(
    info: dict[str, Any],
) -> StockDashboardResult:
    """Build StockDashboardResult from yfinance info dictionary.

    Args:
        info: Raw info dict from yfinance Ticker.

    Returns:
        Populated StockDashboardResult.
    """
    return StockDashboardResult(
        name=info.get("longName"),
        current_price=_extract_price(info),
        currency=info.get("currency"),
        fifty_two_week_high=_to_decimal(info.get("fiftyTwoWeekHigh")),
        fifty_two_week_low=_to_decimal(info.get("fiftyTwoWeekLow")),
        pe_ratio=_to_decimal(info.get("trailingPE")),
        market_cap=_to_decimal(info.get("marketCap")),
        dividend_yield_percent=_calculate_dividend_yield(info),
        analyst_target_price=_to_decimal(info.get("targetMeanPrice")),
        recommendation=info.get("recommendationKey"),
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
    """Convert currency using real-time exchange rates from yfinance.

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
    """Fetch exchange rate from yfinance.

    Args:
        from_currency: Normalized source currency code.
        to_currency: Normalized target currency code.

    Returns:
        Exchange rate as Decimal.

    Raises:
        ValueError: If exchange rate is unavailable.
    """
    yf = _import_yfinance()
    symbol = FOREX_SYMBOL_TEMPLATE.format(from_currency=from_currency, to_currency=to_currency)
    try:
        ticker = yf.Ticker(symbol)  # type: ignore[attr-defined]
        info: dict[str, Any] = ticker.info or {}
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"ไม่สามารถดึงอัตราแลกเปลี่ยน {symbol}: {exc}") from exc

    raw_rate = info.get("regularMarketPrice") or info.get("previousClose")
    rate = _to_decimal(raw_rate)
    if rate is None:
        raise ValueError(f"ไม่พบอัตราแลกเปลี่ยนสำหรับ {from_currency} → {to_currency}")
    return rate


def _import_yahoo_news_tool() -> Any:
    """Lazy-import YahooFinanceNewsTool from langchain_community.

    Returns:
        The YahooFinanceNewsTool class.

    Raises:
        ImportError: If langchain_community is not installed.
    """
    try:
        from langchain_community.tools.yahoo_finance_news import (  # noqa: PLC0415
            YahooFinanceNewsTool,
        )

        return YahooFinanceNewsTool
    except ImportError as exc:
        raise ImportError(
            "langchain-community is required for finance news. "
            "Install it with: pip install langchain-community"
        ) from exc


def fetch_finance_news(symbol: str) -> FinanceNewsResult:
    """Fetch latest finance news for a given asset symbol.

    Uses Yahoo Finance via langchain_community. Returns a graceful
    result if the dependency is missing or no news is found.

    Args:
        symbol: Ticker symbol (e.g., "AAPL", "PTT.BK", "BTC-USD").

    Returns:
        FinanceNewsResult with news content and availability flag.

    Example:
        >>> result = fetch_finance_news("AAPL")
        >>> result.has_news
        True
    """
    import os  # noqa: PLC0415

    os.environ["USER_AGENT"] = NEWS_USER_AGENT

    try:
        news_tool_class = _import_yahoo_news_tool()
        return _run_news_tool(news_tool_class, symbol)
    except ImportError as exc:
        logger.warning("News tool unavailable: %s", exc)
        return FinanceNewsResult(symbol=symbol, news_content=str(exc), has_news=False)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fetch news for %s: %s", symbol, exc)
        return FinanceNewsResult(
            symbol=symbol,
            news_content=f"เกิดข้อผิดพลาดในการดึงข่าว: {exc}",
            has_news=False,
        )


def _run_news_tool(news_tool_class: type, symbol: str) -> FinanceNewsResult:
    """Execute the Yahoo Finance news tool and parse results.

    Args:
        news_tool_class: The YahooFinanceNewsTool class.
        symbol: Ticker symbol to look up.

    Returns:
        FinanceNewsResult with parsed content.
    """
    tool_instance = news_tool_class()
    content = tool_instance.run(symbol)

    if not content or "No news found" in content:
        message = NEWS_NOT_FOUND_MESSAGE.format(symbol=symbol)
        return FinanceNewsResult(symbol=symbol, news_content=message, has_news=False)

    return FinanceNewsResult(symbol=symbol, news_content=content, has_news=True)
