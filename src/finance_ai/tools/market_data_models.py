"""Pydantic models for market data tool outputs.

Provides type-safe, validated structures for stock dashboard data,
currency conversion results, finance news content, and symbol search
candidates.
"""

from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field


class StockDashboardResult(BaseModel):
    """Comprehensive stock/asset overview from yfinance (free, no key).

    Attributes:
        name: Full name of the asset (e.g., "PTT Public Company Limited").
        current_price: Current or last market price.
        currency: Trading currency (e.g., "THB", "USD").
        fifty_two_week_high: 52-week high price.
        fifty_two_week_low: 52-week low price.
        pe_ratio: Trailing P/E ratio.
        market_cap: Market capitalization.
        dividend_yield_percent: Annual dividend yield as percentage.
        analyst_target_price: Mean analyst target price.
        recommendation: Analyst consensus (e.g., "buy", "hold", "sell").

    Example:
        >>> result = StockDashboardResult(
        ...     name="PTT", current_price=Decimal("35.50"),
        ...     currency="THB", dividend_yield_percent=Decimal("3.5")
        ... )
    """

    name: Optional[str] = None
    current_price: Optional[Decimal] = None
    currency: Optional[str] = None
    fifty_two_week_high: Optional[Decimal] = None
    fifty_two_week_low: Optional[Decimal] = None
    pe_ratio: Optional[Decimal] = None
    market_cap: Optional[Decimal] = None
    dividend_yield_percent: Decimal = Field(default=Decimal("0"))
    analyst_target_price: Optional[Decimal] = None
    recommendation: Optional[str] = None


class CurrencyConversionResult(BaseModel):
    """Result of a real-time currency conversion via yfinance FX rates.

    Attributes:
        from_currency: Source currency code (e.g., "USD").
        to_currency: Target currency code (e.g., "THB").
        amount: Original amount to convert.
        exchange_rate: Real-time exchange rate used.
        converted_amount: Resulting amount in target currency.

    Example:
        >>> result = CurrencyConversionResult(
        ...     from_currency="USD", to_currency="THB",
        ...     amount=Decimal("100"), exchange_rate=Decimal("34.50"),
        ...     converted_amount=Decimal("3450.00")
        ... )
    """

    from_currency: str
    to_currency: str
    amount: Decimal
    exchange_rate: Decimal
    converted_amount: Decimal


class FinanceNewsResult(BaseModel):
    """Finance news lookup result for a given asset symbol.

    Attributes:
        symbol: The ticker symbol queried.
        news_content: Raw news text from Google News RSS.
        has_news: Whether any news articles were found.

    Example:
        >>> result = FinanceNewsResult(
        ...     symbol="PTT.BK", news_content="...", has_news=True
        ... )
    """

    symbol: str
    news_content: str
    has_news: bool


class AssetSymbolMatch(BaseModel):
    """A single candidate asset match from a free-text symbol search.

    Attributes:
        symbol: Ticker symbol (e.g., "PTT.BK").
        name: Human-readable asset name (e.g., "PTT Public Company Limited").
        exchange: Exchange display name (e.g., "SET" / "NASDAQ").
        quote_type: Asset type (e.g., "EQUITY" / "CRYPTOCURRENCY" / "COMMODITY").

    Example:
        >>> match = AssetSymbolMatch(
        ...     symbol="PTT.BK",
        ...     name="PTT Public Company Limited",
        ...     exchange="SET", quote_type="EQUITY",
        ... )
    """

    symbol: str  # e.g. "PTT.BK"
    name: str  # e.g. "PTT Public Company Limited"
    exchange: str  # e.g. "SET" / "NASDAQ"
    quote_type: str  # e.g. "EQUITY" / "CRYPTOCURRENCY" / "COMMODITY"


class NewsItem(BaseModel):
    """A single parsed news article for the asset page.

    Attributes:
        title: Article headline.
        source: Publisher name (e.g., "Reuters").
        link: Article URL, or None when unavailable.
        published_at: ISO 8601 timestamp string, or None when unparsable.

    Example:
        >>> item = NewsItem(title="Oil rises", source="Reuters")
    """

    title: str
    source: str = ""
    link: Optional[str] = None
    published_at: Optional[str] = None


class AssetFetchResult(BaseModel):
    """Structured result of an immediate asset data fetch.

    Attributes:
        symbol: Canonical (uppercased) ticker symbol.
        price: Formatted price string with 2 decimals, or None when unavailable.
        currency: Uppercased currency code (e.g., "USD"), or None when unavailable.
        news: Parsed news articles for the symbol.
        error: Combined Thai error message, or None when the requested
            data was fetched successfully.

    Example:
        >>> result = AssetFetchResult(symbol="AAPL", price="254.30", currency="USD")
    """

    symbol: str
    price: Optional[str] = None
    currency: Optional[str] = None
    news: list[NewsItem] = Field(default_factory=list)
    error: Optional[str] = None
