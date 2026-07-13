"""Pydantic models for market data tool outputs.

Provides type-safe, validated structures for stock dashboard data,
currency conversion results, and finance news content.
"""

from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field


class StockDashboardResult(BaseModel):
    """Comprehensive stock/asset overview from Bright Data API.

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
    """Result of a real-time currency conversion via Bright Data API.

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
        news_content: Raw news text from Bright Data API.
        has_news: Whether any news articles were found.

    Example:
        >>> result = FinanceNewsResult(
        ...     symbol="PTT.BK", news_content="...", has_news=True
        ... )
    """

    symbol: str
    news_content: str
    has_news: bool
