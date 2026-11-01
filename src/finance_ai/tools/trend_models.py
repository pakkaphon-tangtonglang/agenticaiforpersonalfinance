"""Response models for the dashboard insight cards (GET /dashboard/trends)."""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from finance_ai.tools.price_series_client import PricePoint
from finance_ai.tools.spending_trend_calculator import CategorySpike

SeriesKind = Literal["stock", "mutual_fund", "benchmark"]


class MonthlySpending(BaseModel):
    """Total spending of one calendar month.

    Example:
        >>> MonthlySpending(year=2026, month=10, total=Decimal("12500.00")).month
        10
    """

    year: int
    month: int
    total: Decimal


class AssetSeries(BaseModel):
    """Daily price (stock/ETF) or NAV (fund) line for one asset.

    Example:
        >>> AssetSeries(symbol="PTT.BK", name="PTT", kind="stock", points=[],
        ...     change_pct=Decimal("1.25")).kind
        'stock'
    """

    symbol: str
    name: str
    kind: SeriesKind
    points: list[PricePoint]
    change_pct: Decimal


class DashboardTrends(BaseModel):
    """Everything the insight carousel needs; empty lists hide cards.

    Example:
        >>> DashboardTrends(spending=[], average_spending=Decimal("0"), spikes=[],
        ...     assets=[], benchmark=None).assets
        []
    """

    spending: list[MonthlySpending]
    average_spending: Decimal
    spikes: list[CategorySpike]
    assets: list[AssetSeries]
    benchmark: AssetSeries | None
