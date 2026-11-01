"""Dashboard insight-card data: spending over time and holdings' price trends.

Spending comes from the user's own expense summaries; prices come from
yfinance (Thai stocks/ETFs) or the SEC NAV API (Thai mutual funds). Every
external failure degrades to a missing line, never an error.
"""

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from typing import Any

from sqlalchemy.orm import Session

from finance_ai.tools.cross_agent_service import get_expense_summary
from finance_ai.tools.fund_nav_client import fetch_nav_series
from finance_ai.tools.investment_calculator import HoldingSummary
from finance_ai.tools.investment_service import get_portfolio_summary
from finance_ai.tools.price_series_client import PricePoint, fetch_price_series
from finance_ai.tools.recommendation_constants import SPENDING_TREND_HISTORY_MONTHS
from finance_ai.tools.recommendation_service import previous_months
from finance_ai.tools.spending_trend_calculator import average_monthly_total, find_category_spikes
from finance_ai.tools.trend_models import AssetSeries, DashboardTrends, MonthlySpending, SeriesKind

SPENDING_MONTHS = 6
PRICE_DAYS = 30
FETCH_WORKERS = 4
PERCENT_PLACES = Decimal("0.01")
# Yahoo has no history for the SET index (^SET.BK), so the SET50 ETF stands in.
BENCHMARK_SYMBOL = "TDEX.BK"
BENCHMARK_NAME = "SET50 (TDEX)"


def build_dashboard_trends(
    session: Session, user_id: str, year: int, month: int
) -> DashboardTrends:
    """Build the insight-card payload for one month.

    Args:
        session: Database session.
        user_id: UUID of the user.
        year: Year of the "current" month.
        month: Current month (1-12).

    Returns:
        Spending months (oldest first), average, spikes, asset lines and the
        SET50 benchmark (only when the user holds something).

    Example:
        >>> build_dashboard_trends(session, uid, 2026, 10).average_spending  # doctest: +SKIP
        Decimal('11250.00')
    """
    months = [(year, month), *previous_months(year, month, SPENDING_MONTHS - 1)]
    summaries = [get_expense_summary(session, user_id, y, m) for y, m in months]
    assets = build_asset_series(get_portfolio_summary(session, user_id).holdings)
    return DashboardTrends(
        spending=_monthly_spending(months, summaries),
        average_spending=average_monthly_total(summaries[1:]),
        spikes=find_category_spikes(summaries[0], summaries[1 : 1 + SPENDING_TREND_HISTORY_MONTHS]),
        assets=assets,
        benchmark=_benchmark_series() if assets else None,
    )


def build_asset_series(holdings: list[HoldingSummary]) -> list[AssetSeries]:
    """One price/NAV line per distinct holding; holdings without data are dropped.

    Example:
        >>> [s.symbol for s in build_asset_series(holdings)]  # doctest: +SKIP
        ['PTT.BK', 'K-USA-A(A)']
    """
    unique = list({holding.symbol: holding for holding in holdings}.values())
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        fetched = list(pool.map(_fetch_series_for, unique))
    lines = [
        _to_asset_series(holding.symbol, holding.name or holding.symbol, _kind(holding), points)
        for holding, points in zip(unique, fetched)
    ]
    return [line for line in lines if line is not None]


def percent_change(points: list[PricePoint]) -> Decimal | None:
    """Change from the first to the last point in percent (None when undefined).

    Example:
        >>> percent_change([PricePoint(day=d1, value=Decimal("40")),
        ...                 PricePoint(day=d2, value=Decimal("44"))])  # doctest: +SKIP
        Decimal('10.00')
    """
    if len(points) < 2 or points[0].value <= 0:
        return None
    change = (points[-1].value - points[0].value) / points[0].value * 100
    return change.quantize(PERCENT_PLACES)


def _monthly_spending(
    months: list[tuple[int, int]], summaries: list[dict[str, Any]]
) -> list[MonthlySpending]:
    """Pair months with their totals, oldest first."""
    pairs = zip(months, summaries)
    return [
        MonthlySpending(year=y, month=m, total=Decimal(str(s.get("total_amount", "0"))))
        for (y, m), s in reversed(list(pairs))
    ]


def _kind(holding: HoldingSummary) -> SeriesKind:
    """Mutual funds use NAV; everything else is a traded price."""
    return "mutual_fund" if holding.asset_type == "mutual_fund" else "stock"


def _fetch_series_for(holding: HoldingSummary) -> list[PricePoint]:
    """NAV series for funds, yfinance closes for stocks."""
    if _kind(holding) == "mutual_fund":
        return fetch_nav_series(holding.symbol, days=PRICE_DAYS)
    return fetch_price_series(holding.symbol, days=PRICE_DAYS)


def _benchmark_series() -> AssetSeries | None:
    """The SET50 ETF line the holdings are compared with."""
    points = fetch_price_series(BENCHMARK_SYMBOL, days=PRICE_DAYS)
    return _to_asset_series(BENCHMARK_SYMBOL, BENCHMARK_NAME, "benchmark", points)


def _to_asset_series(
    symbol: str, name: str, kind: SeriesKind, points: list[PricePoint]
) -> AssetSeries | None:
    """An AssetSeries, or None when there is too little data for a line."""
    change = percent_change(points)
    if change is None:
        return None
    return AssetSeries(symbol=symbol, name=name, kind=kind, points=points, change_pct=change)
