"""One search box for stocks (Yahoo) and Thai mutual funds (SEC).

Both searches run in parallel. Only when both find nothing does the LLM
suggest fund-name terms (slow, so it is the last resort). The response says
when it is unsure (search_clarifier) so the page can ask the user.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from finance_ai.tools.fund_query_expander import expand_fund_query
from finance_ai.tools.fund_search_service import FundSearchResult, search_funds
from finance_ai.tools.market_data_models import AssetSymbolMatch
from finance_ai.tools.search_clarifier import FUND, build_clarification, stock_category


def search_all_assets(
    query: str,
    stock_search: Callable[[str], list[AssetSymbolMatch]],
    expand: Callable[[str], list[str]] = expand_fund_query,
) -> dict[str, Any]:
    """Stocks and funds for a query, plus clarifying questions when unsure.

    Args:
        query: What the user typed.
        stock_search: Yahoo symbol search (injected for tests).
        expand: LLM fund-term expansion, used only when nothing else matched.

    Returns:
        {"status", "results", "expanded_terms", "clarify"}; each result has
        symbol/name/exchange/type/category, and funds add NAV, AMC and payout.

    Example:
        >>> response = search_all_assets("ปันผล", search_asset_symbols)  # doctest: +SKIP
        >>> response["results"][0]["category"]  # doctest: +SKIP
        'fund'
    """
    cleaned = query.strip()
    with ThreadPoolExecutor(max_workers=2) as pool:
        stocks_future = pool.submit(stock_search, cleaned)
        funds_future = pool.submit(search_funds, cleaned)
        stocks = [stock_result(match) for match in stocks_future.result()]
        funds = funds_future.result()
    if not stocks and not funds.results:
        funds = search_funds(cleaned, expand=expand)
    results = stocks + [fund_result(fund) for fund in funds.results]
    clarify = build_clarification(results)
    return {
        "status": "ok",
        "results": results,
        "expanded_terms": funds.expanded_terms,
        "clarify": clarify.model_dump() if clarify else None,
    }


def stock_result(match: AssetSymbolMatch) -> dict[str, Any]:
    """Yahoo match → search card."""
    return {
        "symbol": match.symbol,
        "name": match.name,
        "exchange": match.exchange,
        "type": match.quote_type,
        "category": stock_category(match.symbol, match.quote_type),
    }


def fund_result(fund: FundSearchResult) -> dict[str, Any]:
    """SEC fund class → search card (type FUND, with NAV, fund house and payout)."""
    return {
        "symbol": fund.symbol,
        "name": fund.name_th or fund.name_en or fund.symbol,
        "name_en": fund.name_en,
        "exchange": "SEC",
        "type": "FUND",
        "category": FUND,
        "fund_code": fund.fund_code,
        "amc": fund.amc,
        "payout": fund.payout,
        "nav": format(fund.nav.normalize(), "f"),
        "nav_date": fund.nav_date.isoformat(),
    }
