"""One search box for stocks (Yahoo), Thai mutual funds (SEC) and gold.

Gold searches start with two cards: the Thai 96.5% bar (baht price calculated
from world gold, can be held) and world gold as the API quotes it (GC=F, USD
per ounce — a futures contract, so view/track only).

Both searches run in parallel. Only when both find nothing does the LLM
suggest fund-name terms (slow, so it is the last resort). The response says
when it is unsure (search_clarifier) so the page can ask the user.
"""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from finance_ai.tools.fund_query_expander import expand_fund_query
from finance_ai.tools.fund_search_service import FundSearchResult, search_funds
from finance_ai.tools.market_data_models import AssetSymbolMatch
from finance_ai.tools.search_clarifier import FUND, build_clarification, stock_category
from finance_ai.tools.thai_gold_service import (
    is_thai_gold_query,
    thai_gold_price_per_baht_weight,
    thai_gold_search_result,
)
from finance_ai.tools.world_gold_service import is_world_gold_symbol, world_gold_search_result


def search_all_assets(
    query: str,
    stock_search: Callable[[str], list[AssetSymbolMatch]],
    expand: Callable[[str], list[str]] = expand_fund_query,
    ask_ai: bool = False,
) -> dict[str, Any]:
    """Stocks and funds for a query, plus clarifying questions when unsure.

    Args:
        query: What the user typed.
        stock_search: Yahoo symbol search (injected for tests).
        expand: LLM fund-term expansion, used only when nothing else matched.
        ask_ai: The "ถาม AI" box — always run the LLM expansion for funds.

    Returns:
        {"status", "results", "expanded_terms", "clarify"}; each result has
        symbol/name/exchange/type/category, and funds add NAV, AMC and payout.

    Example:
        >>> response = search_all_assets("ปันผล", search_asset_symbols)  # doctest: +SKIP
        >>> response["results"][0]["category"]  # doctest: +SKIP
        'fund'
    """
    cleaned = query.strip()
    fund_options: dict[str, Any] = {"expand": expand, "always_expand": True} if ask_ai else {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        gold_future = pool.submit(gold_bar_results, cleaned)
        stocks_future = pool.submit(stock_search, cleaned)
        funds_future = pool.submit(search_funds, cleaned, **fund_options)
        stocks = [
            stock_result(m) for m in stocks_future.result() if not _shown_as_gold(m, gold_future)
        ]
        funds = funds_future.result()
        gold = gold_future.result()
    if not ask_ai and not gold and not stocks and not funds.results:
        funds = search_funds(cleaned, expand=expand)
    results = gold + stocks + [fund_result(fund) for fund in funds.results]
    clarify = build_clarification(results)
    return {
        "status": "ok",
        "results": results,
        "expanded_terms": funds.expanded_terms,
        "clarify": clarify.model_dump() if clarify else None,
    }


def gold_bar_results(query: str) -> list[dict[str, Any]]:
    """For gold searches: the calculated Thai bar card, then the API world gold card."""
    if not is_thai_gold_query(query):
        return []
    return [thai_gold_search_result(thai_gold_price_per_baht_weight()), world_gold_search_result()]


def _shown_as_gold(match: AssetSymbolMatch, gold_future: Future[list[dict[str, Any]]]) -> bool:
    """Yahoo's own GC=F row is dropped when the world gold card already shows it."""
    return is_world_gold_symbol(match.symbol) and bool(gold_future.result())


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
