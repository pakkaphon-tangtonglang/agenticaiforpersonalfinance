"""Shared guard that resolves free-text symbols to canonical Yahoo symbols.

Used by POST /assets/watchlist and the manage_watchlist agent tool so user
input can never invent a ticker: the input must match a real Yahoo Finance
search candidate (case-insensitive, with the ".BK" SET suffix stripped) or
be the only candidate returned.
"""

from typing import Optional

from finance_ai.core.logging import get_logger
from finance_ai.tools.market_data_models import AssetSymbolMatch
from finance_ai.tools.symbol_search_service import search_asset_symbols

logger = get_logger(__name__)

# SET stocks carry a ".BK" suffix on Yahoo Finance; user input usually omits it
_SET_SYMBOL_SUFFIX = ".BK"

_EMPTY_INPUT_MESSAGE = "กรุณาระบุสัญลักษณ์สินทรัพย์ที่ต้องการติดตาม (เช่น PTT.BK, AAPL)"

_NOT_FOUND_TEMPLATE = (
    "ไม่พบสัญลักษณ์ '{query}' ในระบบ " + "กรุณาตรวจสอบตัวย่อหุ้นแล้วลองใหม่อีกครั้ง หรือค้นหาชื่อสินทรัพย์ก่อน"
)

_AMBIGUOUS_TEMPLATE = (
    "สัญลักษณ์ '{query}' ไม่ชัดเจน พบหลายรายการที่คล้ายกัน " + "กรุณาระบุสัญลักษณ์ให้เฉพาะเจาะจงมากขึ้น"
)

_SEARCH_FAILED_TEMPLATE = "ไม่สามารถค้นหาสัญลักษณ์ '{query}' ได้ในขณะนี้ กรุณาลองใหม่ภายหลัง"


class SymbolResolutionError(ValueError):
    """Raised when input cannot be resolved to a canonical Yahoo symbol.

    Subclasses ValueError so callers may catch either type.

    Example:
        >>> try:
        ...     resolve_and_validate_symbol("   ")
        ... except SymbolResolutionError:
        ...     print("rejected")
        rejected
    """


def resolve_and_validate_symbol(symbol: str) -> str:
    """Resolve free-text input to a canonical Yahoo Finance symbol.

    The input is never invented: it must case-insensitively match a real
    search candidate's symbol, match a candidate after stripping the ".BK"
    suffix, or be the single candidate the search returns.

    Args:
        symbol: Raw user input (e.g., "appl", "ptt", "PTT.BK").

    Returns:
        The canonical Yahoo symbol (e.g., "AAPL", "PTT.BK").

    Raises:
        SymbolResolutionError: With an actionable Thai message when the
            input is empty, ambiguous, unresolvable, or the search fails.

    Example:
        >>> resolve_and_validate_symbol("aapl")  # doctest: +SKIP
        'AAPL'
    """
    query = symbol.strip()
    if not query:
        raise SymbolResolutionError(_EMPTY_INPUT_MESSAGE)
    try:
        matches = search_asset_symbols(query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Symbol search failed for %r: %s", query, exc)
        raise SymbolResolutionError(_SEARCH_FAILED_TEMPLATE.format(query=query)) from exc
    return _pick_canonical_symbol(query, matches)


def _pick_canonical_symbol(query: str, matches: list[AssetSymbolMatch]) -> str:
    """Pick the canonical symbol from search candidates.

    Args:
        query: Stripped user input.
        matches: Search candidates returned for the query.

    Returns:
        Canonical Yahoo symbol.

    Raises:
        SymbolResolutionError: When no candidate matches or several
            non-exact candidates compete.
    """
    if not matches:
        raise SymbolResolutionError(_NOT_FOUND_TEMPLATE.format(query=query))
    canonical = _exact_symbol_match(query, matches) or _bk_stripped_match(query, matches)
    if canonical:
        return canonical
    if len(matches) == 1:
        return matches[0].symbol
    raise SymbolResolutionError(_AMBIGUOUS_TEMPLATE.format(query=query))


def _exact_symbol_match(query: str, matches: list[AssetSymbolMatch]) -> Optional[str]:
    """Return the candidate whose symbol equals the query, case-insensitively.

    Args:
        query: Stripped user input.
        matches: Search candidates.

    Returns:
        The exact candidate's symbol, or None when no candidate matches.
    """
    upper_query = query.upper()
    for match in matches:
        if match.symbol.upper() == upper_query:
            return match.symbol
    return None


def _bk_stripped_match(query: str, matches: list[AssetSymbolMatch]) -> Optional[str]:
    """Return the candidate whose symbol minus '.BK' equals the query.

    Args:
        query: Stripped user input (e.g., "ptt").
        matches: Search candidates (e.g., [PTT.BK, PTTGC.BK]).

    Returns:
        The matched candidate's symbol (e.g., "PTT.BK"), or None.
    """
    upper_query = query.upper()
    for match in matches:
        if match.symbol.upper().removesuffix(_SET_SYMBOL_SUFFIX) == upper_query:
            return match.symbol
    return None
