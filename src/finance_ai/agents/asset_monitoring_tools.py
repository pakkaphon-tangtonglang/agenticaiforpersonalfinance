"""LangGraph tool wrapper for the user asset watchlist.

Provides a single manage_watchlist tool that lets the LLM
add, remove, or list the user's tracked asset symbols.
"""

from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from finance_ai.agents.session_helper import get_tool_session
from finance_ai.tools.symbol_guard import (
    SymbolResolutionError,
    resolve_and_validate_symbol,
)


@tool
def manage_watchlist(
    action: str,
    symbol: str = "",
    name: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """จัดการรายการสินทรัพย์ที่ต้องการติดตาม (Watchlist).

    ใช้ tool นี้เมื่อผู้ใช้ต้องการ:
    - เพิ่มสินทรัพย์เข้ารายการติดตาม (action="add")
    - ลบสินทรัพย์ออกจากรายการ (action="remove")
    - ดูรายการสินทรัพย์ที่ติดตามอยู่ (action="list")

    Args:
        action: "add", "remove", หรือ "list"
        symbol: สัญลักษณ์สินทรัพย์ เช่น "PTT.BK", "BTC-USD", "GC=F"
                (ต้องระบุสำหรับ action="add" และ "remove")
        name: ชื่อที่แสดง เช่น "PTT", "Bitcoin", "ทองคำ"
              (ใช้กับ action="add" เท่านั้น)
        user_id: UUID ของผู้ใช้ (injected จาก graph state).
        db_session_factory: Session factory (injected จาก graph state).

    Returns:
        Dict with action result.

    Example:
        >>> manage_watchlist(action="add", symbol="PTT.BK", name="PTT")
        >>> manage_watchlist(action="list")
        >>> manage_watchlist(action="remove", symbol="PTT.BK")
    """
    from finance_ai.database.crud.watched_asset_crud import (  # noqa: PLC0415
        WatchedAssetCRUD,
    )

    crud = WatchedAssetCRUD()
    normalized_action = action.strip().lower()

    if normalized_action == "add":
        return _handle_add(crud, db_session_factory, user_id, symbol, name)
    if normalized_action == "remove":
        return _handle_remove(crud, db_session_factory, user_id, symbol)
    if normalized_action == "list":
        return _handle_list(crud, db_session_factory, user_id)

    return {"error": f"Unknown action '{action}'. Use 'add', 'remove', or 'list'."}


def _handle_add(
    crud: Any,
    db_session_factory: Any,
    user_id: str,
    symbol: str,
    name: str,
) -> dict[str, Any]:
    """Add a symbol to the watchlist after validating it via the guard.

    The raw input is resolved to a canonical Yahoo symbol; input that does
    not match a real search candidate is rejected with a Thai message.

    Args:
        crud: WatchedAssetCRUD instance.
        db_session_factory: Session factory callable.
        user_id: UUID of the user.
        symbol: Raw user-provided symbol (e.g., "ptt", "PTT.BK").
        name: Display name.

    Returns:
        Dict with add result, or {"error": ...} when the symbol cannot
        be resolved.
    """
    if not symbol:
        return {"error": "symbol is required for action='add'"}
    try:
        canonical_symbol = resolve_and_validate_symbol(symbol)
    except SymbolResolutionError as exc:
        return {"error": str(exc)}
    with get_tool_session(db_session_factory) as session:
        record = crud.add(session, user_id, canonical_symbol, name)
        session.commit()
        if record is None:
            return {
                "action": "add",
                "symbol": canonical_symbol,
                "status": "already_exists",
            }
        return {
            "action": "add",
            "symbol": record.symbol,
            "name": record.name,
            "status": "added",
        }


def _handle_remove(
    crud: Any,
    db_session_factory: Any,
    user_id: str,
    symbol: str,
) -> dict[str, Any]:
    """Remove a symbol from the watchlist.

    Args:
        crud: WatchedAssetCRUD instance.
        db_session_factory: Session factory callable.
        user_id: UUID of the user.
        symbol: Ticker symbol to remove.

    Returns:
        Dict with remove result.
    """
    if not symbol:
        return {"error": "symbol is required for action='remove'"}
    with get_tool_session(db_session_factory) as session:
        deleted = crud.remove(session, user_id, symbol)
        session.commit()
    status = "removed" if deleted else "not_found"
    return {"action": "remove", "symbol": symbol.upper(), "status": status}


def _handle_list(
    crud: Any,
    db_session_factory: Any,
    user_id: str,
) -> dict[str, Any]:
    """List all watched assets for the user.

    Args:
        crud: WatchedAssetCRUD instance.
        db_session_factory: Session factory callable.
        user_id: UUID of the user.

    Returns:
        Dict with watchlist items.
    """
    with get_tool_session(db_session_factory) as session:
        assets = crud.get_by_user(session, user_id)
        items = [{"symbol": a.symbol, "name": a.name} for a in assets]
    return {"action": "list", "count": len(items), "watchlist": items}
