"""LangGraph tool wrapper for the user asset watchlist.

Provides a single manage_watchlist tool that lets the LLM add, remove or
list tracked symbols, and set or clear each symbol's daily news alert time.
"""

from collections.abc import Callable
from typing import Annotated, Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from finance_ai.agents.clarification import clarification, symbol_problem
from finance_ai.agents.session_helper import get_tool_session
from finance_ai.database.crud.watched_asset_crud import WatchedAssetCRUD
from finance_ai.tools.input_normalizer import ambiguous_hour
from finance_ai.tools.symbol_guard import (
    SymbolResolutionError,
    resolve_and_validate_symbol,
)
from finance_ai.tools.watchlist_alert_service import (
    AlertTimeError,
    clear_watchlist_alert,
    get_watchlist_alert_times,
    set_watchlist_alert,
)

ASK_ALERT_TIME_HINT = (
    "Ask the user what time each day (24-hour HH:MM) they want a news alert"
    " for this asset, then call manage_watchlist(action='set_alert')."
)
NOON = 12
ASK_MORNING_OR_EVENING = "หมายถึงเช้าหรือค่ำ — เลือกเวลาที่ต้องการให้แจ้งข่าวทุกวัน"
ASK_TIME = "อยากให้แจ้งข่าวทุกวันกี่โมง (แบบ 24 ชม. เช่น 08:30 หรือ 20:00)"
ASK_WHICH_ASSET_TO_ADD = "อยากเพิ่มสินทรัพย์ตัวไหนเข้ารายการติดตาม"
ASK_WHICH_WATCHED_ASSET = "หมายถึงสินทรัพย์ตัวไหนในรายการติดตาม"
_ACTIONS = "'add', 'remove', 'list', 'set_alert' or 'clear_alert'"


@tool
def manage_watchlist(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    action: str,
    symbol: str = "",
    name: str = "",
    alert_time: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
    messages: Annotated[list[Any] | None, InjectedState("messages")] = None,
) -> dict[str, Any]:
    """จัดการรายการสินทรัพย์ที่ติดตาม (Watchlist) และเวลาแจ้งเตือนข่าวรายวัน.

    ใช้ tool นี้เมื่อผู้ใช้ต้องการ:
    - เพิ่มสินทรัพย์เข้ารายการติดตาม (action="add") — ใส่ alert_time ถ้าผู้ใช้บอกเวลามาแล้ว
    - ลบสินทรัพย์ออกจากรายการ (action="remove")
    - ดูรายการที่ติดตามพร้อมเวลาแจ้งเตือน (action="list")
    - ตั้ง/เปลี่ยนเวลาแจ้งข่าวรายวันของสินทรัพย์ (action="set_alert")
    - ปิดการแจ้งข่าวรายวัน (action="clear_alert")

    Args:
        action: "add", "remove", "list", "set_alert" หรือ "clear_alert"
        symbol: สัญลักษณ์สินทรัพย์ เช่น "PTT.BK", "BTC-USD", "GC=F"
        name: ชื่อที่แสดง เช่น "PTT", "Bitcoin" (ใช้กับ action="add")
        alert_time: เวลาแจ้งข่าวทุกวันแบบ 24 ชม. "HH:MM" เช่น "08:30", "20:00"
            (2 ทุ่ม = "20:00", 7 โมงเช้า = "07:00")
        user_id: UUID ของผู้ใช้ (injected จาก graph state).
        db_session_factory: Session factory (injected จาก graph state).
        messages: บทสนทนา (injected จาก graph state) ใช้ตรวจว่าผู้ใช้ตอบแค่ "8"
            ซึ่งอาจเป็นเช้าหรือค่ำ — ถ้าใช่ จะถามกลับแทนการตั้งเวลา

    Returns:
        Dict with the action result, or {"error": ...}.

    Example:
        >>> manage_watchlist(action="add", symbol="PTT.BK", name="PTT")
        >>> manage_watchlist(action="set_alert", symbol="PTT.BK", alert_time="08:30")
    """
    hour = ambiguous_hour(_latest_user_text(messages)) if alert_time.strip() else None
    if hour is not None and action.strip().lower() in ("add", "set_alert"):
        return _ask_morning_or_evening(action, hour, db_session_factory, user_id, (symbol, name))
    handlers: dict[str, Callable[[], dict[str, Any]]] = {
        "add": lambda: _handle_add(db_session_factory, user_id, symbol, name, alert_time),
        "remove": lambda: _handle_remove(db_session_factory, user_id, symbol),
        "list": lambda: _handle_list(db_session_factory, user_id),
        "set_alert": lambda: _handle_set_alert(db_session_factory, user_id, symbol, alert_time),
        "clear_alert": lambda: _handle_clear_alert(db_session_factory, user_id, symbol),
    }
    handler = handlers.get(action.strip().lower())
    if handler is None:
        return {"error": f"Unknown action '{action}'. Use {_ACTIONS}."}
    return handler()


def _latest_user_text(messages: list[Any] | None) -> str:
    """Text of the user's most recent chat message ("" when unknown)."""
    for message in reversed(messages or []):
        if isinstance(message, HumanMessage):
            return message.text
    return ""


def _ask_morning_or_evening(
    action: str, hour: int, db_session_factory: Any, user_id: str, asset: tuple[str, str]
) -> dict[str, Any]:
    """The user typed a bare hour: ask which one instead of trusting the model's guess.

    For "add", the asset is still added; only the alert time waits for the answer.
    """
    options = [f"{hhmm} ({thai_clock_label(hhmm)})" for hhmm in both_readings(hour)]
    question = clarification(
        f"'{hour}' could be {options[0]} or {options[1]}.", ASK_MORNING_OR_EVENING, options
    )
    if action.strip().lower() != "add":
        return question
    added = _handle_add(db_session_factory, user_id, asset[0], asset[1], "")
    added.pop("next_step", None)
    return {**added, "alert_error": question["error"], **_question_only(question)}


def _question_only(result: dict[str, Any]) -> dict[str, Any]:
    """The clarification fields of a result, without its error text."""
    keys = ("needs_clarification", "ask_user", "options")
    return {key: result[key] for key in keys if key in result}


def both_readings(hour: int) -> tuple[str, str]:
    """Morning and evening readings of a bare hour: 8 → ("08:00", "20:00"); 12 → noon, midnight.

    Example:
        >>> both_readings(12)
        ('12:00', '00:00')
    """
    if hour == NOON:
        return "12:00", "00:00"
    return f"{hour:02d}:00", f"{hour + NOON:02d}:00"


def thai_clock_label(hhmm: str) -> str:
    """Everyday Thai name of an hour: 08:00 → 8 โมงเช้า, 20:00 → 2 ทุ่ม, 00:00 → เที่ยงคืน.

    Example:
        >>> thai_clock_label("15:00")
        'บ่าย 3'
    """
    hour = int(hhmm[:2])
    if hour == 0:
        return "เที่ยงคืน"
    if hour == NOON:
        return "เที่ยง"
    labels = ((6, "ตี {h}"), (NOON, "{h} โมงเช้า"), (17, "บ่าย {p}"), (19, "{p} โมงเย็น"))
    for limit, template in labels:
        if hour < limit:
            return template.format(h=hour, p=hour - NOON)
    return f"{hour - 18} ทุ่ม"


def _handle_add(
    db_session_factory: Any,
    user_id: str,
    symbol: str,
    name: str,
    alert_time: str,
) -> dict[str, Any]:
    """Add a symbol (validated by the guard), then set its alert if a time was given.

    Returns:
        Add result; without a time it carries next_step asking for one.
        Unknown or ambiguous symbols return a clarification instead.
    """
    if not symbol.strip():
        return clarification("symbol is required for action='add'", ASK_WHICH_ASSET_TO_ADD)
    try:
        canonical_symbol = resolve_and_validate_symbol(symbol)
    except SymbolResolutionError as exc:
        return symbol_problem(exc)
    with get_tool_session(db_session_factory) as session:
        record = WatchedAssetCRUD().add(session, user_id, canonical_symbol, name)
        session.commit()
        added_name = None if record is None else record.name
    result: dict[str, Any] = {
        "action": "add",
        "symbol": canonical_symbol,
        "status": "already_exists" if added_name is None else "added",
    }
    if added_name is not None:
        result["name"] = added_name
    return {**result, **_alert_after_add(db_session_factory, user_id, canonical_symbol, alert_time)}


def _alert_after_add(
    db_session_factory: Any, user_id: str, symbol: str, alert_time: str
) -> dict[str, Any]:
    """Set the alert when a time came with the add; otherwise ask for one."""
    if not alert_time:
        return {"alert_time": None, "next_step": ASK_ALERT_TIME_HINT}
    alert = _handle_set_alert(db_session_factory, user_id, symbol, alert_time)
    return {"alert_time": alert.get("alert_time"), **_only_error(alert)}


def _only_error(result: dict[str, Any]) -> dict[str, Any]:
    """Keep the error (and any question) of a failed result; empty when it succeeded."""
    if "error" not in result:
        return {}
    question: dict[str, Any] = {
        key: result[key] for key in ("needs_clarification", "ask_user") if key in result
    }
    return {"alert_error": result["error"], **question}


def _find_watched_symbol(session: Any, user_id: str, typed: str) -> str | None:
    """Canonical symbol of a watched asset however it was typed ("ptt", "ปตท", "PTT.BK")."""
    crud = WatchedAssetCRUD()
    asset = crud.find_by_user_input(session, user_id, typed)
    if asset is None:
        try:
            asset = crud.find_by_user_input(session, user_id, resolve_and_validate_symbol(typed))
        except SymbolResolutionError:
            return None
    return None if asset is None else asset.symbol


def _not_watched(session: Any, user_id: str, typed: str) -> dict[str, Any]:
    """Ask which watched asset the user meant, listing what they watch."""
    watched = [asset.symbol for asset in WatchedAssetCRUD().get_by_user(session, user_id)]
    problem = f"'{typed}' is not on the watchlist."
    if not watched:
        return {"error": f"{problem} The watchlist is empty."}
    return clarification(problem, ASK_WHICH_WATCHED_ASSET, watched)


def _handle_remove(db_session_factory: Any, user_id: str, symbol: str) -> dict[str, Any]:
    """Remove a watched asset (typed any way) and turn off its alert."""
    if not symbol.strip():
        return {"error": "symbol is required for action='remove'"}
    with get_tool_session(db_session_factory) as session:
        watched_symbol = _find_watched_symbol(session, user_id, symbol)
        if watched_symbol is None:
            question = _not_watched(session, user_id, symbol)
            return {"action": "remove", "symbol": symbol.upper(), "status": "not_found", **question}
        WatchedAssetCRUD().remove(session, user_id, watched_symbol)
        session.commit()
        clear_watchlist_alert(session, user_id, watched_symbol)
    return {"action": "remove", "symbol": watched_symbol, "status": "removed"}


def _handle_list(db_session_factory: Any, user_id: str) -> dict[str, Any]:
    """List watched assets with each one's daily alert time (None = off)."""
    with get_tool_session(db_session_factory) as session:
        assets = WatchedAssetCRUD().get_by_user(session, user_id)
        times = get_watchlist_alert_times(session, user_id)
        items = [
            {"symbol": a.symbol, "name": a.name, "alert_time": times.get(a.symbol)} for a in assets
        ]
    return {"action": "list", "count": len(items), "watchlist": items}


def _handle_set_alert(
    db_session_factory: Any, user_id: str, symbol: str, alert_time: str
) -> dict[str, Any]:
    """Set the daily news time for a watched asset; ask when something is missing."""
    if not alert_time.strip():
        return clarification("alert_time (HH:MM) is required for action='set_alert'", ASK_TIME)
    with get_tool_session(db_session_factory) as session:
        watched_symbol = _find_watched_symbol(session, user_id, symbol) if symbol.strip() else None
        if watched_symbol is None:
            return _not_watched(session, user_id, symbol)
        try:
            saved = set_watchlist_alert(
                session, user_id, watched_symbol, alert_time, db_session_factory
            )
        except AlertTimeError as exc:
            return clarification(str(exc), ASK_TIME)
    return {"action": "set_alert", "symbol": watched_symbol, "alert_time": saved}


def _handle_clear_alert(db_session_factory: Any, user_id: str, symbol: str) -> dict[str, Any]:
    """Turn off the daily news alert for a watched asset (typed any way)."""
    if not symbol.strip():
        return {"error": "symbol is required for action='clear_alert'"}
    with get_tool_session(db_session_factory) as session:
        watched_symbol = _find_watched_symbol(session, user_id, symbol) or symbol.strip().upper()
        cleared = clear_watchlist_alert(session, user_id, watched_symbol)
    status = "cleared" if cleared else "no_alert"
    return {"action": "clear_alert", "symbol": watched_symbol, "status": status}
