"""Streamlit view for fetching asset news and prices on demand.

Provides a simple UI with two dropdowns:
1. Choose fetch type (news or price)
2. Choose asset symbol (from presets + user-added symbols)
Results are summarized by LLM in the same format as the chat agent.
"""

from typing import Any, Callable

import streamlit as st
from langchain_core.language_models import BaseChatModel
from sqlalchemy.orm import Session

from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

# Fetch type options
_FETCH_TYPES: list[dict[str, str]] = [
    {"label": "📰 หาข้อมูลข่าวสารเกี่ยวกับสินทรัพย์", "value": "news"},
    {"label": "💰 ราคาสินทรัพย์", "value": "price"},
]

# Default asset presets
_DEFAULT_ASSETS: list[str] = [
    "GC=F (ทองคำ)",
    "BTC-USD (Bitcoin)",
    "CL=F (น้ำมัน)",
    "^GSPC (S&P 500)",
    "PTT.BK (หุ้น PTT)",
    "AMZN (Amazon)",
]

_SESSION_KEY = "custom_asset_list"


def _get_asset_list() -> list[str]:
    """Get combined default + user-added asset list.

    Returns:
        List of asset label strings.
    """
    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = []
    custom: list[str] = st.session_state[_SESSION_KEY]
    return _DEFAULT_ASSETS + custom


def _extract_symbol(label: str) -> str:
    """Extract ticker symbol from a label like 'GC=F (ทองคำ)'.

    Args:
        label: Asset label string.

    Returns:
        Cleaned ticker symbol.

    Example:
        >>> _extract_symbol("GC=F (ทองคำ)")
        'GC=F'
    """
    return label.split("(")[0].strip().upper()


def render_schedule_view(
    user_id: str,
    db_session_factory: Callable[[], Session],
    chat_model: BaseChatModel | None = None,
) -> None:
    """Render the asset news/price lookup view.

    Args:
        user_id: UUID of the current user.
        db_session_factory: Callable that creates DB sessions.
        chat_model: Optional LLM for news summarization.
    """
    st.markdown("### 🔔 ค้นหาข่าว / ราคาสินทรัพย์")

    # ── Top row: fetch form (left) + add asset (right) ──
    col_fetch, col_add = st.columns(2, gap="large")

    with col_fetch:
        st.markdown("#### 🔍 ค้นหาข้อมูล")
        _render_fetch_form(user_id, db_session_factory, chat_model)

    with col_add:
        st.markdown("#### ➕ เพิ่มสินทรัพย์ใหม่")
        _render_add_asset_form()

    # ── Full-width results below ──
    st.divider()
    _render_results(user_id, db_session_factory)


def _render_fetch_form(
    user_id: str,
    db_session_factory: Callable[[], Session],
    chat_model: BaseChatModel | None = None,
) -> None:
    """Render the two-dropdown fetch form.

    Args:
        user_id: UUID of the current user.
        db_session_factory: Callable that creates DB sessions.
        chat_model: Optional LLM for news summarization.
    """
    fetch_type = st.selectbox(
        "ต้องการดูอะไร",
        options=range(len(_FETCH_TYPES)),
        format_func=lambda i: _FETCH_TYPES[i]["label"],
        key="fetch_type_select",
    )

    asset_list = _get_asset_list()
    asset_choice = st.selectbox(
        "เลือกสินทรัพย์",
        options=asset_list,
        key="asset_select",
    )

    if st.button(
        "🔍 ค้นหา",
        use_container_width=True,
        key="fetch_btn",
    ):
        symbol = _extract_symbol(asset_choice)
        fetch_value = _FETCH_TYPES[fetch_type]["value"]
        _execute_fetch(
            user_id,
            db_session_factory,
            symbol,
            fetch_value,
            chat_model,
        )


def _execute_fetch(
    user_id: str,
    db_session_factory: Callable[[], Session],
    symbol: str,
    fetch_type: str,
    chat_model: BaseChatModel | None = None,
) -> None:
    """Execute fetch and save as notification.

    Args:
        user_id: UUID of the current user.
        db_session_factory: Callable that creates DB sessions.
        symbol: Ticker symbol.
        fetch_type: "news" or "price".
        chat_model: Optional LLM for news summarization.
    """
    from finance_ai.tools.scheduler_service import (  # noqa: PLC0415
        execute_immediate_fetch,
    )

    session = db_session_factory()
    try:
        type_label = "ข่าว" if fetch_type == "news" else "ราคา"
        with st.spinner(f"กำลังดึง{type_label} {symbol}..."):
            execute_immediate_fetch(
                session,
                user_id,
                symbol,
                fetch_type,
                chat_model,
            )
        st.success(f"ดึง{type_label} {symbol} สำเร็จ")
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        logger.error("Fetch failed for %s: %s", symbol, exc)
        st.error(f"เกิดข้อผิดพลาด: {exc}")
    finally:
        session.close()


def _render_add_asset_form() -> None:
    """Render form to add a custom asset to the dropdown list."""
    with st.form("add_asset_form", clear_on_submit=True):
        new_symbol = st.text_input(
            "สัญลักษณ์",
            placeholder="เช่น AOT.BK, GOOGL",
        )
        new_name = st.text_input(
            "ชื่อ (ไม่บังคับ)",
            placeholder="เช่น หุ้น AOT",
        )

        if st.form_submit_button(
            "เพิ่มลงรายการ",
            use_container_width=True,
        ):
            _handle_add_asset(new_symbol, new_name)


def _handle_add_asset(symbol: str, name: str) -> None:
    """Add a custom asset to the session list.

    Args:
        symbol: Ticker symbol.
        name: Optional display name.
    """
    if not symbol or not symbol.strip():
        st.error("กรุณาระบุสัญลักษณ์สินทรัพย์")
        return

    clean = symbol.strip().upper()
    label = f"{clean} ({name.strip()})" if name.strip() else clean

    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = []

    existing = _get_asset_list()
    if any(_extract_symbol(a) == clean for a in existing):
        st.warning(f"{clean} มีอยู่ในรายการแล้ว")
        return

    st.session_state[_SESSION_KEY].append(label)
    st.success(f"เพิ่ม **{label}** ลงรายการแล้ว")


def _render_results(
    user_id: str,
    db_session_factory: Callable[[], Session],
) -> None:
    """Render fetch results full-width below the form.

    Args:
        user_id: UUID of the current user.
        db_session_factory: Callable that creates DB sessions.
    """
    st.markdown("#### 📋 ผลลัพธ์")

    from finance_ai.tools.scheduler_service import (  # noqa: PLC0415
        get_unread_notifications,
        mark_all_notifications_read,
    )

    session = db_session_factory()
    try:
        notifications = get_unread_notifications(session, user_id)

        if not notifications:
            st.info("ยังไม่มีผลลัพธ์ — เลือกสินทรัพย์แล้วกด ค้นหา")
            return

        st.caption(f"📬 {len(notifications)} รายการ")
        if st.button(
            "✅ ล้างผลลัพธ์ทั้งหมด",
            key="mark_all_read",
        ):
            mark_all_notifications_read(session, user_id)
            st.rerun()

        for notif in notifications:
            _render_result_card(notif)
    finally:
        session.close()


def _render_result_card(notification: Any) -> None:
    """Render a single result card.

    Args:
        notification: AssetNotification instance.
    """
    created = notification.created_at
    time_str = created.strftime("%d/%m/%Y %H:%M") if created else ""

    st.markdown(f"---\n**{notification.symbol}** · {time_str}")
    st.markdown(notification.content)
