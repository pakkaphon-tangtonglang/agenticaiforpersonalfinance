"""Streamlit app for Personal Finance AI.

Usage:
    streamlit run app.py
"""

import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from finance_ai.agents.llm_factory import create_chat_model  # noqa: E402
from finance_ai.agents.stream_utils import orchestrate_query_stream  # noqa: E402
from finance_ai.database.crud.user_crud import UserCRUD  # noqa: E402
from finance_ai.database.session import (  # noqa: E402
    create_database_engine,
    create_session_factory,
)
from finance_ai.tools.conversation_service import (  # noqa: E402
    create_conversation,
    get_or_create_active_conversation,
    get_recent_history_as_tuples,
    list_user_conversations,
    load_conversation_messages,
    save_assistant_message,
    save_user_message,
    update_conversation_title,
)
from finance_ai.ui.app_constants import (  # noqa: E402
    CUSTOM_CSS,
    FEATURE_CARDS,
    INTENT_CONFIG,
    MULTI_AGENT_QUERIES,
    RECOMMENDATION_QUERIES,
    REPORT_QUERIES,
    SAMPLE_QUERIES,
)
from finance_ai.ui.dashboard import render_dashboard  # noqa: E402
from finance_ai.ui.evaluation_view import render_evaluation_view  # noqa: E402
from finance_ai.tools.background_scheduler import start_scheduler  # noqa: E402
from finance_ai.ui.schedule_view import render_schedule_view  # noqa: E402
from finance_ai.ui.upload import render_upload_view  # noqa: E402

# ──────────────────────── Cached Resources ────────────────────────


@st.cache_resource
def get_chat_model():  # type: ignore[no-untyped-def]
    """Create and cache the LLM chat model."""
    return create_chat_model()


@st.cache_resource
def get_session_factory():  # type: ignore[no-untyped-def]
    """Create and cache the database session factory."""
    engine = create_database_engine()
    return create_session_factory(engine)


@st.cache_resource
def _init_background_scheduler():  # type: ignore[no-untyped-def]
    """Start the background scheduler once and load active schedules."""
    factory = get_session_factory()
    count = start_scheduler(factory)
    return count


# ──────────────────── Session & Conversation ──────────────────────


def _ensure_user(user_id: str) -> None:
    """Create demo user in DB if needed."""
    session = get_session_factory()()
    try:
        UserCRUD().get_or_create_demo_user(session, user_id)
    finally:
        session.close()


def _load_or_create_conv() -> None:
    """Load latest conversation or create a new one."""
    session = get_session_factory()()
    try:
        conv = get_or_create_active_conversation(session, st.session_state.user_id)
        st.session_state.conversation_id = conv.id
        st.session_state.messages = load_conversation_messages(session, conv.id)
    finally:
        session.close()


def _start_new_conv() -> None:
    """Create new conversation and reset messages."""
    session = get_session_factory()()
    try:
        conv = create_conversation(session, st.session_state.user_id)
        st.session_state.conversation_id = conv.id
        st.session_state.messages = []
    finally:
        session.close()


def _switch_conv(conversation_id: str) -> None:
    """Switch to an existing conversation."""
    session = get_session_factory()()
    try:
        st.session_state.conversation_id = conversation_id
        st.session_state.messages = load_conversation_messages(session, conversation_id)
    finally:
        session.close()


def init_session_state() -> None:
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = uuid.uuid4().hex
        _ensure_user(st.session_state.user_id)
    if "conversation_id" not in st.session_state:
        _load_or_create_conv()


# ──────────────────────────── Sidebar ─────────────────────────────


def _render_conv_list(conversations: list[Any]) -> None:
    """Render clickable conversation list."""
    current_id = st.session_state.get("conversation_id", "")
    for conv in conversations:
        active = conv.id == current_id
        prefix = "💬 " if active else "○ "
        label = conv.title or "แชทใหม่"
        if st.button(
            f"{prefix}{label}",
            key=f"conv_{conv.id}",
            use_container_width=True,
        ):
            _switch_conv(conv.id)
            st.rerun()


def _render_query_buttons(queries: list[dict[str, str]], prefix: str) -> None:
    """Render sample query buttons."""
    for q in queries:
        if st.button(
            q["short"],
            key=f"{prefix}_{q['short']}",
            use_container_width=True,
        ):
            st.session_state.pending_query = q["full"]


def render_sidebar() -> None:
    """Render the sidebar."""
    with st.sidebar:
        st.markdown("### 💬 ประวัติแชท")
        st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
        if st.button("✨ สร้างแชทใหม่", use_container_width=True):
            _start_new_conv()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        session = get_session_factory()()
        try:
            convs = list_user_conversations(session, st.session_state.user_id, limit=10)
        finally:
            session.close()
        if convs:
            _render_conv_list(convs)

        st.divider()
        with st.expander("🚀 ตัวอย่างคำถาม"):
            _render_query_buttons(SAMPLE_QUERIES, "sample")
        with st.expander("🤝 Multi-Agent"):
            _render_query_buttons(MULTI_AGENT_QUERIES, "multi")
        with st.expander("📊 รายงานการเงิน"):
            _render_query_buttons(REPORT_QUERIES, "report")
        with st.expander("💡 คำแนะนำเชิงรุก"):
            _render_query_buttons(RECOMMENDATION_QUERIES, "rec")

        st.divider()
        with st.expander("⚙️ ตั้งค่า"):
            new_uid = st.text_input("User ID", value=st.session_state.user_id)
            if new_uid != st.session_state.user_id:
                st.session_state.user_id = new_uid
                _ensure_user(new_uid)

        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        if st.button("🗑️ ล้างประวัติแชท", use_container_width=True):
            _start_new_conv()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ───────────────────────── Chat View ──────────────────────────────


def _render_intent_badge(intent: str) -> None:
    """Render a colored intent badge."""
    cfg = INTENT_CONFIG.get(intent, INTENT_CONFIG["unknown"])
    html = (
        f'<span class="intent-badge" style="background:{cfg["color"]}">'
        f'{cfg["icon"]} {cfg["label"]}</span>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_welcome() -> None:
    """Render welcome screen with feature cards."""
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown(
            "### 🤖 สวัสดีครับ! ยินดีต้อนรับสู่ Personal Finance AI\n" "พิมพ์คำถาม หรือเลือกตัวอย่างจากเมนูด้านซ้าย"
        )
    cols = st.columns(3) + st.columns(3)
    for col, card in zip(cols, FEATURE_CARDS):
        with col:
            st.info(f"**{card['icon']} {card['title']}**\n\n{card['desc']}")


def render_chat_history() -> None:
    """Render previous chat messages or welcome screen."""
    if not st.session_state.messages:
        _render_welcome()
        return
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("intent"):
                _render_intent_badge(msg["intent"])
            st.markdown(msg["content"])


# ──────────────────── DB Message Helpers ──────────────────────────


def _save_user_msg(query: str) -> None:
    """Save user message to DB and auto-title."""
    session = get_session_factory()()
    try:
        save_user_message(session, st.session_state.conversation_id, query)
        if not st.session_state.messages:
            update_conversation_title(session, st.session_state.conversation_id, query)
    finally:
        session.close()


def _save_assistant_msg(response: str, intent: str) -> None:
    """Save assistant message to DB."""
    session = get_session_factory()()
    try:
        save_assistant_message(session, st.session_state.conversation_id, response, intent)
    finally:
        session.close()


def _load_history() -> list[tuple[str, str]]:
    """Load recent chat history from DB."""
    session = get_session_factory()()
    try:
        return get_recent_history_as_tuples(session, st.session_state.conversation_id)
    finally:
        session.close()


# ──────────────────── Streaming Query ─────────────────────────────


def process_query(query: str) -> None:
    """Process a user query with streaming response."""
    chat_history = _load_history()
    _save_user_msg(query)

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    intent, response = _stream_response(query, chat_history)

    _save_assistant_msg(response, intent)
    st.session_state.messages.append({"role": "assistant", "content": response, "intent": intent})


def _stream_response(
    query: str,
    chat_history: list[tuple[str, str]],
) -> tuple[str, str]:
    """Stream the agent response, return (intent, response)."""
    intent = "unknown"
    full_response = ""

    with st.chat_message("assistant"):
        status_box = st.empty()
        message_box = st.empty()

        for event in orchestrate_query_stream(
            query,
            chat_model=get_chat_model(),
            user_id=st.session_state.user_id,
            db_session_factory=get_session_factory(),
            chat_history=chat_history,
        ):
            if event.event_type == "status":
                status_box.caption(f"⏳ {event.content}")
            elif event.event_type == "token":
                full_response += event.content
                message_box.markdown(full_response + "▌")
            elif event.event_type == "complete":
                intent = event.intent
                status_box.empty()
                message_box.empty()

        _render_intent_badge(intent)
        st.markdown(full_response)

    return intent, full_response


# ─────────────────────────── Main ─────────────────────────────────


def main() -> None:
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title="Personal Finance AI",
        page_icon="💰",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="main-header">'
        "<h1>💰Personal Finance AI</h1>"
        "<p>ผู้ช่วยการเงินส่วนบุคคลอัจฉริยะ"
        " — ภาษี · ค่าใช้จ่าย · การลงทุน · วางแผน</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    init_session_state()
    _init_background_scheduler()
    render_sidebar()

    user_input = st.chat_input("💬 พิมพ์คำถามของคุณที่นี่...")

    tab_chat, tab_dashboard, tab_schedule, tab_upload, tab_eval = st.tabs(
        ["💬 แชท", "📊 แดชบอร์ด", "🔔 แจ้งเตือนสินทรัพย์", "📂 นำเข้าข้อมูล", "🔬 ประเมินระบบ"]
    )

    with tab_chat:
        render_chat_history()
        pending = st.session_state.pop("pending_query", None)
        if pending:
            process_query(pending)
        if user_input:
            process_query(user_input)

    with tab_dashboard:
        render_dashboard(st.session_state.user_id, get_session_factory())

    with tab_schedule:
        render_schedule_view(st.session_state.user_id, get_session_factory(), get_chat_model())

    with tab_upload:
        render_upload_view(st.session_state.user_id, get_session_factory())

    with tab_eval:
        render_evaluation_view(get_chat_model(), get_session_factory())


if __name__ == "__main__":
    main()
