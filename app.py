"""Streamlit demo app for Personal Finance AI.

Usage:
    streamlit run app.py
"""

import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # Load .env BEFORE any LangChain imports

from finance_ai.agents.llm_factory import create_chat_model  # noqa: E402
from finance_ai.agents.router_agent import route_query  # noqa: E402
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

INTENT_LABELS: dict[str, str] = {
    "tax": "ภาษี",
    "expense": "ค่าใช้จ่าย",
    "investment": "การลงทุน",
    "planning": "วางแผนการเงิน",
    "recommendation": "คำแนะนำการเงิน",
    "general": "ทั่วไป",
    "unknown": "ไม่ทราบ",
}

MULTI_AGENT_QUERIES: list[str] = [
    (
        "ช่วยคำนวณภาษีปี 2026 ให้หน่อย เงินเดือนเดือนละ 60,000 บาท"
        " มีลูก 1 คน ซื้อ SSF 100,000 บาท"
        " แล้วช่วยดูพอร์ตการลงทุนของฉันด้วยว่ามีกำไรขาดทุนเท่าไหร่"
        " อยากรู้ว่ามีผลกระทบทางภาษีไหม"
    ),
    (
        "อยากวางแผนการเงินปีนี้ ช่วยดูเป้าหมายทั้งหมดของฉันให้หน่อย"
        " แล้วดึงข้อมูลรายจ่ายเดือนนี้กับรายได้ปีนี้มาเทียบด้วย"
        " ว่าฉันออมเงินได้ตามเป้าไหม ต้องปรับอะไรบ้าง"
    ),
    (
        "สรุปค่าใช้จ่ายเดือนนี้ให้หน่อย แยกตามหมวดหมู่"
        " แล้วดูเป้าหมายการเงินของฉันด้วยว่าค่าใช้จ่ายที่เป็นอยู่"
        " กระทบกับเป้าหมายการออมไหม ถ้ากระทบแนะนำวิธีลดรายจ่ายด้วย"
    ),
]

RECOMMENDATION_QUERIES: list[str] = [
    (
        "ช่วยวิเคราะห์การเงินทั้งหมดของฉันให้หน่อย"
        " ดูรายจ่าย รายได้ ภาษี พอร์ตการลงทุน และเป้าหมายทุกอย่าง"
        " แล้วให้คำแนะนำเชิงรุกว่าควรปรับปรุงจุดไหนบ้าง"
        " เรียงตามความเร่งด่วนจากมากไปน้อย"
    ),
    (
        "ตรวจสุขภาพการเงินของฉันให้หน่อย"
        " อยากรู้คะแนนสุขภาพการเงินของฉัน"
        " และปัญหาเร่งด่วนที่ต้องแก้ไขก่อน 3 อันดับแรก"
        " พร้อมแนะนำว่าต้องทำอะไรบ้างเพื่อเพิ่มคะแนน"
    ),
]

SAMPLE_QUERIES: list[str] = [
    "คำนวณภาษี เงินเดือน 50,000 บาท/เดือน มีลูก 1 คน ซื้อ SSF 100,000",
    "จ่ายค่าอาหาร 350 บาท",
    "สรุปรายจ่ายเดือนนี้",
    "เพิ่มหุ้น PTT.BK 100 หุ้น ราคา 35 บาท",
    "ดูพอร์ตของฉัน",
]


@st.cache_resource
def get_chat_model():  # type: ignore[no-untyped-def]
    """Create and cache the LLM chat model."""
    return create_chat_model()


@st.cache_resource
def get_session_factory():  # type: ignore[no-untyped-def]
    """Create and cache the database session factory."""
    engine = create_database_engine()
    return create_session_factory(engine)


def ensure_user_exists(user_id: str) -> None:
    """Create a demo user record in DB if it doesn't exist.

    Args:
        user_id: UUID hex string for the user.
    """
    factory = get_session_factory()
    session = factory()
    try:
        UserCRUD().get_or_create_demo_user(session, user_id)
    finally:
        session.close()


def load_or_create_conversation() -> None:
    """Load the latest conversation from DB or create a new one."""
    factory = get_session_factory()
    session = factory()
    try:
        conv = get_or_create_active_conversation(session, st.session_state.user_id)
        st.session_state.conversation_id = conv.id
        st.session_state.messages = load_conversation_messages(session, conv.id)
    finally:
        session.close()


def start_new_conversation() -> None:
    """Create a new conversation and reset chat state."""
    factory = get_session_factory()
    session = factory()
    try:
        conv = create_conversation(session, st.session_state.user_id)
        st.session_state.conversation_id = conv.id
        st.session_state.messages = []
    finally:
        session.close()


def switch_to_conversation(conversation_id: str) -> None:
    """Switch to an existing conversation, loading its messages.

    Args:
        conversation_id: UUID of the conversation to switch to.
    """
    factory = get_session_factory()
    session = factory()
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
        ensure_user_exists(st.session_state.user_id)
    if "conversation_id" not in st.session_state:
        load_or_create_conversation()


def render_conversation_sidebar() -> None:
    """Render conversation management section in sidebar."""
    if st.button("สร้างแชทใหม่", use_container_width=True):
        start_new_conversation()
        st.rerun()

    factory = get_session_factory()
    session = factory()
    try:
        conversations = list_user_conversations(session, st.session_state.user_id, limit=10)
    finally:
        session.close()

    _render_conversation_list(conversations)


def _render_conversation_list(
    conversations: list[Any],
) -> None:
    """Render clickable conversation list in sidebar.

    Args:
        conversations: List of Conversation objects.
    """
    current_id = st.session_state.get("conversation_id", "")
    for conv in conversations:
        label = f"{'→ ' if conv.id == current_id else ''}{conv.title}"
        if st.button(label, key=f"conv_{conv.id}", use_container_width=True):
            switch_to_conversation(conv.id)
            st.rerun()


def render_sidebar() -> None:
    """Render the sidebar with settings and sample queries."""
    with st.sidebar:
        st.header("ตั้งค่า")
        new_user_id = st.text_input(
            "User ID",
            value=st.session_state.user_id,
            help="ใช้สำหรับ Expense/Investment Agent",
        )
        if new_user_id != st.session_state.user_id:
            st.session_state.user_id = new_user_id
            ensure_user_exists(new_user_id)

        st.divider()
        st.header("ประวัติแชท")
        render_conversation_sidebar()

        st.divider()
        st.header("ตัวอย่างคำถาม")
        for query in SAMPLE_QUERIES:
            if st.button(query, use_container_width=True):
                st.session_state.pending_query = query

        st.divider()
        st.header("ทดสอบ Multi-Agent")
        for query in MULTI_AGENT_QUERIES:
            if st.button(query, use_container_width=True, key=f"multi_{query}"):
                st.session_state.pending_query = query

        st.divider()
        st.header("ทดสอบคำแนะนำเชิงรุก")
        for query in RECOMMENDATION_QUERIES:
            if st.button(query, use_container_width=True, key=f"rec_{query}"):
                st.session_state.pending_query = query

        st.divider()
        if st.button("ล้างประวัติแชท", use_container_width=True):
            start_new_conversation()
            st.rerun()


def render_chat_history() -> None:
    """Render all previous chat messages."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("intent"):
                label = INTENT_LABELS.get(msg["intent"], msg["intent"])
                st.caption(f"Agent: {label}")
            st.markdown(msg["content"])


def _load_chat_history() -> list[tuple[str, str]]:
    """Load recent chat history from DB for LLM context.

    Returns:
        List of (role, content) tuples.
    """
    factory = get_session_factory()
    session = factory()
    try:
        return get_recent_history_as_tuples(session, st.session_state.conversation_id)
    finally:
        session.close()


def _save_user_message_to_db(query: str) -> None:
    """Save user message to DB and auto-title if first message.

    Args:
        query: The user's query text.
    """
    factory = get_session_factory()
    session = factory()
    try:
        save_user_message(session, st.session_state.conversation_id, query)
        _auto_title_if_first(session, query)
    finally:
        session.close()


def _auto_title_if_first(session: Any, query: str) -> None:
    """Update conversation title from first user message.

    Args:
        session: Database session.
        query: The user's query text.
    """
    if len(st.session_state.messages) == 0:
        update_conversation_title(session, st.session_state.conversation_id, query)


def _save_assistant_message_to_db(response: str, intent: str) -> None:
    """Save assistant message to DB.

    Args:
        response: The assistant's response text.
        intent: The classified intent.
    """
    factory = get_session_factory()
    session = factory()
    try:
        save_assistant_message(session, st.session_state.conversation_id, response, intent)
    finally:
        session.close()


def process_query(query: str) -> None:
    """Process a user query through the router agent.

    Flow: load history → save user msg → route → save assistant msg.

    Args:
        query: The user's natural language query.
    """
    chat_history = _load_chat_history()
    _save_user_message_to_db(query)

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    intent, response = _execute_query(query, chat_history)

    with st.chat_message("assistant"):
        label = INTENT_LABELS.get(intent, intent)
        st.caption(f"Agent: {label}")
        st.markdown(response)

    _save_assistant_message_to_db(response, intent)
    st.session_state.messages.append({"role": "assistant", "content": response, "intent": intent})


def _execute_query(query: str, chat_history: list[tuple[str, str]]) -> tuple[str, str]:
    """Execute query through router and return (intent, response).

    Args:
        query: The user's query.
        chat_history: Previous messages for context.

    Returns:
        Tuple of (intent, response).
    """
    try:
        model = get_chat_model()
        result = route_query(
            query,
            chat_model=model,
            user_id=st.session_state.user_id,
            db_session_factory=get_session_factory(),
            chat_history=chat_history,
        )
        intent = result.get("intent", "unknown")
        response = result.get("response", "ไม่สามารถประมวลผลได้")
    except Exception as exc:  # noqa: BLE001
        intent = "error"
        response = f"เกิดข้อผิดพลาด: {exc}"
    return intent, response


def main() -> None:
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title="Finance AI - ผู้ช่วยการเงินส่วนบุคคล",
        page_icon="💰",
        layout="centered",
    )
    st.title("💰 Finance AI")
    st.caption("ผู้ช่วยการเงินส่วนบุคคลสำหรับคนไทย — ภาษี, ค่าใช้จ่าย, การลงทุน")

    init_session_state()
    render_sidebar()
    render_chat_history()

    pending = st.session_state.pop("pending_query", None)
    if pending:
        process_query(pending)

    user_input = st.chat_input("พิมพ์คำถามของคุณที่นี่...")
    if user_input:
        process_query(user_input)


if __name__ == "__main__":
    main()
