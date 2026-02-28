"""Streamlit demo app for Personal Finance AI.

Usage:
    streamlit run app.py
"""

import uuid

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

INTENT_LABELS: dict[str, str] = {
    "tax": "ภาษี",
    "expense": "ค่าใช้จ่าย",
    "investment": "การลงทุน",
    "general": "ทั่วไป",
    "unknown": "ไม่ทราบ",
}

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


def init_session_state() -> None:
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = uuid.uuid4().hex
        ensure_user_exists(st.session_state.user_id)


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
        st.header("ตัวอย่างคำถาม")
        for query in SAMPLE_QUERIES:
            if st.button(query, use_container_width=True):
                st.session_state.pending_query = query

        st.divider()
        if st.button("ล้างประวัติแชท", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_chat_history() -> None:
    """Render all previous chat messages."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("intent"):
                label = INTENT_LABELS.get(msg["intent"], msg["intent"])
                st.caption(f"Agent: {label}")
            st.markdown(msg["content"])


def process_query(query: str) -> None:
    """Process a user query through the router agent.

    Args:
        query: The user's natural language query.
    """
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("กำลังคิด..."):
            try:
                model = get_chat_model()
                result = route_query(
                    query,
                    chat_model=model,
                    user_id=st.session_state.user_id,
                    db_session_factory=get_session_factory(),
                )
                intent = result.get("intent", "unknown")
                response = result.get("response", "ไม่สามารถประมวลผลได้")
            except Exception as exc:  # noqa: BLE001
                intent = "error"
                response = f"เกิดข้อผิดพลาด: {exc}"

        label = INTENT_LABELS.get(intent, intent)
        st.caption(f"Agent: {label}")
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response, "intent": intent})


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
