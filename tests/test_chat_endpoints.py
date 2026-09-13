"""Tests for the chat endpoints (POST /chat and GET /chat/stream).

Postgres rejects writes with a blank conversation_id (FK violation);
these tests enable SQLite foreign-key enforcement so the same class of
bug is caught locally.
"""

# pylint: disable=unsubscriptable-object
# ^ false positive: sessionmaker IS subscriptable (SQLAlchemy 2.x generic);
#   astroid cannot infer it, unlike mypy.

# pylint: disable=redefined-outer-name

from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from finance_ai.agents.stream_utils import StreamEvent
from finance_ai.database.base import Base
from finance_ai.database.models.conversation import Conversation
from finance_ai.database.models.conversation_message import ConversationMessage
from finance_ai.main import app


@pytest.fixture
def temp_db(monkeypatch: pytest.MonkeyPatch) -> sessionmaker[Session]:
    """Create an FK-enforcing in-memory DB and patch main._session_factory.

    SQLite skips FK checks by default (Postgres does not) — enabling the
    pragma makes these tests fail exactly like production would.

    Returns:
        sessionmaker bound to the in-memory engine (schema created).
    """
    engine: Engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _enable_fk_constraints(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr("finance_ai.main._session_factory", factory)
    return factory


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient with fake LLM collaborators patched out.

    No DB fixture here on purpose: each test requests ``temp_db`` itself,
    so the session-factory patch is in place whenever the endpoint hits
    the database.
    """
    monkeypatch.setattr("finance_ai.main.get_chat_model", lambda: None)
    monkeypatch.setattr(
        "finance_ai.main.orchestrate_query",
        lambda **_kwargs: {"intent": "general_chat", "response": "ตอบกลับ"},
    )

    def _fake_stream(**_kwargs: Any) -> Iterator[StreamEvent]:
        yield StreamEvent(event_type="status", content="กำลังประมวลผล")
        yield StreamEvent(event_type="token", content="ตอบกลับ")
        yield StreamEvent(event_type="complete", intent="general_chat")

    monkeypatch.setattr("finance_ai.main.orchestrate_query_stream", _fake_stream)
    return TestClient(app)


def _rows(factory: sessionmaker[Session], model: Any) -> list[Any]:
    """Return all rows of a model from the temp DB."""
    with factory() as session:
        return list(session.query(model).all())


class TestPostChat:
    """Tests for POST /chat."""

    def test_no_conversation_id_creates_conversation(
        self, client: TestClient, temp_db: sessionmaker[Session]
    ) -> None:
        """Missing conversation_id must not 500: a conversation is created."""
        user_id = "chat-user-no-conv"
        response = client.post(
            "/chat",
            json={"query": "สวัสดี", "user_id": user_id},
        )
        assert response.status_code == 200
        assert response.json()["response"] == "ตอบกลับ"

        conversations = _rows(temp_db, Conversation)
        assert len(conversations) == 1
        assert conversations[0].user_id == user_id

    def test_no_conversation_id_saves_both_messages(
        self, client: TestClient, temp_db: sessionmaker[Session]
    ) -> None:
        """User + assistant messages land in the auto-created conversation."""
        client.post("/chat", json={"query": "สวัสดี", "user_id": "chat-user-2"})

        messages = _rows(temp_db, ConversationMessage)
        roles = sorted(message.role for message in messages)
        assert roles == ["assistant", "user"]

    def test_missing_id_reuses_latest_conversation(
        self, client: TestClient, temp_db: sessionmaker[Session]
    ) -> None:
        """A returning user without an id continues their conversation."""
        client.post("/chat", json={"query": "คำถาม 1", "user_id": "chat-user-3"})
        client.post("/chat", json={"query": "คำถาม 2", "user_id": "chat-user-3"})

        conversations = _rows(temp_db, Conversation)
        assert len(conversations) == 1

    def test_explicit_conversation_id_is_kept(
        self, client: TestClient, temp_db: sessionmaker[Session]
    ) -> None:
        """An explicit conversation_id is used as-is (no extra one created)."""
        client.post("/chat", json={"query": "คำถาม", "user_id": "chat-user-4"})
        first = _rows(temp_db, Conversation)[0]

        client.post(
            "/chat",
            json={
                "query": "ต่อเนื่อง",
                "user_id": "chat-user-4",
                "conversation_id": first.id,
            },
        )
        assert len(_rows(temp_db, Conversation)) == 1


class TestChatStream:
    """Tests for GET /chat/stream."""

    def test_stream_without_conversation_id_succeeds(
        self, client: TestClient, temp_db: sessionmaker[Session]
    ) -> None:
        """Missing conversation_id streams 200 and persists messages."""
        response = client.get(
            "/chat/stream",
            params={"query": "สวัสดี", "user_id": "stream-user-1"},
        )
        assert response.status_code == 200
        assert "data:" in response.text

        assert len(_rows(temp_db, Conversation)) == 1
        roles = sorted(message.role for message in _rows(temp_db, ConversationMessage))
        assert roles == ["assistant", "user"]
