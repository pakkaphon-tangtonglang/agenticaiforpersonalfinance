"""Tests for conversation management endpoints (rename, auto-title, delete)."""

# pylint: disable=unsubscriptable-object
# ^ false positive: sessionmaker IS subscriptable (SQLAlchemy 2.x generic).

# pylint: disable=redefined-outer-name

from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from finance_ai.database.base import Base
from finance_ai.database.models.conversation import Conversation
from finance_ai.main import app
from finance_ai.tools.conversation_service import (
    create_conversation,
    save_assistant_message,
    save_user_message,
)

USER_ID = "conv-endpoint-user"


class _FakeChatModel:
    """Chat model double that always returns the same title."""

    def invoke(self, _messages: Any) -> AIMessage:
        """Return a fixed title."""
        return AIMessage(content="คำนวณภาษีเงินเดือน")


@pytest.fixture
def temp_db(monkeypatch: pytest.MonkeyPatch) -> sessionmaker[Session]:
    """Create an in-memory DB and patch main._session_factory to use it."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr("finance_ai.main._session_factory", factory)
    return factory


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient with the chat model replaced by a fake."""
    monkeypatch.setattr("finance_ai.main.get_chat_model", _FakeChatModel)
    return TestClient(app)


def _make_conversation(factory: sessionmaker[Session], with_exchange: bool = True) -> str:
    """Create a default-titled conversation, optionally with one exchange."""
    with factory() as session:
        conversation = create_conversation(session, USER_ID)
        if with_exchange:
            save_user_message(session, conversation.id, "คำนวณภาษีเงินเดือน 50,000")
            save_assistant_message(session, conversation.id, "ภาษีของคุณคือ...", "tax")
        return conversation.id


def _title(factory: sessionmaker[Session], conversation_id: str) -> str | None:
    """Return the stored title, or None if the conversation is gone."""
    with factory() as session:
        conversation = session.get(Conversation, conversation_id)
        return conversation.title if conversation else None


class TestRenameConversationEndpoint:
    """Tests for PATCH /conversations/{id}."""

    def test_renames_conversation(self, client: TestClient, temp_db: sessionmaker[Session]) -> None:
        """The owner can rename a conversation."""
        conversation_id = _make_conversation(temp_db)
        response = client.patch(
            f"/conversations/{conversation_id}",
            json={"user_id": USER_ID, "title": "  ภาษีปี 2026  "},
        )
        assert response.status_code == 200
        assert response.json()["title"] == "ภาษีปี 2026"
        assert _title(temp_db, conversation_id) == "ภาษีปี 2026"

    def test_blank_title_is_rejected(
        self, client: TestClient, temp_db: sessionmaker[Session]
    ) -> None:
        """A whitespace-only title fails validation."""
        conversation_id = _make_conversation(temp_db)
        response = client.patch(
            f"/conversations/{conversation_id}",
            json={"user_id": USER_ID, "title": "   "},
        )
        assert response.status_code == 422

    def test_other_user_gets_404(self, client: TestClient, temp_db: sessionmaker[Session]) -> None:
        """Another user cannot rename the conversation."""
        conversation_id = _make_conversation(temp_db)
        response = client.patch(
            f"/conversations/{conversation_id}",
            json={"user_id": "someone-else", "title": "x"},
        )
        assert response.status_code == 404


class TestAutoTitleEndpoint:
    """Tests for POST /conversations/{id}/auto-title."""

    def test_generates_title(self, client: TestClient, temp_db: sessionmaker[Session]) -> None:
        """A default-titled conversation is named by the LLM."""
        conversation_id = _make_conversation(temp_db)
        response = client.post(
            f"/conversations/{conversation_id}/auto-title", json={"user_id": USER_ID}
        )
        assert response.status_code == 200
        assert response.json()["title"] == "คำนวณภาษีเงินเดือน"
        assert _title(temp_db, conversation_id) == "คำนวณภาษีเงินเดือน"

    def test_missing_conversation_gets_404(
        self, client: TestClient, temp_db: sessionmaker[Session]
    ) -> None:
        """An unknown conversation id returns 404."""
        _make_conversation(temp_db)
        response = client.post("/conversations/missing/auto-title", json={"user_id": USER_ID})
        assert response.status_code == 404


class TestDeleteConversationEndpoint:
    """Tests for DELETE /conversations/{id}."""

    def test_deletes_conversation(self, client: TestClient, temp_db: sessionmaker[Session]) -> None:
        """The owner can delete a conversation."""
        conversation_id = _make_conversation(temp_db)
        response = client.delete(f"/conversations/{conversation_id}", params={"user_id": USER_ID})
        assert response.status_code == 200
        assert _title(temp_db, conversation_id) is None

    def test_other_user_gets_404(self, client: TestClient, temp_db: sessionmaker[Session]) -> None:
        """Another user cannot delete the conversation."""
        conversation_id = _make_conversation(temp_db)
        response = client.delete(
            f"/conversations/{conversation_id}", params={"user_id": "someone-else"}
        )
        assert response.status_code == 404
        assert _title(temp_db, conversation_id) is not None
