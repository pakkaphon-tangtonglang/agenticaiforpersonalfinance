"""Tests for POST /conversations/{id}/attachments (in-chat file uploads)."""

# pylint: disable=unsubscriptable-object,redefined-outer-name

from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from finance_ai.database.base import Base
from finance_ai.database.models.conversation_message import ConversationMessage
from finance_ai.main import app


@pytest.fixture
def temp_db(monkeypatch: pytest.MonkeyPatch) -> sessionmaker[Session]:
    """In-memory DB patched into main._session_factory."""
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr("finance_ai.main._session_factory", factory)
    return factory


def _rows(factory: sessionmaker[Session]) -> list[Any]:
    """All conversation messages in the temp DB."""
    with factory() as session:
        return list(session.query(ConversationMessage).all())


def _new_chat(client: TestClient, user_id: str) -> str:
    """Create a main-chat conversation and return its id."""
    conversation_id: str = client.post("/conversations", json={"user_id": user_id}).json()["id"]
    return conversation_id


class TestAttachmentNote:
    """Tests for recording an attachment in a chat."""

    def test_saves_note_into_own_chat(self, temp_db: sessionmaker[Session]) -> None:
        """The file and its summary are stored with the upload intent."""
        client = TestClient(app)
        conversation_id = _new_chat(client, "u1")

        response = client.post(
            f"/conversations/{conversation_id}/attachments",
            json={"user_id": "u1", "file_name": "slip.jpg", "summary": "บันทึกแล้ว 1 รายการ"},
        )

        assert response.status_code == 200
        assert [(row.role, row.intent) for row in _rows(temp_db)] == [
            ("user", None),
            ("assistant", "upload"),
        ]

    def test_other_users_chat_returns_404(self, temp_db: sessionmaker[Session]) -> None:
        """A note cannot be written into someone else's conversation."""
        client = TestClient(app)
        conversation_id = _new_chat(client, "owner")

        response = client.post(
            f"/conversations/{conversation_id}/attachments",
            json={"user_id": "intruder", "file_name": "a.jpg", "summary": "x"},
        )

        assert response.status_code == 404
        assert not _rows(temp_db)

    @pytest.mark.usefixtures("temp_db")
    def test_empty_summary_is_rejected(self) -> None:
        """Blank file names or summaries fail validation."""
        client = TestClient(app)
        conversation_id = _new_chat(client, "u1")

        response = client.post(
            f"/conversations/{conversation_id}/attachments",
            json={"user_id": "u1", "file_name": "a.jpg", "summary": ""},
        )

        assert response.status_code == 422
