"""Tests for the Guru tab endpoints (GET /gurus, GET /guru/chat/stream)."""

# pylint: disable=unsubscriptable-object
# ^ false positive: sessionmaker IS subscriptable (SQLAlchemy 2.x generic).

# pylint: disable=redefined-outer-name

from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from finance_ai.agents.guru_panel import GuruStance, PanelEvent, PanelSynthesis
from finance_ai.agents.stream_utils import StreamEvent
from finance_ai.database.base import Base
from finance_ai.database.models.conversation import Conversation
from finance_ai.database.models.conversation_message import ConversationMessage
from finance_ai.main import app


@pytest.fixture
def temp_db(monkeypatch: pytest.MonkeyPatch) -> sessionmaker[Session]:
    """In-memory DB patched into main._session_factory."""
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
def stream_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Patch the guru stream with a fake and record its kwargs."""
    calls: list[dict[str, Any]] = []

    def _fake_stream(**kwargs: Any) -> Iterator[StreamEvent]:
        calls.append(kwargs)
        yield StreamEvent(event_type="token", content="มุมมองแบบ Buffett")
        yield StreamEvent(event_type="complete", intent=f"guru:{kwargs['guru_id']}")

    monkeypatch.setattr("finance_ai.main.get_chat_model", lambda: None)
    monkeypatch.setattr("finance_ai.main.stream_guru_query", _fake_stream)
    return calls


def _rows(factory: sessionmaker[Session], model: Any) -> list[Any]:
    """Return all rows of a model from the temp DB."""
    with factory() as session:
        return list(session.query(model).all())


class TestListGurus:
    """Tests for GET /gurus."""

    def test_returns_all_four_personas(self) -> None:
        """The Guru tab gets every shipped persona with its card data."""
        response = TestClient(app).get("/gurus")

        assert response.status_code == 200
        gurus = {guru["guru_id"]: guru for guru in response.json()}
        assert set(gurus) == {"buffett", "dalio", "niwes", "coach_noom"}
        assert gurus["buffett"]["display_name"] == "Warren Buffett"
        assert gurus["coach_noom"]["typical_questions"]


class TestGuruChatStream:
    """Tests for GET /guru/chat/stream."""

    @pytest.mark.usefixtures("temp_db")
    def test_streams_tokens_and_guru_intent(self, stream_calls: list[dict[str, Any]]) -> None:
        """SSE carries the answer and a guru:<id> complete event."""
        response = TestClient(app).get(
            "/guru/chat/stream", params={"query": "ตลาดตก", "user_id": "u1", "guru_id": "buffett"}
        )

        assert response.status_code == 200
        assert 'data: "มุมมองแบบ Buffett"' in response.text
        assert 'event: complete\ndata: "guru:buffett"' in response.text
        assert stream_calls[0]["guru_id"] == "buffett"

    def test_creates_guru_conversation_and_saves_messages(
        self, temp_db: sessionmaker[Session], stream_calls: list[dict[str, Any]]
    ) -> None:
        """A new guru chat is created and both turns are stored."""
        TestClient(app).get(
            "/guru/chat/stream", params={"query": "ตลาดตก", "user_id": "u1", "guru_id": "dalio"}
        )

        conversations = _rows(temp_db, Conversation)
        messages = _rows(temp_db, ConversationMessage)
        assert [(conv.kind, conv.guru_id) for conv in conversations] == [("guru", "dalio")]
        assert [(msg.role, msg.intent) for msg in messages] == [
            ("user", None),
            ("assistant", "guru:dalio"),
        ]
        assert stream_calls

    @pytest.mark.usefixtures("temp_db")
    def test_guru_chat_is_hidden_from_main_sidebar(
        self, stream_calls: list[dict[str, Any]]
    ) -> None:
        """GET /conversations lists main chats; kind=guru lists guru chats."""
        client = TestClient(app)
        client.get("/guru/chat/stream", params={"query": "q", "user_id": "u1", "guru_id": "niwes"})

        main_list = client.get("/conversations", params={"user_id": "u1"}).json()
        guru_list = client.get("/conversations", params={"user_id": "u1", "kind": "guru"}).json()

        assert not main_list
        assert [conv["guru_id"] for conv in guru_list] == ["niwes"]
        assert stream_calls

    def test_unknown_guru_returns_422(
        self, temp_db: sessionmaker[Session], stream_calls: list[dict[str, Any]]
    ) -> None:
        """An invalid guru_id is rejected before anything is saved."""
        response = TestClient(app).get(
            "/guru/chat/stream", params={"query": "q", "user_id": "u1", "guru_id": "elon"}
        )

        assert response.status_code == 422
        assert "Available" in response.json()["detail"]
        assert not _rows(temp_db, ConversationMessage)
        assert not stream_calls


@pytest.fixture
def panel_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Patch run_guru_panel with a fake that answers, fails one, then synthesizes."""
    calls: list[dict[str, Any]] = []

    def _fake_panel(**kwargs: Any) -> Iterator[PanelEvent]:
        calls.append(kwargs)
        yield PanelEvent(event_type="guru_answer", guru_id="buffett", content="จุดยืน: ซื้อเพิ่ม")
        yield PanelEvent(event_type="guru_error", guru_id="dalio", content="quota")
        yield PanelEvent(
            event_type="synthesis",
            synthesis=PanelSynthesis(
                stances=[
                    GuruStance(
                        guru_id="buffett", stance="ซื้อเพิ่ม", key_reason="ราคาถูก", risk="ผันผวน"
                    )
                ],
                consensus="ถือยาว",
                differences="-",
            ),
        )

    monkeypatch.setattr("finance_ai.main.get_chat_model", lambda: None)
    monkeypatch.setattr("finance_ai.main.run_guru_panel", _fake_panel)
    return calls


class TestGuruPanelStream:
    """Tests for GET /guru/panel/stream."""

    def test_streams_named_panel_events(
        self, temp_db: sessionmaker[Session], panel_calls: list[dict[str, Any]]
    ) -> None:
        """Each guru answer, error and the synthesis arrive as named SSE events."""
        response = TestClient(app).get(
            "/guru/panel/stream", params={"query": "ตลาดตก", "user_id": "u1"}
        )

        assert response.status_code == 200
        assert "event: guru_answer\n" in response.text
        assert '"guru_id": "buffett"' in response.text
        assert "event: guru_error\n" in response.text
        assert "event: synthesis\n" in response.text
        assert 'event: complete\ndata: "guru:panel"' in response.text
        assert panel_calls[0]["guru_ids"] == ["buffett", "coach_noom", "dalio", "niwes"]
        assert _rows(temp_db, Conversation)[0].guru_id == "panel"

    def test_saves_answers_and_synthesis(
        self, temp_db: sessionmaker[Session], panel_calls: list[dict[str, Any]]
    ) -> None:
        """The question, each answer and the comparison table are stored."""
        TestClient(app).get(
            "/guru/panel/stream",
            params={"query": "ตลาดตก", "user_id": "u1", "guru_ids": "buffett,dalio"},
        )

        messages = _rows(temp_db, ConversationMessage)
        assert [(msg.role, msg.intent) for msg in messages] == [
            ("user", None),
            ("assistant", "guru:buffett"),
            ("assistant", "guru:synthesis"),
        ]
        assert "| buffett |" in messages[2].content
        assert panel_calls[0]["guru_ids"] == ["buffett", "dalio"]

    def test_unknown_guru_in_list_returns_422(
        self, temp_db: sessionmaker[Session], panel_calls: list[dict[str, Any]]
    ) -> None:
        """Every id in guru_ids must be a known persona."""
        response = TestClient(app).get(
            "/guru/panel/stream",
            params={"query": "q", "user_id": "u1", "guru_ids": "buffett,elon"},
        )

        assert response.status_code == 422
        assert not _rows(temp_db, ConversationMessage)
        assert not panel_calls


class TestCreateGuruConversation:
    """Tests for POST /conversations with the Guru tab fields."""

    def test_creates_guru_conversation_with_truncated_title(
        self, temp_db: sessionmaker[Session]
    ) -> None:
        """The Guru tab creates its chat up front, titled from the first question."""
        response = TestClient(app).post(
            "/conversations",
            json={"user_id": "u1", "title": "ก" * 80, "kind": "guru", "guru_id": "buffett"},
        )

        assert response.status_code == 200
        assert response.json()["guru_id"] == "buffett"
        conversation = _rows(temp_db, Conversation)[0]
        assert (conversation.kind, len(conversation.title)) == ("guru", 50)


class TestGuruConversationOwnership:
    """Guru endpoints only continue the caller's own guru conversation."""

    def _create_conversation(self, user_id: str, guru_id: str) -> str:
        """Create a guru conversation through the API and return its id."""
        response = TestClient(app).post(
            "/conversations",
            json={"user_id": user_id, "title": "q", "kind": "guru", "guru_id": guru_id},
        )
        conversation_id: str = response.json()["id"]
        return conversation_id

    def test_other_users_conversation_returns_404(
        self, temp_db: sessionmaker[Session], stream_calls: list[dict[str, Any]]
    ) -> None:
        """Passing someone else's conversation id is rejected before saving."""
        conversation_id = self._create_conversation("owner", "dalio")

        response = TestClient(app).get(
            "/guru/chat/stream",
            params={
                "query": "q",
                "user_id": "intruder",
                "guru_id": "dalio",
                "conversation_id": conversation_id,
            },
        )

        assert response.status_code == 404
        assert not _rows(temp_db, ConversationMessage)
        assert not stream_calls

    def test_own_conversation_continues(
        self, temp_db: sessionmaker[Session], stream_calls: list[dict[str, Any]]
    ) -> None:
        """The owner can keep chatting in the same guru conversation."""
        conversation_id = self._create_conversation("owner", "dalio")

        response = TestClient(app).get(
            "/guru/chat/stream",
            params={
                "query": "q",
                "user_id": "owner",
                "guru_id": "dalio",
                "conversation_id": conversation_id,
            },
        )

        assert response.status_code == 200
        assert len(_rows(temp_db, Conversation)) == 1
        assert stream_calls

    @pytest.mark.usefixtures("temp_db")
    def test_panel_rejects_single_guru_conversation(
        self, panel_calls: list[dict[str, Any]]
    ) -> None:
        """A one-guru chat cannot be continued as a panel."""
        conversation_id = self._create_conversation("owner", "dalio")

        response = TestClient(app).get(
            "/guru/panel/stream",
            params={"query": "q", "user_id": "owner", "conversation_id": conversation_id},
        )

        assert response.status_code == 404
        assert not panel_calls


class TestStartupGuruRefresh:
    """The background source refresh only starts when enabled."""

    def test_refresh_starts_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GURU_FETCH_ON_STARTUP=true starts the background refresh on startup."""
        started: list[bool] = []
        monkeypatch.setattr(
            "finance_ai.main.start_guru_refresh_in_background", lambda: started.append(True)
        )
        monkeypatch.setattr("finance_ai.main.get_settings", lambda: _Settings(True))

        with TestClient(app):
            pass

        assert started == [True]

    def test_refresh_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Local runs and tests never hit the network on startup."""
        started: list[bool] = []
        monkeypatch.setattr(
            "finance_ai.main.start_guru_refresh_in_background", lambda: started.append(True)
        )
        monkeypatch.setattr("finance_ai.main.get_settings", lambda: _Settings(False))

        with TestClient(app):
            pass

        assert not started


class _Settings:  # pylint: disable=too-few-public-methods
    """Minimal settings stand-in for the startup flag."""

    def __init__(self, guru_fetch_on_startup: bool) -> None:
        self.guru_fetch_on_startup = guru_fetch_on_startup
