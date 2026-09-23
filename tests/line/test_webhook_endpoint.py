"""Tests for the POST /line/webhook endpoint."""

import base64
import hashlib
import hmac
from typing import Any

from sqlalchemy.orm import sessionmaker

from finance_ai.main import app
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.pool import StaticPool

CHANNEL_SECRET = "test-channel-secret"
ACCESS_TOKEN = "test-access-token"

_TEXT_EVENT_BODY = (
    '{"destination": "Udest", "events": [{'
    '"type": "message", "replyToken": "tok", '
    '"source": {"type": "user", "userId": "Uline-user-1"}, '
    '"message": {"type": "text", "text": "ภาษีของฉัน"}}]}'
)


def _sign(body: str) -> str:
    """Compute the LINE signature for a body."""
    digest = hmac.new(CHANNEL_SECRET.encode(), body.encode(), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


class FakeMessagingApi:
    """Captures push_message calls instead of hitting the LINE API."""

    calls: list[dict[str, Any]] = []

    def __init__(self, *args: Any) -> None:
        """Accept the ApiClient argument like the real SDK class."""

    def push_message(self, **kwargs: Any) -> None:
        """Record the push request."""
        FakeMessagingApi.calls.append(kwargs)


def _make_client(test_engine: Engine, monkeypatch: Any) -> TestClient:
    """Build a TestClient wired to the in-memory engine and fake credentials."""
    monkeypatch.setenv("LINE_CHANNEL_SECRET", CHANNEL_SECRET)
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", ACCESS_TOKEN)
    monkeypatch.setattr("finance_ai.main._session_factory", sessionmaker(bind=test_engine))
    monkeypatch.setattr(
        "finance_ai.line.line_bot_service.orchestrate_query",
        lambda **kwargs: {"intent": "tax", "response": "คำตอบภาษี"},
    )
    monkeypatch.setattr("finance_ai.line.messaging_client.MessagingApi", FakeMessagingApi)
    return TestClient(app)


class TestLineWebhookEndpoint:
    """Tests for POST /line/webhook."""

    def setup_method(self) -> None:
        """Clear recorded push calls between tests."""
        FakeMessagingApi.calls = []

    def test_signed_text_message_gets_pushed_reply(
        self, test_engine: Engine, monkeypatch: Any
    ) -> None:
        """A valid signed message returns 200 and pushes the agent reply."""
        client = _make_client(test_engine, monkeypatch)
        response = client.post(
            "/line/webhook",
            content=_TEXT_EVENT_BODY.encode(),
            headers={
                "x-line-signature": _sign(_TEXT_EVENT_BODY),
                "content-type": "application/json",
            },
        )
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert len(FakeMessagingApi.calls) == 1
        request = FakeMessagingApi.calls[0]["push_message_request"]
        assert request.to == "Uline-user-1"
        assert request.messages[0].text == "คำตอบภาษี"

    def test_invalid_signature_rejected(self, test_engine: Engine, monkeypatch: Any) -> None:
        """A bad signature is rejected with 403 and nothing is processed."""
        client = _make_client(test_engine, monkeypatch)
        response = client.post(
            "/line/webhook",
            content=_TEXT_EVENT_BODY.encode(),
            headers={"x-line-signature": "bad-signature", "content-type": "application/json"},
        )
        assert response.status_code == 403
        assert FakeMessagingApi.calls == []

    def test_unconfigured_secret_returns_service_unavailable(
        self, test_engine: Engine, monkeypatch: Any
    ) -> None:
        """Without credentials the endpoint is explicitly unavailable."""
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        monkeypatch.setattr("finance_ai.main._session_factory", sessionmaker(bind=test_engine))
        client = TestClient(app)
        response = client.post(
            "/line/webhook",
            content=_TEXT_EVENT_BODY.encode(),
            headers={
                "x-line-signature": _sign(_TEXT_EVENT_BODY),
                "content-type": "application/json",
            },
        )
        assert response.status_code == 503

    def test_non_text_events_acknowledged_without_action(
        self, test_engine: Engine, monkeypatch: Any
    ) -> None:
        """Non-text events (follow/unfollow/sticker) get 200 but no push."""
        client = _make_client(test_engine, monkeypatch)
        body = (
            '{"destination": "Udest", "events": ['
            '{"type": "follow", "replyToken": "t", '
            '"source": {"type": "user", "userId": "Uline-user-2"}}]}'
        )
        response = client.post(
            "/line/webhook",
            content=body.encode(),
            headers={"x-line-signature": _sign(body), "content-type": "application/json"},
        )
        assert response.status_code == 200
        assert FakeMessagingApi.calls == []
