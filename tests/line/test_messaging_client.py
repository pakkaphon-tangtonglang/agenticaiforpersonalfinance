"""Tests for the LINE push messaging client."""

from typing import Any

import pytest

from finance_ai.line.messaging_client import (
    CLARIFY_QUICK_REPLY_LABELS,
    send_line_push,
)


class FakeMessagingApi:
    """Captures push_message calls instead of hitting the LINE API."""

    calls: list[dict[str, Any]] = []

    def __init__(self, *args: Any) -> None:
        """Accept the ApiClient argument like the real SDK class."""

    def push_message(self, **kwargs: Any) -> None:
        """Record the request for assertions."""
        FakeMessagingApi.calls.append(kwargs)


class TestSendLinePush:
    """Tests for send_line_push."""

    def setup_method(self) -> None:
        """Clear recorded calls between tests."""
        FakeMessagingApi.calls = []

    def test_pushes_text_to_recipient(self, monkeypatch: Any) -> None:
        """The answer text is pushed to the LINE user id."""
        monkeypatch.setattr("finance_ai.line.messaging_client.MessagingApi", FakeMessagingApi)
        send_line_push("token-123", "Uline-user-1", "คำตอบจากเอเจนต์")
        assert len(FakeMessagingApi.calls) == 1
        request = FakeMessagingApi.calls[0]["push_message_request"]
        assert request.to == "Uline-user-1"
        assert request.messages[0].text == "คำตอบจากเอเจนต์"

    def test_push_includes_thai_quick_reply(self, monkeypatch: Any) -> None:
        """The clarify menu is attached as tappable Quick Reply options."""
        monkeypatch.setattr("finance_ai.line.messaging_client.MessagingApi", FakeMessagingApi)
        send_line_push("token-123", "Uline-user-1", "คำตอบ")
        request = FakeMessagingApi.calls[0]["push_message_request"]
        quick_reply = request.messages[0].quick_reply
        labels = [item.action.label for item in quick_reply.items]
        assert labels == list(CLARIFY_QUICK_REPLY_LABELS)

    def test_sdk_failure_raises_runtime_error(self, monkeypatch: Any) -> None:
        """API errors surface as a clear runtime error, not a silent drop."""

        class ExplodingApi(FakeMessagingApi):
            def push_message(self, **kwargs: Any) -> None:
                raise ConnectionError("LINE API unreachable")

        monkeypatch.setattr("finance_ai.line.messaging_client.MessagingApi", ExplodingApi)
        with pytest.raises(ConnectionError):
            send_line_push("token-123", "Uline-user-1", "คำตอบ")
