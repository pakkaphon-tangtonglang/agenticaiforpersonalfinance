"""Tests for the LINE push messaging client."""

from typing import Any

import pytest

from finance_ai.line.messaging_client import (
    CLARIFY_QUICK_REPLY_LABELS,
    LOADING_ANIMATION_SECONDS,
    send_line_push,
    send_line_reply,
    show_line_loading,
)


class FakeMessagingApi:
    """Captures push_message calls instead of hitting the LINE API."""

    calls: list[dict[str, Any]] = []

    def __init__(self, *args: Any) -> None:
        """Accept the ApiClient argument like the real SDK class."""

    def push_message(self, **kwargs: Any) -> None:
        """Record the request for assertions."""
        FakeMessagingApi.calls.append(kwargs)

    def show_loading_animation(self, **kwargs: Any) -> None:
        """Record the loading request for assertions."""
        FakeMessagingApi.calls.append(kwargs)

    def reply_message(self, **kwargs: Any) -> None:
        """Record the reply request for assertions."""
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


class TestShowLineLoading:
    """Tests for show_line_loading (the LINE "typing" indicator)."""

    def setup_method(self) -> None:
        """Clear recorded calls between tests."""
        FakeMessagingApi.calls = []

    def test_requests_loading_animation_for_chat(self, monkeypatch: Any) -> None:
        """The loading animation is shown in the user's 1:1 chat."""
        monkeypatch.setattr("finance_ai.line.messaging_client.MessagingApi", FakeMessagingApi)
        show_line_loading("token-123", "Uline-user-1")
        request = FakeMessagingApi.calls[0]["show_loading_animation_request"]
        assert request.chat_id == "Uline-user-1"
        assert request.loading_seconds == LOADING_ANIMATION_SECONDS

    def test_loading_seconds_is_valid_for_line(self) -> None:
        """LINE accepts 5-60 seconds in steps of 5."""
        assert 5 <= LOADING_ANIMATION_SECONDS <= 60
        assert LOADING_ANIMATION_SECONDS % 5 == 0

    def test_api_failure_is_swallowed(self, monkeypatch: Any) -> None:
        """The indicator is cosmetic, so its failure must not block the answer."""

        class ExplodingApi(FakeMessagingApi):
            def show_loading_animation(self, **kwargs: Any) -> None:
                raise ConnectionError("LINE API unreachable")

        monkeypatch.setattr("finance_ai.line.messaging_client.MessagingApi", ExplodingApi)
        show_line_loading("token-123", "Uline-user-1")


class TestSendLineReply:
    """Tests for send_line_reply (free Reply API, used for the processing notice)."""

    def setup_method(self) -> None:
        """Clear recorded calls between tests."""
        FakeMessagingApi.calls = []

    def test_replies_with_text_to_token(self, monkeypatch: Any) -> None:
        """The text is sent with the event's reply token."""
        monkeypatch.setattr("finance_ai.line.messaging_client.MessagingApi", FakeMessagingApi)
        send_line_reply("token-123", "reply-token-1", "ได้รับคำถามแล้ว")
        request = FakeMessagingApi.calls[0]["reply_message_request"]
        assert request.reply_token == "reply-token-1"
        assert request.messages[0].text == "ได้รับคำถามแล้ว"
