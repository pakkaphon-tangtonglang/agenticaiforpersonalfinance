"""Tests for SSE event formatting in the chat stream endpoint.

The stream must preserve Thai characters and newlines inside token content.
SSE ``data:`` fields cannot contain raw newlines, so content is JSON-encoded.
"""

import json
from typing import cast

from finance_ai.agents.stream_utils import StreamEvent
from finance_ai.main import _format_sse_event


def _token_sse(event: StreamEvent) -> str:
    """Return the SSE string for an event, asserting it is not ``None``.

    Args:
        event: StreamEvent to format.

    Returns:
        SSE-formatted string.
    """
    sse = _format_sse_event(event)
    assert sse is not None, "expected an SSE event, got None"
    return sse


def _parse_sse_data(sse: str) -> str:
    """Extract and JSON-decode the payload of a single ``data:`` SSE event.

    Args:
        sse: Raw SSE string emitted by ``_format_sse_event``.

    Returns:
        Decoded content string.
    """
    assert sse.startswith("data: "), f"expected data: prefix, got: {sse!r}"
    assert sse.endswith("\n\n"), f"expected \\n\\n terminator, got: {sse!r}"
    payload = sse[len("data: ") : -2]
    return cast(str, json.loads(payload))


class TestFormatSseEvent:
    """Tests for :func:`_format_sse_event`."""

    def test_plain_thai_token_round_trips(self) -> None:
        """A plain Thai token survives the SSE encoding losslessly."""
        event = StreamEvent(event_type="token", content="คำนวณภาษีให้คุณเลยครับ")
        assert _parse_sse_data(_token_sse(event)) == "คำนวณภาษีให้คุณเลยครับ"

    def test_token_with_newline_round_trips(self) -> None:
        """Newlines inside content are preserved (not dropped by SSE parser)."""
        content = "ค่าลดหย่อนส่วนตัว\nค่าลดหย่อนบุตร\nกองทุน SSF"
        event = StreamEvent(event_type="token", content=content)
        assert _parse_sse_data(_token_sse(event)) == content

    def test_token_with_leading_newline_round_trips(self) -> None:
        """A leading newline (common at table-row boundaries) is preserved."""
        content = "\n| รายการ | จำนวนเงิน |"
        event = StreamEvent(event_type="token", content=content)
        decoded = _parse_sse_data(_token_sse(event))
        assert decoded == content
        assert decoded.startswith("\n")

    def test_token_with_pipe_and_emoji_round_trips(self) -> None:
        """Markdown table pipes and emoji survive encoding."""
        content = "| 💰 ภาษีที่ต้องจ่าย | **8,500 บาท** |"
        event = StreamEvent(event_type="token", content=content)
        assert _parse_sse_data(_token_sse(event)) == content

    def test_data_field_has_no_raw_newline(self) -> None:
        """The SSE data line itself contains no raw newline (SSE-safe)."""
        event = StreamEvent(event_type="token", content="a\nb\nc")
        sse = _token_sse(event)
        data_line = sse[: -len("\n\n")]
        assert "\n" not in data_line, "raw newline in data: would break EventSource"

    def test_complete_event_has_event_header(self) -> None:
        """The complete event carries an ``event: complete`` header."""
        event = StreamEvent(event_type="complete", intent="tax")
        sse = _token_sse(event)
        assert sse.startswith("event: complete\n")
        assert "tax" in sse

    def test_status_event_returns_none(self) -> None:
        """Status events are not emitted to the client."""
        event = StreamEvent(event_type="status", content="กำลังประมวลผล...")
        assert _format_sse_event(event) is None
