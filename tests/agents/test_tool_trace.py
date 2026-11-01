"""Tests for tool_trace: summaries of tool calls for the chat UI."""

import json

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from finance_ai.agents.tool_trace import (
    MAX_PREVIEW_LINES,
    ToolCallTracker,
    ToolTrace,
    format_argument_values,
    summarize_tool_result,
)


class TestFormatArgumentValues:
    """Tests for format_argument_values."""

    def test_stringifies_scalars(self) -> None:
        """Numbers and strings become strings."""
        assert format_argument_values({"amount": 120, "note": "ข้าว"}) == {
            "amount": "120",
            "note": "ข้าว",
        }

    def test_skips_empty_values(self) -> None:
        """None and empty strings are dropped."""
        assert not format_argument_values({"query": "", "limit": None, "tags": [], "extra": {}})

    def test_truncates_long_values(self) -> None:
        """Long values are cut with an ellipsis."""
        formatted = format_argument_values({"query": "ก" * 300})
        assert len(formatted["query"]) <= 120
        assert formatted["query"].endswith("…")

    def test_lists_become_json(self) -> None:
        """Lists keep Thai text readable."""
        assert format_argument_values({"symbols": ["PTT.BK"]}) == {"symbols": '["PTT.BK"]'}


class TestSummarizeToolResult:
    """Tests for summarize_tool_result."""

    def test_empty_text(self) -> None:
        """Empty result gives no lines."""
        assert not summarize_tool_result("")

    def test_plain_text_lines(self) -> None:
        """Blank lines are skipped and the count is capped."""
        text = "บรรทัด 1\n\nบรรทัด 2\nบรรทัด 3\nบรรทัด 4"
        lines = summarize_tool_result(text)
        assert lines == ["บรรทัด 1", "บรรทัด 2", "บรรทัด 3"][:MAX_PREVIEW_LINES]

    def test_json_object_shows_scalar_fields(self) -> None:
        """JSON objects show key: value pairs for simple fields."""
        text = json.dumps({"total_tax": "12500.00", "items": [1, 2], "ok": True})
        assert summarize_tool_result(text) == ["total_tax: 12500.00", "ok: True"]

    def test_json_without_scalars_falls_back_to_text(self) -> None:
        """A JSON list is shown as text."""
        assert summarize_tool_result("[1, 2]") == ["[1, 2]"]


class TestToolCallTracker:
    """Tests for ToolCallTracker."""

    def test_args_from_complete_message(self) -> None:
        """A complete AIMessage records args by call id."""
        tracker = ToolCallTracker()
        tracker.observe(
            AIMessage(
                content="",
                tool_calls=[{"name": "calculate_thai_tax", "args": {"income": 500000}, "id": "c1"}],
            )
        )
        trace = tracker.trace_for(
            ToolMessage(content="ภาษี 12,500 บาท", tool_call_id="c1", name="calculate_thai_tax")
        )
        assert trace == ToolTrace(
            call_id="c1",
            name="calculate_thai_tax",
            args={"income": "500000"},
            preview=["ภาษี 12,500 บาท"],
            failed=False,
        )

    def test_args_from_streamed_chunks(self) -> None:
        """Chunked tool-call args are joined before parsing."""
        tracker = ToolCallTracker()
        tracker.observe(
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "get_stock_price", "args": '{"sym', "id": "c2", "index": 0}
                ],
            )
        )
        tracker.observe(
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": None, "args": 'bol": "PTT.BK"}', "id": None, "index": 0}
                ],
            )
        )
        trace = tracker.trace_for(ToolMessage(content="35.50", tool_call_id="c2"))
        assert trace.args == {"symbol": "PTT.BK"}
        assert trace.name == "get_stock_price"

    def test_unknown_call_id(self) -> None:
        """A tool message with no matching call still gives a trace."""
        trace = ToolCallTracker().trace_for(
            ToolMessage(content="x", tool_call_id="zz", name="add_expense")
        )
        assert trace.name == "add_expense"
        assert not trace.args

    def test_error_status_marks_failed(self) -> None:
        """Tool errors are flagged."""
        trace = ToolCallTracker().trace_for(
            ToolMessage(
                content="Error: boom", tool_call_id="e1", name="add_expense", status="error"
            )
        )
        assert trace.failed

    def test_ignores_other_chunks(self) -> None:
        """Text-only chunks do not break tracking."""
        tracker = ToolCallTracker()
        tracker.observe(AIMessageChunk(content="สวัสดี"))
        tracker.observe("not a message")
        trace = tracker.trace_for(ToolMessage(content="ok", tool_call_id="n1"))
        assert trace.name == "tool"
