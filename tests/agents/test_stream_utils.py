"""Tests for finance_ai.agents.stream_utils."""

from unittest.mock import MagicMock, patch

import pytest

from finance_ai.agents.stream_utils import (
    StreamEvent,
    _extract_tool_name,
    _handle_ai_chunk,
    _process_chunk,
    orchestrate_query_stream,
    stream_agent_response,
)


class TestStreamEvent:
    """Tests for StreamEvent model."""

    def test_default_values(self) -> None:
        """Default event_type is 'token' with empty content."""
        event = StreamEvent()
        assert event.event_type == "token"
        assert event.content == ""
        assert event.intent == ""

    def test_status_event(self) -> None:
        """Status event has correct type."""
        event = StreamEvent(event_type="status", content="loading")
        assert event.event_type == "status"

    def test_complete_event_with_intent(self) -> None:
        """Complete event can carry intent."""
        event = StreamEvent(event_type="complete", content="done", intent="tax")
        assert event.intent == "tax"


class TestProcessChunk:
    """Tests for _process_chunk."""

    def test_non_ai_chunk_returns_none(self) -> None:
        """Non-AIMessageChunk returns None."""
        assert _process_chunk("string") is None
        assert _process_chunk(42) is None

    def test_ai_chunk_with_content(self) -> None:
        """AIMessageChunk with content returns token event."""
        from langchain_core.messages import AIMessageChunk

        chunk = AIMessageChunk(content="hello")
        event = _process_chunk(chunk)
        assert event is not None
        assert event.event_type == "token"
        assert event.content == "hello"

    def test_ai_chunk_empty_content(self) -> None:
        """AIMessageChunk with empty content returns None."""
        from langchain_core.messages import AIMessageChunk

        chunk = AIMessageChunk(content="")
        event = _process_chunk(chunk)
        assert event is None


class TestHandleAiChunk:
    """Tests for _handle_ai_chunk."""

    def test_tool_call_chunk(self) -> None:
        """Tool call chunk returns status event."""
        from langchain_core.messages import AIMessageChunk

        chunk = AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": "calculate_tax", "args": "", "id": "1", "index": 0}],
        )
        event = _handle_ai_chunk(chunk)
        assert event is not None
        assert event.event_type == "status"
        assert "calculate_tax" in event.content

    def test_content_chunk(self) -> None:
        """Content chunk returns token event."""
        from langchain_core.messages import AIMessageChunk

        chunk = AIMessageChunk(content="ภาษี")
        event = _handle_ai_chunk(chunk)
        assert event is not None
        assert event.event_type == "token"
        assert event.content == "ภาษี"


class TestExtractToolName:
    """Tests for _extract_tool_name."""

    def test_extracts_name(self) -> None:
        """Tool name is extracted from first chunk."""
        from langchain_core.messages import AIMessageChunk

        chunk = AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": "add_expense", "args": "", "id": "1", "index": 0}],
        )
        assert _extract_tool_name(chunk) == "add_expense"

    def test_empty_chunks(self) -> None:
        """Empty tool call chunks returns empty string."""
        from langchain_core.messages import AIMessageChunk

        chunk = AIMessageChunk(content="", tool_call_chunks=[])
        assert _extract_tool_name(chunk) == ""


class TestStreamAgentResponse:
    """Tests for stream_agent_response."""

    def test_yields_status_first(self) -> None:
        """First event is a status event."""
        mock_graph = MagicMock()
        mock_graph.stream.return_value = iter([])

        events = list(stream_agent_response(mock_graph, {}, "tax"))
        assert events[0].event_type == "status"
        assert events[-1].event_type == "complete"

    def test_yields_complete_last(self) -> None:
        """Last event is a complete event with intent."""
        mock_graph = MagicMock()
        mock_graph.stream.return_value = iter([])

        events = list(stream_agent_response(mock_graph, {}, "expense"))
        assert events[-1].event_type == "complete"
        assert events[-1].intent == "expense"

    def test_handles_stream_error(self) -> None:
        """Stream error yields error token then complete."""
        mock_graph = MagicMock()
        mock_graph.stream.side_effect = RuntimeError("LLM down")

        events = list(stream_agent_response(mock_graph, {}, "tax"))
        token_events = [e for e in events if e.event_type == "token"]
        assert any("ข้อผิดพลาด" in e.content for e in token_events)
        assert events[-1].event_type == "complete"


class TestRouteQueryStream:
    """Tests for orchestrate_query_stream."""

    @patch("finance_ai.agents.stream_utils.classify_query")
    @patch("finance_ai.agents.stream_utils._get_agent_graph")
    def test_unsupported_intent(
        self,
        mock_get_graph: MagicMock,
        mock_classify: MagicMock,
    ) -> None:
        """Unsupported intent yields response then complete."""
        from decimal import Decimal

        from finance_ai.agents.schemas import OrchestratorDecision

        mock_classify.return_value = OrchestratorDecision(
            intent="unknown", confidence=Decimal("0.5")
        )
        mock_get_graph.return_value = None

        events = list(orchestrate_query_stream("random question"))
        assert any(e.event_type == "token" for e in events)
        assert events[-1].event_type == "complete"
        assert events[-1].intent == "unknown"

    @patch("finance_ai.agents.stream_utils.classify_query")
    @patch("finance_ai.agents.stream_utils._get_agent_graph")
    def test_supported_intent_streams(
        self,
        mock_get_graph: MagicMock,
        mock_classify: MagicMock,
    ) -> None:
        """Supported intent delegates to stream_agent_response."""
        from decimal import Decimal

        from finance_ai.agents.schemas import OrchestratorDecision

        mock_classify.return_value = OrchestratorDecision(intent="tax", confidence=Decimal("0.95"))
        mock_graph = MagicMock()
        mock_graph.stream.return_value = iter([])
        mock_get_graph.return_value = mock_graph

        events = list(orchestrate_query_stream("คำนวณภาษี"))
        assert events[0].event_type == "status"
        assert events[-1].event_type == "complete"
        assert events[-1].intent == "tax"

    @patch("finance_ai.agents.stream_utils.classify_query")
    @patch("finance_ai.agents.stream_utils._get_agent_graph")
    def test_general_intent_streams_via_planning(
        self,
        mock_get_graph: MagicMock,
        mock_classify: MagicMock,
    ) -> None:
        """General intent streams through planning agent graph."""
        from decimal import Decimal

        from finance_ai.agents.schemas import OrchestratorDecision

        mock_classify.return_value = OrchestratorDecision(
            intent="general",
            confidence=Decimal("0.8"),
        )
        mock_graph = MagicMock()
        mock_graph.stream.return_value = iter([])
        mock_get_graph.return_value = mock_graph

        events = list(orchestrate_query_stream("ออมเงินยังไงดี"))
        assert events[0].event_type == "status"
        assert events[-1].event_type == "complete"
        assert events[-1].intent == "general"
        mock_get_graph.assert_called_once_with("general", None)
