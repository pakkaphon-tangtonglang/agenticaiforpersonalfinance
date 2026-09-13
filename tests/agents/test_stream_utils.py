"""Tests for finance_ai.agents.stream_utils."""

from unittest.mock import MagicMock, patch

import pytest

from finance_ai.agents.stream_utils import (
    StreamEvent,
    _extract_last_ai_from_messages,
    _extract_tool_name,
    _handle_ai_chunk,
    _handle_complete_ai_message,
    _invoke_and_extract,
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


class TestHandleCompleteAiMessage:
    """Tests for _handle_complete_ai_message."""

    def test_content_message_returns_token(self) -> None:
        """Complete AIMessage with content returns token event."""
        from langchain_core.messages import AIMessage

        msg = AIMessage(content="บันทึกค่าอาหาร 350 บาท เรียบร้อย")
        event = _handle_complete_ai_message(msg)
        assert event is not None
        assert event.event_type == "token"
        assert "350" in event.content

    def test_tool_call_message_returns_status(self) -> None:
        """Complete AIMessage with tool_calls returns status event."""
        from langchain_core.messages import AIMessage

        msg = AIMessage(
            content="",
            tool_calls=[{"name": "add_expense", "args": {}, "id": "1"}],
        )
        event = _handle_complete_ai_message(msg)
        assert event is not None
        assert event.event_type == "status"
        assert "add_expense" in event.content

    def test_empty_message_returns_none(self) -> None:
        """Complete AIMessage with no content or tools returns None."""
        from langchain_core.messages import AIMessage

        msg = AIMessage(content="")
        event = _handle_complete_ai_message(msg)
        assert event is None


class TestProcessChunkCompleteMessage:
    """Tests for _process_chunk with complete AIMessage."""

    def test_complete_ai_message_with_content(self) -> None:
        """Complete AIMessage (not chunk) is handled."""
        from langchain_core.messages import AIMessage

        msg = AIMessage(content="สรุปค่าใช้จ่าย 5,000 บาท")
        event = _process_chunk(msg)
        assert event is not None
        assert event.event_type == "token"
        assert "5,000" in event.content

    def test_ai_message_chunk_still_works(self) -> None:
        """AIMessageChunk is still handled (not broken by AIMessage support)."""
        from langchain_core.messages import AIMessageChunk

        chunk = AIMessageChunk(content="hello")
        event = _process_chunk(chunk)
        assert event is not None
        assert event.event_type == "token"


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

    @patch("finance_ai.agents.stream_utils.execute_general_chat")
    @patch("finance_ai.agents.stream_utils.classify_query")
    @patch("finance_ai.agents.stream_utils._get_agent_graph")
    def test_unsupported_intent(
        self,
        mock_get_graph: MagicMock,
        mock_classify: MagicMock,
        mock_general_chat: MagicMock,
    ) -> None:
        """Unsupported intent falls back to general chat."""
        from decimal import Decimal

        from finance_ai.agents.schemas import OrchestratorDecision

        mock_classify.return_value = OrchestratorDecision(
            intent="unknown", confidence=Decimal("0.5")
        )
        mock_get_graph.return_value = None
        mock_general_chat.return_value = {
            "intent": "general_chat",
            "response": "สวัสดีครับ ยินดีให้บริการ",
        }

        events = list(orchestrate_query_stream("สวัสดี"))
        assert any(e.event_type == "token" for e in events)
        assert events[-1].event_type == "complete"
        assert events[-1].intent == "general_chat"
        assert "สวัสดี" in events[-1].content

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

    @patch("finance_ai.agents.stream_utils.execute_general_chat")
    @patch("finance_ai.agents.stream_utils.classify_query")
    @patch("finance_ai.agents.stream_utils._get_agent_graph")
    def test_general_intent_streams_via_general_chat(
        self,
        mock_get_graph: MagicMock,
        mock_classify: MagicMock,
        mock_general_chat: MagicMock,
    ) -> None:
        """General intent streams through general chat (no graph registered)."""
        from decimal import Decimal

        from finance_ai.agents.schemas import OrchestratorDecision

        mock_classify.return_value = OrchestratorDecision(
            intent="general",
            confidence=Decimal("0.8"),
        )
        mock_get_graph.return_value = None
        mock_general_chat.return_value = {"intent": "general_chat", "response": "คำตอบทั่วไป"}

        events = list(orchestrate_query_stream("ดอกเบี้ยทบต้นคืออะไร"))
        assert any(e.event_type == "token" for e in events)
        assert events[-1].event_type == "complete"
        assert events[-1].intent == "general_chat"
        mock_get_graph.assert_called_once_with("general", None)

    @patch("finance_ai.agents.stream_utils.classify_query")
    def test_low_confidence_yields_clarify(self, mock_classify: MagicMock) -> None:
        """Low-confidence stream yields a clarify completion instead of an agent."""
        from decimal import Decimal

        from finance_ai.agents.schemas import OrchestratorDecision

        mock_classify.return_value = OrchestratorDecision(
            intent="expense", confidence=Decimal("0.4")
        )

        events = list(orchestrate_query_stream(query="เงิน", chat_model=MagicMock()))

        assert events[-1].event_type == "complete"
        assert events[-1].intent == "clarify"
        assert "รายจ่าย" in events[-1].content

    @patch("finance_ai.agents.stream_utils.execute_general_chat")
    @patch("finance_ai.agents.stream_utils.classify_query")
    @patch("finance_ai.agents.stream_utils._get_agent_graph")
    def test_classifier_receives_history(
        self,
        mock_get_graph: MagicMock,
        mock_classify: MagicMock,
        mock_general_chat: MagicMock,
    ) -> None:
        """Streaming classifier receives chat history."""
        from decimal import Decimal

        from finance_ai.agents.schemas import OrchestratorDecision

        mock_classify.return_value = OrchestratorDecision(
            intent="unknown", confidence=Decimal("0.1")
        )
        mock_get_graph.return_value = None
        mock_general_chat.return_value = {"intent": "general_chat", "response": "สวัสดีค่ะ"}
        history = [("user", "ก่อนหน้า")]

        list(orchestrate_query_stream("สวัสดี", chat_history=history))

        assert mock_classify.call_args[0][2] == history


class TestExtractLastAiFromMessages:
    """Tests for _extract_last_ai_from_messages."""

    def test_returns_last_ai_content(self) -> None:
        """Returns content from final AIMessage."""
        from langchain_core.messages import AIMessage, HumanMessage

        messages = [
            HumanMessage(content="สรุปค่าใช้จ่าย"),
            AIMessage(content="ค่าใช้จ่ายทั้งหมด 5,000 บาท"),
        ]
        assert _extract_last_ai_from_messages(messages) == "ค่าใช้จ่ายทั้งหมด 5,000 บาท"

    def test_skips_tool_call_ai_message(self) -> None:
        """Skips AIMessage that has tool_calls."""
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        messages = [
            HumanMessage(content="สรุปค่าใช้จ่าย"),
            AIMessage(content="", tool_calls=[{"name": "get_expenses", "args": {}, "id": "1"}]),
            ToolMessage(content="expenses: 5000", tool_call_id="1"),
            AIMessage(content="ค่าใช้จ่ายทั้งหมด 5,000 บาท"),
        ]
        assert _extract_last_ai_from_messages(messages) == "ค่าใช้จ่ายทั้งหมด 5,000 บาท"

    def test_fallback_to_tool_message_when_ai_empty(self) -> None:
        """Falls back to ToolMessage when final AIMessage has empty content."""
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        messages = [
            HumanMessage(content="สรุปค่าใช้จ่าย"),
            AIMessage(content="", tool_calls=[{"name": "get_expenses", "args": {}, "id": "1"}]),
            ToolMessage(content="รายจ่ายเดือนนี้: อาหาร 3,000 บาท", tool_call_id="1"),
            AIMessage(content=""),  # Gemini empty content bug
        ]
        result = _extract_last_ai_from_messages(messages)
        assert result == "รายจ่ายเดือนนี้: อาหาร 3,000 บาท"

    def test_empty_messages_returns_empty(self) -> None:
        """Empty message list returns empty string."""
        assert _extract_last_ai_from_messages([]) == ""

    def test_only_human_message_returns_empty(self) -> None:
        """Only HumanMessage returns empty string."""
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content="hello")]
        assert _extract_last_ai_from_messages(messages) == ""


class TestStreamAgentResponseFallback:
    """Tests for stream_agent_response invoke fallback."""

    def test_fallback_invokes_graph_when_stream_empty(self) -> None:
        """When streaming yields no content, graph.invoke is called."""
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        mock_graph = MagicMock()
        mock_graph.stream.return_value = iter([])  # empty stream
        mock_graph.invoke.return_value = {
            "messages": [
                HumanMessage(content="จ่ายค่าอาหาร"),
                AIMessage(content="", tool_calls=[{"name": "add_expense", "args": {}, "id": "1"}]),
                ToolMessage(content="บันทึกสำเร็จ", tool_call_id="1"),
                AIMessage(content="บันทึกค่าอาหารเรียบร้อยแล้วครับ"),
            ],
        }

        events = list(stream_agent_response(mock_graph, {"messages": []}, "expense"))
        token_events = [e for e in events if e.event_type == "token"]
        assert len(token_events) == 1
        assert "เรียบร้อย" in token_events[0].content
        mock_graph.invoke.assert_called_once()

    def test_fallback_uses_tool_message_when_ai_empty(self) -> None:
        """Fallback invoke returns ToolMessage when final AI is empty."""
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        mock_graph = MagicMock()
        mock_graph.stream.return_value = iter([])
        mock_graph.invoke.return_value = {
            "messages": [
                HumanMessage(content="สรุปค่าใช้จ่าย"),
                AIMessage(content="", tool_calls=[{"name": "get_expenses", "args": {}, "id": "1"}]),
                ToolMessage(content="อาหาร 3,000 บาท", tool_call_id="1"),
                AIMessage(content=""),  # Gemini empty content
            ],
        }

        events = list(stream_agent_response(mock_graph, {"messages": []}, "expense"))
        token_events = [e for e in events if e.event_type == "token"]
        assert len(token_events) == 1
        assert "อาหาร 3,000 บาท" in token_events[0].content

    def test_complete_ai_message_via_messages_mode(self) -> None:
        """Complete AIMessage (non-streaming LLM) is captured as token."""
        from langchain_core.messages import AIMessage

        mock_graph = MagicMock()
        complete_msg = AIMessage(content="บันทึกค่าอาหาร 350 บาท เรียบร้อย")
        mock_graph.stream.return_value = iter(
            [
                (complete_msg, {"langgraph_node": "agent"}),
            ]
        )

        events = list(stream_agent_response(mock_graph, {}, "expense"))
        token_events = [e for e in events if e.event_type == "token"]
        assert len(token_events) == 1
        assert "350 บาท" in token_events[0].content

    def test_no_fallback_when_streaming_has_content(self) -> None:
        """When streaming yields content, no fallback is triggered."""
        from langchain_core.messages import AIMessageChunk

        mock_graph = MagicMock()
        chunk = AIMessageChunk(content="สรุปค่าใช้จ่าย 5,000 บาท")
        mock_graph.stream.return_value = iter(
            [
                (chunk, {"langgraph_node": "agent"}),
            ]
        )

        events = list(stream_agent_response(mock_graph, {}, "expense"))
        token_events = [e for e in events if e.event_type == "token"]
        assert len(token_events) == 1
        assert "5,000 บาท" in token_events[0].content
        mock_graph.invoke.assert_not_called()


class TestInvokeAndExtract:
    """Tests for _invoke_and_extract."""

    def test_extracts_ai_content(self) -> None:
        """Extracts content from final AIMessage."""
        from langchain_core.messages import AIMessage, HumanMessage

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [
                HumanMessage(content="query"),
                AIMessage(content="response text"),
            ],
        }
        assert _invoke_and_extract(mock_graph, {}) == "response text"

    def test_falls_back_to_tool_message(self) -> None:
        """Falls back to ToolMessage when AI is empty."""
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [
                HumanMessage(content="query"),
                AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
                ToolMessage(content="tool result", tool_call_id="1"),
                AIMessage(content=""),
            ],
        }
        assert _invoke_and_extract(mock_graph, {}) == "tool result"

    def test_returns_empty_on_exception(self) -> None:
        """Returns empty string when invoke raises."""
        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = RuntimeError("API error")
        assert _invoke_and_extract(mock_graph, {}) == ""
