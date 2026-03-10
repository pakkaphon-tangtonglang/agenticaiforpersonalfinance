"""Tests for the Report Agent LangGraph graph."""

from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from finance_ai.agents.report_agent import (
    REPORT_AGENT_TOOLS,
    build_report_agent_graph,
    create_llm_node,
    should_continue,
)


class TestShouldContinue:
    """Tests for the should_continue routing function."""

    def test_returns_tools_when_tool_calls_present(self) -> None:
        """When last message has tool_calls, should route to 'tools'."""
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "generate_financial_report_tool",
                    "args": {},
                }
            ],
        )
        state = {
            "messages": [message],
            "report_result": None,
            "user_id": "",
            "db_session_factory": None,
        }
        assert should_continue(state) == "tools"  # type: ignore[arg-type]

    def test_returns_end_when_no_tool_calls(self) -> None:
        """When last message has no tool_calls, should route to 'end'."""
        message = AIMessage(content="รายงานการเงิน: ...")
        state = {
            "messages": [message],
            "report_result": None,
            "user_id": "",
            "db_session_factory": None,
        }
        assert should_continue(state) == "end"  # type: ignore[arg-type]

    def test_returns_end_when_tool_calls_empty(self) -> None:
        """When tool_calls is empty list, should route to 'end'."""
        message = AIMessage(content="done", tool_calls=[])
        state = {
            "messages": [message],
            "report_result": None,
            "user_id": "",
            "db_session_factory": None,
        }
        assert should_continue(state) == "end"  # type: ignore[arg-type]


class TestCreateLlmNode:
    """Tests for the create_llm_node factory function."""

    def test_prepends_system_prompt(self, mock_chat_model: MagicMock) -> None:
        """System prompt should be prepended to messages."""
        mock_chat_model.invoke.return_value = AIMessage(content="response")
        node = create_llm_node(mock_chat_model)
        state: dict[str, Any] = {
            "messages": [("user", "สร้างรายงานการเงิน")],
            "report_result": None,
            "user_id": "",
            "db_session_factory": None,
        }
        node(state)
        call_args = mock_chat_model.invoke.call_args[0][0]
        assert call_args[0].content  # System message is not empty

    def test_returns_messages_dict(self, mock_chat_model: MagicMock) -> None:
        """Node should return dict with 'messages' key."""
        response = AIMessage(content="report result")
        mock_chat_model.invoke.return_value = response
        node = create_llm_node(mock_chat_model)
        state: dict[str, Any] = {
            "messages": [("user", "test")],
            "report_result": None,
            "user_id": "",
            "db_session_factory": None,
        }
        result = node(state)
        assert "messages" in result
        assert result["messages"] == [response]

    def test_binds_report_tools(self, mock_chat_model: MagicMock) -> None:
        """Model should have report tools bound."""
        create_llm_node(mock_chat_model)
        mock_chat_model.bind_tools.assert_called_once_with(REPORT_AGENT_TOOLS)


class TestBuildReportAgentGraph:
    """Tests for build_report_agent_graph function."""

    def test_builds_with_mock_model(self, mock_chat_model: MagicMock) -> None:
        """Should build graph successfully with a mock model."""
        graph = build_report_agent_graph(mock_chat_model)
        assert graph is not None

    def test_has_agent_and_tools_nodes(self, mock_chat_model: MagicMock) -> None:
        """Compiled graph should have 'agent' and 'tools' nodes."""
        graph = build_report_agent_graph(mock_chat_model)
        node_names = list(graph.get_graph().nodes.keys())
        assert "agent" in node_names
        assert "tools" in node_names


class TestReportAgentTools:
    """Tests for the REPORT_AGENT_TOOLS list."""

    def test_has_three_tools(self) -> None:
        """Should contain exactly 3 tools."""
        assert len(REPORT_AGENT_TOOLS) == 3

    def test_contains_expected_tool_names(self) -> None:
        """Should contain report, summary, and RAG tools."""
        names = [t.name for t in REPORT_AGENT_TOOLS]
        assert "generate_financial_report_tool" in names
        assert "get_financial_summary" in names
        assert "search_finance_knowledge" in names
