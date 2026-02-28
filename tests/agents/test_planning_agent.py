"""Tests for the LangGraph Planning Agent."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from finance_ai.agents.planning_agent import (
    PLANNING_TOOLS,
    build_planning_agent_graph,
    create_llm_node,
    should_continue,
)
from finance_ai.agents.prompts import PLANNING_AGENT_SYSTEM_PROMPT


class TestShouldContinue:
    """Tests for the conditional edge function."""

    def test_returns_tools_when_tool_calls_present(
        self,
        planning_tool_call_message: AIMessage,
    ) -> None:
        """Routes to 'tools' when last message has tool_calls."""
        state = {
            "messages": [planning_tool_call_message],
            "planning_result": None,
            "user_id": "test-user",
            "db_session_factory": None,
        }
        assert should_continue(state) == "tools"  # type: ignore[arg-type]

    def test_returns_end_when_no_tool_calls(
        self,
        planning_formatted_response: AIMessage,
    ) -> None:
        """Routes to 'end' when last message has no tool_calls."""
        state = {
            "messages": [planning_formatted_response],
            "planning_result": None,
            "user_id": "test-user",
            "db_session_factory": None,
        }
        assert should_continue(state) == "end"  # type: ignore[arg-type]

    def test_returns_end_for_empty_tool_calls(self) -> None:
        """Routes to 'end' when tool_calls is an empty list."""
        message = AIMessage(content="response", tool_calls=[])
        state = {
            "messages": [message],
            "planning_result": None,
            "user_id": "test-user",
            "db_session_factory": None,
        }
        assert should_continue(state) == "end"  # type: ignore[arg-type]


class TestCreateLlmNode:
    """Tests for the LLM node factory."""

    def test_prepends_system_prompt(self, mock_chat_model: MagicMock) -> None:
        """System prompt is prepended to messages sent to the LLM."""
        mock_chat_model.invoke.return_value = AIMessage(content="response")
        node = create_llm_node(mock_chat_model)

        state = {
            "messages": [HumanMessage(content="อยากออมเงิน 100,000 บาท")],
            "planning_result": None,
            "user_id": "test-user",
            "db_session_factory": None,
        }
        node(state)

        call_args = mock_chat_model.invoke.call_args[0][0]
        assert call_args[0].content == PLANNING_AGENT_SYSTEM_PROMPT
        assert call_args[1].content == "อยากออมเงิน 100,000 บาท"

    def test_returns_message_list(self, mock_chat_model: MagicMock) -> None:
        """Returns dict with messages list containing model response."""
        expected_response = AIMessage(content="สร้างเป้าหมายให้แล้ว")
        mock_chat_model.invoke.return_value = expected_response
        node = create_llm_node(mock_chat_model)

        state = {
            "messages": [HumanMessage(content="test")],
            "planning_result": None,
            "user_id": "test-user",
            "db_session_factory": None,
        }
        result = node(state)

        assert "messages" in result
        assert result["messages"] == [expected_response]

    def test_binds_tools_to_model(self, mock_chat_model: MagicMock) -> None:
        """Verifies tools are bound to the chat model."""
        create_llm_node(mock_chat_model)
        mock_chat_model.bind_tools.assert_called_once_with(PLANNING_TOOLS)


class TestBuildPlanningAgentGraph:
    """Tests for the graph builder."""

    def test_builds_graph_with_mock_model(self, mock_chat_model: MagicMock) -> None:
        """Builds a compiled graph when given a mock model."""
        graph = build_planning_agent_graph(mock_chat_model)
        assert graph is not None

    def test_graph_has_agent_and_tools_nodes(self, mock_chat_model: MagicMock) -> None:
        """Graph contains 'agent' and 'tools' nodes."""
        graph = build_planning_agent_graph(mock_chat_model)
        node_names = list(graph.get_graph().nodes.keys())
        assert "agent" in node_names
        assert "tools" in node_names

    def test_planning_tools_list_complete(self) -> None:
        """PLANNING_TOOLS contains expected tools."""
        tool_names = [t.name for t in PLANNING_TOOLS]
        assert "create_financial_goal" in tool_names
        assert "view_financial_goals" in tool_names
        assert "update_goal_progress" in tool_names
        assert "calculate_saving_plan" in tool_names
        assert "search_finance_knowledge" in tool_names
