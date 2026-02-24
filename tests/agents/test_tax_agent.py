"""Tests for the LangGraph Tax Agent."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage  # type: ignore[import-untyped]

from finance_ai.agents.prompts import TAX_AGENT_SYSTEM_PROMPT
from finance_ai.agents.tax_agent import (
    TAX_TOOLS,
    build_tax_agent_graph,
    create_llm_node,
    should_continue,
)


class TestShouldContinue:
    """Tests for the conditional edge function."""

    def test_returns_tools_when_tool_calls_present(
        self,
        tax_tool_call_message: AIMessage,
    ) -> None:
        """Routes to 'tools' when last message has tool_calls."""
        state = {"messages": [tax_tool_call_message], "tax_result": None}
        assert should_continue(state) == "tools"

    def test_returns_end_when_no_tool_calls(
        self,
        tax_formatted_response: AIMessage,
    ) -> None:
        """Routes to 'end' when last message has no tool_calls."""
        state = {"messages": [tax_formatted_response], "tax_result": None}
        assert should_continue(state) == "end"

    def test_returns_end_for_empty_tool_calls(self) -> None:
        """Routes to 'end' when tool_calls is an empty list."""
        message = AIMessage(content="response", tool_calls=[])
        state = {"messages": [message], "tax_result": None}
        assert should_continue(state) == "end"


class TestCreateLlmNode:
    """Tests for the LLM node factory."""

    def test_prepends_system_prompt(self, mock_chat_model: MagicMock) -> None:
        """System prompt is prepended to messages sent to the LLM."""
        mock_chat_model.invoke.return_value = AIMessage(content="response")
        node = create_llm_node(mock_chat_model)

        state = {
            "messages": [HumanMessage(content="คำนวณภาษี")],
            "tax_result": None,
        }
        node(state)

        call_args = mock_chat_model.invoke.call_args[0][0]
        assert call_args[0].content == TAX_AGENT_SYSTEM_PROMPT
        assert call_args[1].content == "คำนวณภาษี"

    def test_returns_message_list(self, mock_chat_model: MagicMock) -> None:
        """Node returns dict with messages list."""
        response = AIMessage(content="ผลลัพธ์")
        mock_chat_model.invoke.return_value = response
        node = create_llm_node(mock_chat_model)

        result = node({"messages": [HumanMessage(content="test")], "tax_result": None})
        assert result == {"messages": [response]}

    def test_binds_tools_to_model(self, mock_chat_model: MagicMock) -> None:
        """LLM node binds tax tools to the model."""
        create_llm_node(mock_chat_model)
        mock_chat_model.bind_tools.assert_called_once_with(TAX_TOOLS)


class TestBuildTaxAgentGraph:
    """Tests for building the Tax Agent LangGraph."""

    def test_returns_compiled_graph(self, mock_chat_model: MagicMock) -> None:
        """build_tax_agent_graph returns a compiled graph."""
        graph = build_tax_agent_graph(chat_model=mock_chat_model)
        assert graph is not None
        assert hasattr(graph, "invoke")

    def test_accepts_custom_model(self, mock_chat_model: MagicMock) -> None:
        """Custom chat model is used instead of factory."""
        graph = build_tax_agent_graph(chat_model=mock_chat_model)
        assert graph is not None
        mock_chat_model.bind_tools.assert_called_once()

    def test_full_react_loop(
        self,
        mock_chat_model: MagicMock,
        tax_tool_call_message: AIMessage,
        tax_formatted_response: AIMessage,
    ) -> None:
        """Full ReAct loop: LLM calls tool, tool executes, LLM formats response."""
        mock_chat_model.invoke.side_effect = [
            tax_tool_call_message,
            tax_formatted_response,
        ]

        graph = build_tax_agent_graph(chat_model=mock_chat_model)
        result = graph.invoke({"messages": [("user", "คำนวณภาษี เงินเดือน 1.2 ล้าน")]})

        last_message = result["messages"][-1]
        assert "ผลการคำนวณภาษี" in last_message.content
        assert mock_chat_model.invoke.call_count == 2
