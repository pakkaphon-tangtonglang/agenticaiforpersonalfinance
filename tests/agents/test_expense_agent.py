"""Tests for the LangGraph Expense Agent."""

from collections.abc import Callable
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy.orm import Session

from finance_ai.agents.expense_agent import (
    EXPENSE_TOOLS,
    build_expense_agent_graph,
    create_llm_node,
    should_continue,
)
from finance_ai.agents.prompts import EXPENSE_AGENT_SYSTEM_PROMPT
from finance_ai.database.models.user import User


class TestShouldContinue:
    """Tests for the conditional edge function."""

    def test_returns_tools_when_tool_calls_present(
        self,
        expense_tool_call_message: AIMessage,
    ) -> None:
        """Routes to 'tools' when last message has tool_calls."""
        state = {
            "messages": [expense_tool_call_message],
            "expense_result": None,
            "user_id": "test-user",
            "db_session_factory": None,
        }
        assert should_continue(state) == "tools"  # type: ignore[arg-type]

    def test_returns_end_when_no_tool_calls(
        self,
        expense_formatted_response: AIMessage,
    ) -> None:
        """Routes to 'end' when last message has no tool_calls."""
        state = {
            "messages": [expense_formatted_response],
            "expense_result": None,
            "user_id": "test-user",
            "db_session_factory": None,
        }
        assert should_continue(state) == "end"  # type: ignore[arg-type]

    def test_returns_end_for_empty_tool_calls(self) -> None:
        """Routes to 'end' when tool_calls is an empty list."""
        message = AIMessage(content="response", tool_calls=[])
        state = {
            "messages": [message],
            "expense_result": None,
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
            "messages": [HumanMessage(content="จ่ายค่ากาแฟ 80 บาท")],
            "expense_result": None,
            "user_id": "test-user",
            "db_session_factory": None,
        }
        node(state)

        call_args = mock_chat_model.invoke.call_args[0][0]
        assert EXPENSE_AGENT_SYSTEM_PROMPT in call_args[0].content
        assert call_args[1].content == "จ่ายค่ากาแฟ 80 บาท"

    def test_returns_message_list(self, mock_chat_model: MagicMock) -> None:
        """Node returns dict with messages list."""
        response = AIMessage(content="บันทึกแล้ว")
        mock_chat_model.invoke.return_value = response
        node = create_llm_node(mock_chat_model)

        result = node(
            {
                "messages": [HumanMessage(content="test")],
                "expense_result": None,
                "user_id": "test-user",
                "db_session_factory": None,
            }
        )
        assert result == {"messages": [response]}

    def test_binds_tools_to_model(self, mock_chat_model: MagicMock) -> None:
        """LLM node binds expense tools to the model."""
        create_llm_node(mock_chat_model)
        mock_chat_model.bind_tools.assert_called_once_with(EXPENSE_TOOLS)


class TestBuildExpenseAgentGraph:
    """Tests for building the Expense Agent LangGraph."""

    def test_returns_compiled_graph(self, mock_chat_model: MagicMock) -> None:
        """build_expense_agent_graph returns a compiled graph."""
        graph = build_expense_agent_graph(chat_model=mock_chat_model)
        assert graph is not None
        assert hasattr(graph, "invoke")

    def test_accepts_custom_model(self, mock_chat_model: MagicMock) -> None:
        """Custom chat model is used instead of factory."""
        graph = build_expense_agent_graph(chat_model=mock_chat_model)
        assert graph is not None
        mock_chat_model.bind_tools.assert_called_once()

    def test_full_react_loop(
        self,
        mock_chat_model: MagicMock,
        expense_tool_call_message: AIMessage,
        expense_formatted_response: AIMessage,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Full ReAct loop: LLM calls tool, tool executes, LLM formats response."""
        mock_chat_model.invoke.side_effect = [
            expense_tool_call_message,
            expense_formatted_response,
        ]

        graph = build_expense_agent_graph(chat_model=mock_chat_model)
        result = graph.invoke(
            {
                "messages": [("user", "จ่ายค่ากาแฟ 80 บาท")],
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )

        last_message = result["messages"][-1]
        assert "บันทึกรายจ่าย" in last_message.content
        assert mock_chat_model.invoke.call_count == 2
