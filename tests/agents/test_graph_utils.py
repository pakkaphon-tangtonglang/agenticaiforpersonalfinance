"""Tests for shared graph utilities."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, MessagesState, StateGraph

from finance_ai.agents.graph_utils import build_tool_node, should_continue


def _compile(graph: StateGraph[MessagesState]) -> Any:
    """Compiled graph typed loosely (LangGraph's invoke overloads reject plain dicts)."""
    return graph.compile()


class TestShouldContinue:
    """Tests for the shared should_continue function."""

    def test_returns_tools_when_tool_calls_exist(self) -> None:
        """Return 'tools' when last message has tool_calls."""
        msg = AIMessage(content="", tool_calls=[{"id": "c1", "name": "tool", "args": {}}])
        state: dict[str, Any] = {"messages": [msg]}
        assert should_continue(state) == "tools"

    def test_returns_end_when_no_tool_calls(self) -> None:
        """Return 'end' when last message has no tool_calls."""
        msg = AIMessage(content="Hello")
        state: dict[str, Any] = {"messages": [msg]}
        assert should_continue(state) == "end"

    def test_returns_end_when_tool_calls_empty(self) -> None:
        """Return 'end' when tool_calls list is empty."""
        msg = AIMessage(content="", tool_calls=[])
        state: dict[str, Any] = {"messages": [msg]}
        assert should_continue(state) == "end"

    def test_returns_end_when_no_tool_calls_attr(self) -> None:
        """Return 'end' when message lacks tool_calls attribute."""
        msg = MagicMock(spec=[])
        state: dict[str, Any] = {"messages": [msg]}
        assert should_continue(state) == "end"


class TestBuildToolNode:
    """build_tool_node turns tool input errors into messages the model can read."""

    @staticmethod
    def _call(tool_name: str) -> dict[str, Any]:
        """Run the tool node on one AI message that calls the named tool."""
        message = AIMessage(content="", tool_calls=[{"name": tool_name, "args": {}, "id": "c1"}])
        graph = StateGraph(MessagesState)
        graph.add_node("tools", build_tool_node([_rejects_input, _breaks]))
        graph.set_entry_point("tools")
        graph.add_edge("tools", END)
        state = _compile(graph).invoke({"messages": [message]})
        return {"messages": state["messages"][1:]}

    def test_value_error_becomes_error_tool_message(self) -> None:
        """A ValueError is returned to the model instead of crashing the agent."""
        result = self._call("_rejects_input")
        tool_message = result["messages"][0]
        assert isinstance(tool_message, ToolMessage)
        assert tool_message.status == "error"
        assert "Monthly saving must be positive" in tool_message.text

    def test_wrong_arguments_become_error_tool_message(self) -> None:
        """Arguments that do not match the tool schema go back to the model too.

        LangGraph's default handler did this; the custom handler must keep it.
        """
        message = AIMessage(
            content="",
            tool_calls=[{"name": "_needs_symbol", "args": {"action": "add"}, "id": "c2"}],
        )
        graph = StateGraph(MessagesState)
        graph.add_node("tools", build_tool_node([_needs_symbol]))
        graph.set_entry_point("tools")
        graph.add_edge("tools", END)
        tool_message = _compile(graph).invoke({"messages": [message]})["messages"][-1]
        assert isinstance(tool_message, ToolMessage) and tool_message.status == "error"
        assert "symbol" in tool_message.text

    def test_other_errors_still_raise(self) -> None:
        """Unexpected errors (real bugs) are not hidden."""
        with pytest.raises(RuntimeError, match="database down"):
            self._call("_breaks")


@tool
def _rejects_input() -> str:
    """Raise the kind of input error our tools raise."""
    raise ValueError("Monthly saving must be positive. Received: 0 THB")


@tool
def _needs_symbol(symbol: str) -> str:
    """Require a symbol argument."""
    return symbol


@tool
def _breaks() -> str:
    """Raise an unexpected error."""
    raise RuntimeError("database down")
