"""Shared utilities for LangGraph agent graphs.

Contains reusable functions shared across all specialist agents
to avoid code duplication.
"""

from collections.abc import Sequence
from typing import Any

from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolInvocationError


def should_continue(state: dict[str, Any]) -> str:
    """Determine whether to route to tools or end the graph.

    Checks the last message in the state for tool_calls.
    Returns 'tools' if the LLM requested tool invocation,
    'end' otherwise.

    Args:
        state: Current LangGraph agent state with 'messages' key.

    Returns:
        'tools' if tool_calls exist, 'end' otherwise.

    Example:
        >>> from langchain_core.messages import AIMessage
        >>> should_continue({"messages": [AIMessage(content="hi")]})
        'end'
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


def tool_input_error_message(error: ValueError | ToolInvocationError) -> str:
    """Turn a tool's input error into a message the model can act on.

    Two kinds of bad input are returned to the model instead of crashing
    the answer: arguments that do not match the tool's schema
    (ToolInvocationError, which LangGraph's default handler also caught) and
    values a tool rejects with a clear reason (ValueError, e.g. a zero saving
    or a malformed date). ToolNode reads this annotation and catches only
    these types, so unexpected errors (real bugs) still raise.

    Args:
        error: The input error raised by the tool.

    Returns:
        Error text sent back to the model as the tool result.

    Example:
        >>> message = tool_input_error_message(ValueError("Amount must be positive"))
        >>> message.startswith("Error: Amount must be positive")
        True
    """
    return f"Error: {error}. Fix the arguments and try again, or explain the problem to the user."


def build_tool_node(tools: Sequence[Any]) -> ToolNode:
    """Create a ToolNode that returns tool input errors to the model.

    Args:
        tools: Tools the agent may call.

    Returns:
        ToolNode that answers a ValueError with an error ToolMessage.

    Example:
        >>> node = build_tool_node([])
    """
    return ToolNode(list(tools), handle_tool_errors=tool_input_error_message)
