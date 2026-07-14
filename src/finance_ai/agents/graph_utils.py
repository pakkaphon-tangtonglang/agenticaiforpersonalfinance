"""Shared utilities for LangGraph agent graphs.

Contains reusable functions shared across all specialist agents
to avoid code duplication.
"""

from typing import Any


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
