"""Streaming utilities for LangGraph agent execution.

Provides a generator-based API for streaming agent responses,
yielding status updates during tool calls and token-level
chunks during the final LLM response.
"""

from collections.abc import Callable, Generator
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from finance_ai.agents.router_agent import (
    _build_messages,
    build_unsupported_response,
    classify_query,
)
from finance_ai.core.logging import get_logger

logger = get_logger(__name__)


class StreamEvent(BaseModel):
    """A single event in the agent response stream.

    Attributes:
        event_type: Type of event (status, token, or complete).
        content: Text content of the event.
        intent: The classified intent (set on 'complete' event).

    Example:
        >>> e = StreamEvent(event_type="token", content="สวัสดี")
    """

    event_type: Literal["status", "token", "complete"] = "token"
    content: str = ""
    intent: str = Field(default="")


def stream_agent_response(
    graph: CompiledStateGraph,  # type: ignore[type-arg]
    input_state: dict[str, Any],
    intent: str = "unknown",
) -> Generator[StreamEvent, None, None]:
    """Stream responses from a LangGraph agent.

    Uses stream_mode='messages' to get token-level LLM output.
    Tool calls emit status events; final LLM output emits tokens.

    Args:
        graph: Compiled LangGraph state graph.
        input_state: Initial state dict with messages, user_id, etc.
        intent: The classified intent for the complete event.

    Yields:
        StreamEvent objects (status, token, or complete).
    """
    yield StreamEvent(event_type="status", content="กำลังประมวลผล...")

    full_response = ""
    try:
        for chunk, _metadata in graph.stream(input_state, stream_mode="messages"):
            event = _process_chunk(chunk)
            if event is not None:
                if event.event_type == "token":
                    full_response += event.content
                yield event
    except Exception as exc:  # noqa: BLE001
        logger.error("Stream error: %s", exc)
        full_response = f"เกิดข้อผิดพลาด: {exc}"
        yield StreamEvent(event_type="token", content=full_response)

    yield StreamEvent(
        event_type="complete",
        content=full_response,
        intent=intent,
    )


def _process_chunk(chunk: Any) -> StreamEvent | None:
    """Process a single stream chunk into a StreamEvent.

    Args:
        chunk: A message chunk from LangGraph stream.

    Returns:
        StreamEvent or None if chunk should be skipped.
    """
    if isinstance(chunk, AIMessageChunk):
        return _handle_ai_chunk(chunk)
    return None


def _handle_ai_chunk(chunk: AIMessageChunk) -> StreamEvent | None:
    """Handle an AI message chunk.

    Args:
        chunk: AI message chunk from LLM.

    Returns:
        Token event for content, status for tool calls, None otherwise.
    """
    if chunk.tool_call_chunks:
        name = _extract_tool_name(chunk)
        if name:
            return StreamEvent(
                event_type="status",
                content=f"กำลังใช้เครื่องมือ: {name}",
            )
        return None

    if chunk.content:
        return StreamEvent(
            event_type="token",
            content=str(chunk.content),
        )
    return None


def _extract_tool_name(chunk: AIMessageChunk) -> str:
    """Extract tool name from a tool call chunk.

    Args:
        chunk: AI message chunk with tool call info.

    Returns:
        Tool name string, or empty string.
    """
    for tool_call in chunk.tool_call_chunks:
        name = tool_call.get("name", "")
        if name:
            return str(name)
    return ""


def route_query_stream(
    query: str,
    chat_model: BaseChatModel | None = None,
    user_id: str = "",
    db_session_factory: Callable[[], Session] | None = None,
    chat_history: list[tuple[str, str]] | None = None,
) -> Generator[StreamEvent, None, None]:
    """Route and stream a user query response.

    Classifies intent (non-streaming), then streams the agent.

    Args:
        query: The user's natural language query.
        chat_model: Optional ChatModel override.
        user_id: UUID of the user.
        db_session_factory: Session factory for DB access.
        chat_history: Previous messages for context.

    Yields:
        StreamEvent objects.
    """
    decision = classify_query(query, chat_model)
    intent = decision.intent
    logger.info("Stream routing to: %s", intent)

    graph = _get_agent_graph(intent, chat_model)
    if graph is None:
        result = build_unsupported_response(decision)
        yield StreamEvent(event_type="token", content=result["response"])
        yield StreamEvent(
            event_type="complete",
            content=result["response"],
            intent=intent,
        )
        return

    input_state = {
        "messages": _build_messages(query, chat_history),
        "user_id": user_id,
        "db_session_factory": db_session_factory,
    }
    yield from stream_agent_response(graph, input_state, intent)


def _get_agent_graph(
    intent: str,
    chat_model: BaseChatModel | None = None,
) -> CompiledStateGraph | None:  # type: ignore[type-arg]
    """Build and return the agent graph for a given intent.

    Args:
        intent: Classified intent string.
        chat_model: Optional ChatModel override.

    Returns:
        Compiled graph or None for unsupported intents.
    """
    builder_map = _get_builder_map()
    builder_fn = builder_map.get(intent)
    if builder_fn is None:
        return None
    return builder_fn(chat_model)


def _get_builder_map() -> dict[str, Callable[..., CompiledStateGraph]]:  # type: ignore[type-arg]
    """Return mapping of intent to graph builder function.

    Returns:
        Dict mapping intent strings to builder callables.
    """
    from finance_ai.agents.expense_agent import build_expense_agent_graph  # noqa: PLC0415
    from finance_ai.agents.investment_agent import build_investment_agent_graph  # noqa: PLC0415
    from finance_ai.agents.planning_agent import build_planning_agent_graph  # noqa: PLC0415
    from finance_ai.agents.recommendation_agent import (
        build_recommendation_agent_graph,
    )  # noqa: PLC0415, E501
    from finance_ai.agents.report_agent import build_report_agent_graph  # noqa: PLC0415
    from finance_ai.agents.tax_agent import build_tax_agent_graph  # noqa: PLC0415

    return {
        "tax": build_tax_agent_graph,
        "expense": build_expense_agent_graph,
        "investment": build_investment_agent_graph,
        "planning": build_planning_agent_graph,
        "recommendation": build_recommendation_agent_graph,
        "report": build_report_agent_graph,
    }
