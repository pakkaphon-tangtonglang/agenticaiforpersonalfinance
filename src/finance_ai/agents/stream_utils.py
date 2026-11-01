"""Streaming utilities for LangGraph agent execution.

Provides a generator-based API for streaming agent responses,
yielding status updates during tool calls and token-level
chunks during the final LLM response.
"""

from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from finance_ai.agents.citation_sources import CitationSource, collect_turn_citations
from finance_ai.agents.graph_cache import get_compiled_graph
from finance_ai.agents.llm_error_messages import describe_llm_error
from finance_ai.agents.router_agent import (
    _build_clarify_response,
    _build_messages,
    _should_clarify,
    classify_query,
    execute_general_chat,
)
from finance_ai.core.logging import get_logger

logger = get_logger(__name__)


class StreamEvent(BaseModel):
    """A single event in the agent response stream.

    Attributes:
        event_type: Type of event (status, token, sources, or complete).
        content: Text content of the event.
        intent: The classified intent (set on 'complete' event).
        sources: Numbered citation sources (set on 'sources' event).

    Example:
        >>> e = StreamEvent(event_type="token", content="สวัสดี")
    """

    event_type: Literal["status", "token", "sources", "complete"] = "token"
    content: str = ""
    intent: str = Field(default="")
    sources: list[CitationSource] = Field(default_factory=list)


@dataclass
class _StreamResult:
    """What one streamed graph run produced: answer text and tool messages."""

    text: str = ""
    tool_messages: list[ToolMessage] = field(default_factory=list)


def stream_agent_response(
    graph: CompiledStateGraph,  # type: ignore[type-arg]
    input_state: dict[str, Any],
    intent: str = "unknown",
) -> Generator[StreamEvent, None, None]:
    """Stream responses from a LangGraph agent.

    Streams with stream_mode='messages' for token-level output.
    If streaming produces no content (Gemini intermittent empty
    response), falls back to graph.invoke() with fresh state.
    Sources cited by search tools are sent once, before 'complete'.

    Args:
        graph: Compiled LangGraph state graph.
        input_state: Initial state dict with messages, user_id, etc.
        intent: The classified intent for the complete event.

    Yields:
        StreamEvent objects (status, token, sources, or complete).
    """
    yield StreamEvent(event_type="status", content="กำลังประมวลผล...")
    original_messages = list(input_state.get("messages", []))
    result = _StreamResult()
    yield from _stream_graph(graph, input_state, result)
    citations = collect_turn_citations(result.tool_messages)
    if not result.text.strip():
        fallback_state = {**input_state, "messages": original_messages}
        final_messages = _invoke_graph_messages(graph, fallback_state)
        result.text = _extract_last_ai_from_messages(final_messages)
        citations = collect_turn_citations(final_messages)
        if result.text:
            yield StreamEvent(event_type="token", content=result.text)
    if citations:
        yield StreamEvent(event_type="sources", sources=citations)
    yield StreamEvent(event_type="complete", content=result.text, intent=intent)


def _stream_graph(
    graph: CompiledStateGraph,  # type: ignore[type-arg]
    input_state: dict[str, Any],
    result: _StreamResult,
) -> Generator[StreamEvent, None, None]:
    """Stream one graph run, recording the answer text and tool messages.

    An LLM failure becomes a user-facing Thai message instead of an error.

    Args:
        graph: Compiled LangGraph state graph.
        input_state: Initial state dict.
        result: Accumulator filled with text and tool messages.

    Yields:
        Status and token events.
    """
    try:
        for chunk, _metadata in graph.stream(input_state, stream_mode="messages"):
            if isinstance(chunk, ToolMessage):
                result.tool_messages.append(chunk)
            event = _process_chunk(chunk)
            if event is not None:
                if event.event_type == "token":
                    result.text += event.content
                yield event
    except Exception as exc:  # noqa: BLE001
        logger.error("Stream error: %s", exc, exc_info=True)
        result.text = describe_llm_error(exc)
        yield StreamEvent(event_type="token", content=result.text)


def _invoke_graph_messages(
    graph: CompiledStateGraph,  # type: ignore[type-arg]
    input_state: dict[str, Any],
) -> list[Any]:
    """Invoke the graph and return its final messages ([] on failure).

    Args:
        graph: Compiled LangGraph state graph.
        input_state: Fresh input state (not mutated by prior stream).

    Returns:
        Final message list, or [] if the invoke failed.
    """
    try:
        return list(graph.invoke(input_state).get("messages", []))
    except Exception:  # noqa: BLE001
        logger.warning("Fallback invoke failed", exc_info=True)
    return []


def _invoke_and_extract(
    graph: CompiledStateGraph,  # type: ignore[type-arg]
    input_state: dict[str, Any],
) -> str:
    """Invoke graph and extract the response content.

    Used as fallback when streaming produces no content.

    Args:
        graph: Compiled LangGraph state graph.
        input_state: Fresh input state (not mutated by prior stream).

    Returns:
        Response content string, or empty string if none found.
    """
    return _extract_last_ai_from_messages(_invoke_graph_messages(graph, input_state))


def _process_chunk(chunk: Any) -> StreamEvent | None:
    """Process a single stream chunk into a StreamEvent.

    Handles both AIMessageChunk (token-level streaming) and
    complete AIMessage (emitted by on_chain_end when the LLM
    uses invoke() internally instead of streaming).

    Args:
        chunk: A message chunk from LangGraph stream.

    Returns:
        StreamEvent or None if chunk should be skipped.
    """
    # Check AIMessageChunk first (it inherits from AIMessage)
    if isinstance(chunk, AIMessageChunk):
        return _handle_ai_chunk(chunk)
    # Handle complete AIMessage from node output (non-streaming LLM)
    if isinstance(chunk, AIMessage):
        return _handle_complete_ai_message(chunk)
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
            content=chunk.text,
        )
    return None


def _handle_complete_ai_message(message: AIMessage) -> StreamEvent | None:
    """Handle a complete AIMessage from node output.

    When the LLM uses invoke() internally (not streaming),
    LangGraph emits the complete AIMessage via on_chain_end.

    Args:
        message: Complete AI message from node output.

    Returns:
        Token event for content, status for tool calls, None otherwise.
    """
    if message.tool_calls:
        names = [tc.get("name", "") for tc in message.tool_calls]
        name = next((n for n in names if n), "")
        if name:
            return StreamEvent(
                event_type="status",
                content=f"กำลังใช้เครื่องมือ: {name}",
            )
        return None

    if message.content:
        return StreamEvent(
            event_type="token",
            content=message.text,
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


def _extract_last_ai_from_messages(messages: list[Any]) -> str:
    """Extract the last AI response content from graph messages.

    Walks backwards through messages, skipping ToolMessages,
    looking for the final AIMessage with text content. If the
    final AIMessage has empty content (common Gemini bug after
    tool calls), falls back to the last ToolMessage content.

    Args:
        messages: List of LangChain message objects.

    Returns:
        Content string, or empty string if none found.
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            continue
        if isinstance(msg, AIMessage) and msg.content:
            if not msg.tool_calls:
                return msg.text
        break

    # Fallback: use last ToolMessage content when AI returned empty
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.content:
            return msg.text
        if isinstance(msg, AIMessage) and msg.tool_calls:
            break
    return ""


def orchestrate_query_stream(
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
    try:
        decision = classify_query(query, chat_model, chat_history)
    except Exception as exc:  # noqa: BLE001
        logger.error("Stream routing failed: %s", exc, exc_info=True)
        yield from _stream_error_response(exc)
        return
    logger.info("Stream routing to: %s", decision.intent)
    if _should_clarify(decision):
        yield from _stream_clarify_response()
        return
    graph = _get_agent_graph(decision.intent, chat_model)
    if graph is None:
        yield from _stream_general_chat(
            query, chat_model, user_id, db_session_factory, chat_history
        )
        return
    input_state = {
        "messages": _build_messages(query, chat_history),
        "user_id": user_id,
        "db_session_factory": db_session_factory,
    }
    yield from stream_agent_response(graph, input_state, decision.intent)


def _stream_error_response(error: Exception) -> Generator[StreamEvent, None, None]:
    """Yield a user-facing error reply so the SSE stream ends cleanly.

    Args:
        error: The exception that stopped the request.

    Yields:
        Token + complete events with intent='error'.
    """
    message = describe_llm_error(error)
    yield StreamEvent(event_type="token", content=message)
    yield StreamEvent(event_type="complete", content=message, intent="error")


def _stream_clarify_response() -> Generator[StreamEvent, None, None]:
    """Yield a clarify-back exchange for low-confidence routing.

    Yields:
        Token + complete events with intent='clarify'.
    """
    clarify = _build_clarify_response()
    yield StreamEvent(event_type="token", content=clarify["response"])
    yield StreamEvent(event_type="complete", content=clarify["response"], intent="clarify")


def _stream_general_chat(
    query: str,
    chat_model: BaseChatModel | None,
    user_id: str,
    db_session_factory: Callable[[], Session] | None,
    chat_history: list[tuple[str, str]] | None,
) -> Generator[StreamEvent, None, None]:
    """Execute general chat and emit it as non-streamed events.

    Yields:
        Token + complete events with intent='general_chat'.
    """
    result = execute_general_chat(query, chat_model, user_id, db_session_factory, chat_history)
    response_text = str(result["response"])
    yield StreamEvent(event_type="token", content=response_text)
    yield StreamEvent(event_type="complete", content=response_text, intent="general_chat")


def _get_agent_graph(
    intent: str,
    chat_model: BaseChatModel | None = None,
) -> CompiledStateGraph | None:  # type: ignore[type-arg]
    """Get the cached agent graph for a given intent.

    Args:
        intent: Classified intent string.
        chat_model: Optional ChatModel override.

    Returns:
        Compiled graph or None for unsupported intents.
    """
    if chat_model is None:
        from finance_ai.agents.llm_factory import create_chat_model  # noqa: PLC0415

        chat_model = create_chat_model()
    return get_compiled_graph(intent, chat_model)
