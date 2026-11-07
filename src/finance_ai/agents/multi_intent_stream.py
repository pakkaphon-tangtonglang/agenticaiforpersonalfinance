"""Streaming reply for a message about several topics.

Each part streams from its own agent, one after another, under a heading.
Citation numbers are made unique across the whole reply, and the reply ends
with any unsure parts and a summary of what was saved.
"""

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

from finance_ai.agents.citation_sources import CitationSource, collect_turn_citations
from finance_ai.agents.multi_intent import (
    FAILED_SECTION_TEXT,
    CitationRenumberer,
    TaskSection,
    build_task_query,
    closing_notes,
    failed_section,
    merge_citations,
    section_heading,
    split_by_confidence,
)
from finance_ai.agents.router_agent import (
    RunContext,
    _build_clarify_response,
    _build_messages,
    execute_general_chat,
)
from finance_ai.agents.schemas import RoutedTask
from finance_ai.agents.stream_utils import (
    StreamEvent,
    _extract_last_ai_from_messages,
    _get_agent_graph,
    _invoke_graph_messages,
    _stream_graph,
    _StreamResult,
)
from finance_ai.agents.tool_trace import ToolTrace, traces_from_messages

MULTI_INTENT_STATUS = "กำลังตอบทีละเรื่อง..."


@dataclass
class _TurnState:
    """What the reply has produced so far."""

    sections: list[TaskSection] = field(default_factory=list)
    citations: list[CitationSource] = field(default_factory=list)
    text: str = ""


@dataclass
class _SectionStream:
    """One part while it streams: its tools, renumbering and starting citations."""

    citations_before: list[CitationSource]
    renumberer: CitationRenumberer = field(default_factory=CitationRenumberer)
    tools: list[ToolTrace] = field(default_factory=list)


def stream_multi_intent(
    query: str, tasks: list[RoutedTask], context: RunContext
) -> Generator[StreamEvent, None, None]:
    """Stream one reply that answers every confident part of a message.

    Args:
        query: The user's whole message.
        tasks: Merged tasks (two or more agents).
        context: Shared request context.

    Yields:
        Status, token, tool and sources events, then one 'complete' event
        whose ``intents`` lists every agent that answered.

    Example:
        >>> list(stream_multi_intent(query, tasks, context))  # doctest: +SKIP
    """
    confident, unsure = split_by_confidence(tasks)
    if not confident:
        yield from _clarify_events()
        return
    yield StreamEvent(event_type="status", content=MULTI_INTENT_STATUS)
    turn = _TurnState()
    for index, task in enumerate(confident):
        if len(confident) > 1:
            yield _token(turn, section_heading(task.intent, index == 0))
        section = yield from _stream_section(task, query, turn, context)
        turn.sections.append(section)
    yield from _finish(turn, unsure)


def _clarify_events() -> Generator[StreamEvent, None, None]:
    """Yield the ask-back reply used when no part is clear enough."""
    clarify = str(_build_clarify_response()["response"])
    yield StreamEvent(event_type="token", content=clarify)
    yield StreamEvent(event_type="complete", content=clarify, intent="clarify")


def _token(turn: _TurnState, text: str) -> StreamEvent:
    """Record text in the reply and wrap it as a token event."""
    turn.text += text
    return StreamEvent(event_type="token", content=text)


def _stream_section(
    task: RoutedTask, query: str, turn: _TurnState, context: RunContext
) -> Generator[StreamEvent, None, TaskSection]:
    """Stream one part's agent and return its finished section.

    Args:
        task: The part to answer.
        query: The user's whole message.
        turn: Reply so far (earlier sections and citations).
        context: Shared request context.

    Yields:
        Token, tool and status events for this part.

    Returns:
        The part's section (answer text and tool calls).
    """
    question = build_task_query(task, query, turn.sections)
    graph = _get_agent_graph(task.intent, context.chat_model)
    if graph is None:
        text = _general_chat_text(question, context)
        yield _token(turn, text)
        return TaskSection(intent=task.intent, text=text)
    input_state = _input_state(question, context)
    stream = _SectionStream(citations_before=list(turn.citations))
    result = _StreamResult()
    start = len(turn.text)
    for event in _stream_graph(graph, input_state, result):
        yield from _relay(event, stream, result, turn)
    tail = stream.renumberer.flush()
    if tail:
        yield _token(turn, tail)
    if not result.text.strip():
        yield from _fallback_invoke(graph, input_state, stream, turn)
    if not turn.text[start:].strip():
        yield _token(turn, FAILED_SECTION_TEXT)
        return failed_section(task.intent, stream.tools)
    return TaskSection(intent=task.intent, text=turn.text[start:], tools=tuple(stream.tools))


def _relay(
    event: StreamEvent, stream: _SectionStream, result: _StreamResult, turn: _TurnState
) -> Generator[StreamEvent, None, None]:
    """Pass one agent event on, renumbering citations in its text."""
    if event.event_type == "tool":
        if event.tool is not None:
            stream.tools.append(event.tool)
        _update_citations(stream, collect_turn_citations(result.tool_messages), turn)
        yield event
        return
    if event.event_type != "token":
        yield event
        return
    text = stream.renumberer.feed(event.content)
    if text:
        yield _token(turn, text)


def _update_citations(
    stream: _SectionStream, section_citations: list[CitationSource], turn: _TurnState
) -> None:
    """Give this part's citations reply-wide numbers."""
    turn.citations, stream.renumberer.mapping = merge_citations(
        stream.citations_before, section_citations
    )


def _fallback_invoke(
    graph: Any, input_state: dict[str, Any], stream: _SectionStream, turn: _TurnState
) -> Generator[StreamEvent, None, None]:
    """Re-run a part without streaming when the stream produced no text."""
    messages = _invoke_graph_messages(graph, input_state)
    _update_citations(stream, collect_turn_citations(messages), turn)
    stream.tools = traces_from_messages(messages)
    text = stream.renumberer.rewrite(_extract_last_ai_from_messages(messages))
    if text:
        yield _token(turn, text)


def _finish(turn: _TurnState, unsure: list[RoutedTask]) -> Generator[StreamEvent, None, None]:
    """Yield the closing notes, the reply's sources and the 'complete' event."""
    notes = closing_notes(turn.sections, unsure)
    if notes:
        yield _token(turn, notes)
    if turn.citations:
        yield StreamEvent(event_type="sources", sources=turn.citations)
    intents = [section.intent for section in turn.sections]
    yield StreamEvent(event_type="complete", content=turn.text, intent=intents[0], intents=intents)


def _general_chat_text(question: str, context: RunContext) -> str:
    """Answer a general part with general chat (not streamed)."""
    result = execute_general_chat(
        question,
        context.chat_model,
        context.user_id,
        context.db_session_factory,
        context.chat_history,
    )
    return str(result["response"])


def _input_state(question: str, context: RunContext) -> dict[str, Any]:
    """Build an agent graph's input for one part."""
    return {
        "messages": _build_messages(question, context.chat_history),
        "user_id": context.user_id,
        "db_session_factory": context.db_session_factory,
    }
