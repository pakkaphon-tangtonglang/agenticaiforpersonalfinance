"""Tests for streaming a multi-topic reply."""

import json
from decimal import Decimal
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from finance_ai.agents import multi_intent_stream, stream_utils
from finance_ai.agents.multi_intent import FAILED_SECTION_TEXT, WRITE_SUMMARY_HEADER
from finance_ai.agents.multi_intent_stream import stream_multi_intent
from finance_ai.agents.router_agent import RunContext
from finance_ai.agents.schemas import OrchestratorDecision, RoutedTask
from finance_ai.agents.stream_utils import StreamEvent, orchestrate_query_stream

CONTEXT = RunContext(chat_model=None, user_id="user-1", db_session_factory=None, chat_history=None)


def task(intent: str, sub_query: str, confidence: str = "0.9") -> RoutedTask:
    """Build a routed task."""
    return RoutedTask(intent=intent, sub_query=sub_query, confidence=Decimal(confidence))  # type: ignore[arg-type]


def search_reply(call_id: str, *docs: tuple[int, str]) -> ToolMessage:
    """A knowledge-search reply citing documents as (number, key)."""
    citations = [
        {"index": index, "document_key": key, "title": f"doc {key}"} for index, key in docs
    ]
    return ToolMessage(
        content=json.dumps({"citations": citations, "context": "..."}),
        name="search_finance_knowledge",
        tool_call_id=call_id,
    )


class FakeGraph:
    """Streams a fixed list of messages; invoke returns them as final state."""

    def __init__(self, streamed: list[Any], final: list[Any] | None = None) -> None:
        """Keep what stream and invoke return."""
        self.streamed = streamed
        self.final = final if final is not None else streamed
        self.questions: list[str] = []

    def stream(self, state: dict[str, Any], stream_mode: str) -> Any:
        """Yield (message, metadata) pairs like LangGraph."""
        self.questions.append(state["messages"][-1][1])
        for message in self.streamed:
            yield message, {}

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return the final messages."""
        return {"messages": [HumanMessage(content="q"), *self.final]}


def tool_call(call_id: str, name: str, args: dict[str, Any]) -> AIMessage:
    """An AI message requesting one tool call."""
    return AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": args}])


def run(
    graphs: dict[str, FakeGraph], tasks: list[RoutedTask], monkeypatch: pytest.MonkeyPatch
) -> list[StreamEvent]:
    """Stream a multi-topic reply with fake agent graphs."""
    monkeypatch.setattr(
        multi_intent_stream, "_get_agent_graph", lambda intent, model: graphs.get(intent)
    )
    return list(stream_multi_intent("ข้อความ", tasks, CONTEXT))


def text_of(events: list[StreamEvent]) -> str:
    """Join the token events (what the user sees and what is saved)."""
    return "".join(event.content for event in events if event.event_type == "token")


class TestStreamMultiIntent:
    """Sections stream in order under headings and end with one 'complete'."""

    def test_sections_and_complete_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two agents stream two headed sections; complete lists both intents."""
        graphs = {
            "tax": FakeGraph(
                [AIMessageChunk(content="ภาษี "), AIMessageChunk(content="5,000 บาท")]
            ),
            "planning": FakeGraph([AIMessageChunk(content="ออมเดือนละ 3,000")]),
        }
        events = run(
            graphs, [task("tax", "ภาษีเท่าไร"), task("planning", "ออมเท่าไร")], monkeypatch
        )
        assert text_of(events) == "## ภาษี\nภาษี 5,000 บาท\n\n## วางแผนการเงิน\nออมเดือนละ 3,000"
        complete = events[-1]
        assert complete.event_type == "complete"
        assert complete.intent == "tax" and complete.intents == ["tax", "planning"]
        assert complete.content == text_of(events)
        assert [event.event_type for event in events].count("complete") == 1

    def test_later_agent_sees_earlier_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The second agent's question carries the first section's answer."""
        graphs = {
            "tax": FakeGraph([AIMessageChunk(content="ภาษี 5,000")]),
            "planning": FakeGraph([AIMessageChunk(content="ok")]),
        }
        run(graphs, [task("tax", "ภาษี"), task("planning", "ออม")], monkeypatch)
        assert graphs["tax"].questions == ["ภาษี"]
        assert "ภาษี 5,000" in graphs["planning"].questions[0]

    def test_citations_are_unique_across_sections(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both agents cite [1]; the second becomes [2] and sources list both once."""
        graphs = {
            "tax": FakeGraph(
                [
                    tool_call("c1", "search_finance_knowledge", {"query": "ลดหย่อน"}),
                    search_reply("c1", (1, "doc-a")),
                    AIMessageChunk(content="ลดหย่อน 60,000 ["),
                    AIMessageChunk(content="1]"),
                ]
            ),
            "planning": FakeGraph(
                [
                    tool_call("c2", "search_finance_knowledge", {"query": "เงินฉุกเฉิน"}),
                    search_reply("c2", (1, "doc-b")),
                    AIMessageChunk(content="เงินฉุกเฉิน 6 เดือน [1]"),
                ]
            ),
        }
        events = run(graphs, [task("tax", "ลดหย่อน"), task("planning", "ฉุกเฉิน")], monkeypatch)
        text = text_of(events)
        assert "ลดหย่อน 60,000 [1]" in text and "เงินฉุกเฉิน 6 เดือน [2]" in text
        sources = [event for event in events if event.event_type == "sources"]
        assert len(sources) == 1
        assert [(item.index, item.document_key) for item in sources[0].sources] == [
            (1, "doc-a"),
            (2, "doc-b"),
        ]

    def test_writes_are_summarised_at_the_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A write tool call in any section shows up in the closing summary."""
        graphs = {
            "expense": FakeGraph(
                [
                    tool_call("w1", "add_expense", {"amount": "250", "category": "food"}),
                    ToolMessage(content='{"success": true}', name="add_expense", tool_call_id="w1"),
                    AIMessageChunk(content="บันทึกค่าอาหารแล้ว"),
                ]
            ),
            "tax": FakeGraph([AIMessageChunk(content="ภาษี 0")]),
        }
        events = run(graphs, [task("expense", "ค่าอาหาร 250"), task("tax", "ภาษี")], monkeypatch)
        assert text_of(events).endswith(f"{WRITE_SUMMARY_HEADER}\n- รายจ่าย 250 บาท")
        assert any(event.event_type == "tool" for event in events)

    def test_empty_stream_falls_back_to_invoke(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A part whose stream gives no text is re-run without streaming."""
        graphs = {
            "tax": FakeGraph([], final=[AIMessage(content="ภาษีจาก invoke")]),
            "planning": FakeGraph([AIMessageChunk(content="ok")]),
        }
        events = run(graphs, [task("tax", "ภาษี"), task("planning", "ออม")], monkeypatch)
        assert "## ภาษี\nภาษีจาก invoke" in text_of(events)

    def test_empty_part_shows_apology(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A part with no text even after the fallback shows the apology."""
        graphs = {
            "tax": FakeGraph([], final=[]),
            "expense": FakeGraph([AIMessageChunk(content="ค่าสุขภาพ 300 บาท")]),
        }
        events = run(graphs, [task("tax", "ลดหย่อน"), task("expense", "ค่าสุขภาพ")], monkeypatch)
        assert text_of(events).startswith(f"## ภาษี\n{FAILED_SECTION_TEXT}\n\n## รายรับ–รายจ่าย")

    def test_general_part_uses_general_chat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A general part (no agent graph) is answered by general chat."""
        monkeypatch.setattr(
            multi_intent_stream,
            "execute_general_chat",
            lambda *args: {"intent": "general_chat", "response": "ดอกเบี้ยคือ..."},
        )
        graphs = {"tax": FakeGraph([AIMessageChunk(content="ภาษี 0")])}
        events = run(graphs, [task("general", "ดอกเบี้ยคืออะไร"), task("tax", "ภาษี")], monkeypatch)
        assert text_of(events).startswith("## ความรู้การเงินทั่วไป\nดอกเบี้ยคือ...")

    def test_single_confident_part_has_no_heading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With one confident part, the reply has no heading but names the unsure part."""
        graphs = {"tax": FakeGraph([AIMessageChunk(content="ภาษี 0")])}
        events = run(graphs, [task("tax", "ภาษี"), task("report", "อันนั้น", "0.3")], monkeypatch)
        assert text_of(events).startswith("ภาษี 0\n\n")
        assert "- อันนั้น" in text_of(events)
        assert events[-1].intents == ["tax"]

    def test_all_unsure_asks_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No confident part gives the ask-back reply."""
        events = run({}, [task("tax", "?", "0.3"), task("report", "?", "0.2")], monkeypatch)
        assert events[-1].intent == "clarify"


class TestOrchestrateQueryStream:
    """The streaming orchestrator hands multi-topic messages to the multi-intent stream."""

    def test_multi_topic_message_streams_sections(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two tasks from the router produce one combined streamed reply."""
        routed = OrchestratorDecision(
            intent="tax",
            confidence=Decimal("0.9"),
            tasks=[task("tax", "ภาษี"), task("planning", "ออม")],
        )
        monkeypatch.setattr(stream_utils, "classify_query", lambda *args, **kwargs: routed)
        graphs = {
            "tax": FakeGraph([AIMessageChunk(content="A")]),
            "planning": FakeGraph([AIMessageChunk(content="B")]),
        }
        monkeypatch.setattr(
            multi_intent_stream, "_get_agent_graph", lambda intent, model: graphs.get(intent)
        )
        events = list(orchestrate_query_stream("ภาษี แล้วออม", previous_intents=["report", "tax"]))
        assert text_of(events) == "## ภาษี\nA\n\n## วางแผนการเงิน\nB"
        assert events[-1].intents == ["tax", "planning"]

    def test_previous_intents_reach_the_router(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The previous reply's topics are passed to classify_query."""
        seen: dict[str, Any] = {}

        def fake_classify(*args: Any, **kwargs: Any) -> OrchestratorDecision:
            seen.update(kwargs)
            return OrchestratorDecision(intent="general", confidence=Decimal("0.9"))

        monkeypatch.setattr(stream_utils, "classify_query", fake_classify)
        monkeypatch.setattr(
            stream_utils,
            "execute_general_chat",
            lambda *args: {"intent": "general_chat", "response": "hi"},
        )
        list(orchestrate_query_stream("สวัสดี", previous_intents=["tax", "planning"]))
        assert seen["previous_intents"] == ["tax", "planning"]
