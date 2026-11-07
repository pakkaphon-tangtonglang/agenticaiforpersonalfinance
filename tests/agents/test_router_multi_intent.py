"""Tests for multi-intent routing in the production router."""

from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from finance_ai.agents import router_agent
from finance_ai.agents.multi_intent import UNSURE_HEADER, WRITE_SUMMARY_HEADER
from finance_ai.agents.prompts import (
    ORCHESTRATOR_EXAMPLES_SECTION,
    ORCHESTRATOR_MULTI_INTENT_SECTION,
    build_orchestrator_system_prompt,
)
from finance_ai.agents.router_agent import (
    RouterAblationConfig,
    classify_query,
    orchestrate_query,
    parse_orchestrator_response,
    parse_routed_tasks,
    previous_topics_hint,
)
from finance_ai.agents.schemas import OrchestratorDecision, RoutedTask
from finance_ai.agents.tool_trace import ToolTrace

NO_HINT = RouterAblationConfig(resolve_asset_hint=False)


def decision(*tasks: tuple[str, str, str]) -> OrchestratorDecision:
    """A router decision whose first task is also the outer intent."""
    routed = [
        RoutedTask(intent=intent, sub_query=query, confidence=Decimal(conf))  # type: ignore[arg-type]
        for intent, query, conf in tasks
    ]
    return OrchestratorDecision(
        intent=routed[0].intent, confidence=routed[0].confidence, tasks=routed
    )


@pytest.fixture
def agents(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Replace every agent with a fake that records the question it got."""
    asked: dict[str, list[str]] = {}

    def make(intent: str) -> Any:
        def run(query: str, *args: Any) -> dict[str, Any]:
            asked.setdefault(intent, []).append(query)
            tools = (
                [ToolTrace(name="add_expense", args={"amount": "250"})]
                if intent == "expense"
                else []
            )
            return {"intent": intent, "response": f"คำตอบ {intent}", "tools": tools}

        return run

    agent_map = {name: make(name) for name in ("tax", "expense", "planning", "report", "general")}
    monkeypatch.setattr(router_agent, "_build_agent_map", lambda: agent_map)
    return asked


def route_with(monkeypatch: pytest.MonkeyPatch, routed: OrchestratorDecision) -> None:
    """Make classify_query return a fixed decision."""
    monkeypatch.setattr(router_agent, "classify_query", lambda *args, **kwargs: routed)


class TestParseTasks:
    """The router JSON may carry a tasks list; each part is validated alone."""

    def test_tasks_are_parsed(self) -> None:
        """A multi-topic reply keeps its parts in order."""
        content = (
            '{"intent": "tax", "confidence": 0.9, "tasks": ['
            '{"intent": "tax", "sub_query": "ภาษี", "confidence": 0.9},'
            '{"intent": "planning", "sub_query": "ออม", "confidence": 0.8}]}'
        )
        parsed = parse_orchestrator_response(content)
        assert parsed.intent == "tax"
        assert [(item.intent, item.sub_query) for item in parsed.tasks] == [
            ("tax", "ภาษี"),
            ("planning", "ออม"),
        ]

    def test_single_topic_reply_has_no_tasks(self) -> None:
        """Today's single-topic format still parses, with no tasks."""
        parsed = parse_orchestrator_response('{"intent": "expense", "confidence": 0.95}')
        assert parsed.intent == "expense" and parsed.tasks == []

    def test_bad_part_is_dropped_not_the_decision(self) -> None:
        """An unknown intent or out-of-range confidence drops only that part."""
        tasks = parse_routed_tasks(
            [
                {"intent": "tax", "sub_query": "ภาษี", "confidence": 0.9},
                {"intent": "weather", "sub_query": "ฝน", "confidence": 0.9},
                {"intent": "report", "sub_query": "สรุป", "confidence": 1.5},
            ]
        )
        assert [item.intent for item in tasks] == ["tax"]

    @pytest.mark.parametrize("raw", [None, "tax", {"intent": "tax"}])
    def test_non_list_tasks_are_ignored(self, raw: Any) -> None:
        """A tasks value that is not a list is ignored."""
        assert parse_routed_tasks(raw) == []

    def test_non_object_reply_falls_back(self) -> None:
        """A JSON list instead of an object gives the default decision."""
        assert parse_orchestrator_response("[1, 2]").intent == "unknown"


class TestPrompt:
    """The multi-topic rules sit apart from the single-topic examples."""

    def test_multi_section_in_both_prompt_variants(self) -> None:
        """Multi-topic rules are present with and without few-shot examples."""
        assert ORCHESTRATOR_MULTI_INTENT_SECTION in build_orchestrator_system_prompt(True)
        assert ORCHESTRATOR_MULTI_INTENT_SECTION in build_orchestrator_system_prompt(False)

    def test_single_topic_examples_are_unchanged_in_format(self) -> None:
        """The single-topic example block (reused by Study A) has no tasks key."""
        assert '"tasks"' not in ORCHESTRATOR_EXAMPLES_SECTION

    def test_no_cap_is_stated(self) -> None:
        """The prompt says there is no limit on the number of parts."""
        assert "ไม่จำกัดจำนวนงาน" in ORCHESTRATOR_MULTI_INTENT_SECTION


class TestPreviousTopicsHint:
    """A follow-up after a multi-topic reply knows which topics were covered."""

    @pytest.mark.parametrize("intents", [None, [], ["tax"]])
    def test_single_topic_adds_nothing(self, intents: Any) -> None:
        """Single-topic history leaves routing exactly as before."""
        assert previous_topics_hint(intents) is None

    def test_several_topics_are_named(self) -> None:
        """Each topic appears with its Thai name and key."""
        hint = previous_topics_hint(["tax", "planning"])
        assert hint is not None
        assert "ภาษี (tax)" in hint and "วางแผนการเงิน (planning)" in hint

    def test_hint_sits_right_before_the_question(self) -> None:
        """classify_query adds the hint just before the user's message."""
        model = MagicMock()
        model.invoke.return_value = AIMessage(content='{"intent": "tax", "confidence": 0.9}')
        classify_query(
            "แล้วเรื่องภาษีล่ะ", model, None, NO_HINT, previous_intents=["tax", "planning"]
        )
        sent = model.invoke.call_args[0][0]
        assert isinstance(sent[-1], HumanMessage)
        assert isinstance(sent[-2], SystemMessage) and "ภาษี (tax)" in sent[-2].content


class TestOrchestrateMultiIntent:
    """A multi-topic message gets one combined reply."""

    def test_every_part_is_answered_in_order(
        self, monkeypatch: pytest.MonkeyPatch, agents: dict[str, list[str]]
    ) -> None:
        """Three topics run three agents and come back as three sections."""
        route_with(
            monkeypatch,
            decision(
                ("tax", "ภาษีเท่าไร", "0.9"),
                ("planning", "ควรออมเท่าไร", "0.85"),
                ("report", "สรุปเดือนนี้", "0.8"),
            ),
        )
        result = orchestrate_query("ภาษีเท่าไร ควรออมเท่าไร สรุปเดือนนี้")
        assert result["intent"] == "tax"
        assert result["intents"] == ["tax", "planning", "report"]
        response = result["response"]
        assert (
            response.index("## ภาษี")
            < response.index("## วางแผนการเงิน")
            < response.index("## รายงานการเงิน")
        )
        assert agents["tax"] == ["ภาษีเท่าไร"]
        assert "คำตอบ tax" in agents["planning"][0]

    def test_saved_items_are_listed(
        self, monkeypatch: pytest.MonkeyPatch, agents: dict[str, list[str]]
    ) -> None:
        """The reply ends with what was written."""
        route_with(
            monkeypatch, decision(("expense", "จ่ายค่าอาหาร 250", "0.9"), ("tax", "ภาษี", "0.9"))
        )
        response = orchestrate_query("จ่ายค่าอาหาร 250 แล้วภาษีเท่าไร")["response"]
        assert response.endswith(f"{WRITE_SUMMARY_HEADER}\n- รายจ่าย 250 บาท")

    def test_all_parts_unsure_asks_back(
        self, monkeypatch: pytest.MonkeyPatch, agents: dict[str, list[str]]
    ) -> None:
        """When no part is clear, the normal ask-back menu is shown."""
        route_with(monkeypatch, decision(("tax", "?", "0.4"), ("planning", "?", "0.5")))
        assert orchestrate_query("เงิน ๆ ๆ")["intent"] == "clarify"
        assert not agents

    def test_unsure_part_is_named_not_answered(
        self, monkeypatch: pytest.MonkeyPatch, agents: dict[str, list[str]]
    ) -> None:
        """A confident part is answered; the unsure one is listed for the user."""
        route_with(
            monkeypatch, decision(("tax", "ภาษีเท่าไร", "0.9"), ("report", "อันนั้นด้วย", "0.4"))
        )
        result = orchestrate_query("ภาษีเท่าไร อันนั้นด้วย")
        assert result["intents"] == ["tax"]
        assert result["response"] == f"คำตอบ tax\n\n{UNSURE_HEADER}\n- อันนั้นด้วย"
        assert "report" not in agents

    def test_failed_part_does_not_break_reply(
        self, monkeypatch: pytest.MonkeyPatch, agents: dict[str, list[str]]
    ) -> None:
        """One agent raising leaves the other section intact."""

        def boom(*args: Any) -> dict[str, Any]:
            raise RuntimeError("down")

        agent_map = router_agent._build_agent_map()  # pylint: disable=protected-access
        agent_map["tax"] = boom
        monkeypatch.setattr(router_agent, "_build_agent_map", lambda: agent_map)
        route_with(monkeypatch, decision(("tax", "ภาษี", "0.9"), ("planning", "ออม", "0.9")))
        result = orchestrate_query("ภาษี แล้วออม")
        assert "ตอบไม่สำเร็จ" in result["response"] and "คำตอบ planning" in result["response"]

    def test_single_topic_path_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, agents: dict[str, list[str]]
    ) -> None:
        """A one-topic message returns exactly the agent's result (no intents, no tools)."""
        routed = OrchestratorDecision(intent="tax", confidence=Decimal("0.9"))
        route_with(monkeypatch, routed)
        assert orchestrate_query("ภาษีเท่าไร") == {"intent": "tax", "response": "คำตอบ tax"}

    def test_multi_intent_can_be_turned_off(
        self, monkeypatch: pytest.MonkeyPatch, agents: dict[str, list[str]]
    ) -> None:
        """The study baseline sends a multi-topic message to the first agent only."""
        route_with(monkeypatch, decision(("tax", "ภาษี", "0.9"), ("planning", "ออม", "0.9")))
        result = orchestrate_query("ภาษี แล้วออม", allow_multi_intent=False)
        assert result == {"intent": "tax", "response": "คำตอบ tax"}

    def test_previous_intents_reach_the_router(
        self, monkeypatch: pytest.MonkeyPatch, agents: dict[str, list[str]]
    ) -> None:
        """orchestrate_query passes the previous topics to classify_query."""
        seen: dict[str, Any] = {}

        def fake_classify(*args: Any, **kwargs: Any) -> OrchestratorDecision:
            seen.update(kwargs)
            return OrchestratorDecision(intent="tax", confidence=Decimal("0.9"))

        monkeypatch.setattr(router_agent, "classify_query", fake_classify)
        orchestrate_query("แล้วภาษีล่ะ", previous_intents=["tax", "planning"])
        assert seen["previous_intents"] == ["tax", "planning"]
