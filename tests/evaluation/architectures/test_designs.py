"""Tests for the four architecture designs with fake models (no network)."""

from collections.abc import Callable
from typing import Any

import pytest

from finance_ai.agents.prompts import ANTI_HALLUCINATION_RULES
from finance_ai.evaluation.architectures import hierarchical, hub_spoke, peer_to_peer
from finance_ai.evaluation.architectures.design_context import DesignContext, DesignError
from finance_ai.evaluation.architectures.hierarchical import run_hierarchical
from finance_ai.evaluation.architectures.hub_spoke import run_hub_spoke
from finance_ai.evaluation.architectures.peer_to_peer import (
    MAX_HOPS,
    build_handoff_prompt,
    entry_agent_for,
    run_peer_to_peer,
)
from finance_ai.evaluation.architectures.single_agent import (
    all_tools,
    build_single_agent_prompt,
    run_single_agent,
)

ContextFactory = Callable[..., DesignContext]


def _answer_with_agent_name(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make run_specialist in a module answer with the agent's name."""
    monkeypatch.setattr(module, "run_specialist", lambda agent, query, context: f"ans:{agent}")


class TestHubSpoke:
    """Hub-and-spoke wraps production orchestration."""

    @pytest.mark.parametrize(
        ("intent", "route"),
        [
            ("tax", "tax"),
            ("general_chat", "general"),
            ("unknown", "general"),
            ("clarify", "clarify"),
        ],
    )
    def test_route_from_intent(
        self,
        intent: str,
        route: str,
        monkeypatch: pytest.MonkeyPatch,
        make_context: ContextFactory,
    ) -> None:
        """The production intent becomes a one-step route."""
        monkeypatch.setattr(
            hub_spoke, "orchestrate_query", lambda *a, **k: {"intent": intent, "response": "ok"}
        )
        outcome = run_hub_spoke("q", make_context())
        assert outcome.route == [route] and outcome.answer == "ok"

    def test_error_raises(
        self, monkeypatch: pytest.MonkeyPatch, make_context: ContextFactory
    ) -> None:
        """A production error becomes a DesignError."""
        monkeypatch.setattr(
            hub_spoke, "orchestrate_query", lambda *a, **k: {"intent": "error", "response": "down"}
        )
        with pytest.raises(DesignError, match="down"):
            run_hub_spoke("q", make_context())


class TestHierarchical:
    """Two routing calls, then one specialist."""

    @pytest.fixture(autouse=True)
    def _fake_specialists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Answer with the agent name instead of running real agents."""
        _answer_with_agent_name(hierarchical, monkeypatch)

    def test_two_levels_pick_agent(self, make_context: ContextFactory) -> None:
        """Group then agent."""
        context = make_context(
            '{"group": "operations", "confidence": 0.9}', '{"intent": "expense", "confidence": 0.9}'
        )
        outcome = run_hierarchical("กาแฟ 80", context)
        assert outcome.route == ["expense"] and outcome.answer == "ans:expense"

    def test_low_group_confidence_asks_back(self, make_context: ContextFactory) -> None:
        """An unsure first level asks the user back."""
        outcome = run_hierarchical(
            "เงิน", make_context('{"group": "operations", "confidence": 0.4}')
        )
        assert outcome.route == ["clarify"]

    def test_general_group_skips_level_two(self, make_context: ContextFactory) -> None:
        """The general group goes straight to general chat."""
        outcome = run_hierarchical(
            "สวัสดี", make_context('{"group": "general", "confidence": 0.9}')
        )
        assert outcome.route == ["general"]

    def test_low_agent_confidence_asks_back(self, make_context: ContextFactory) -> None:
        """An unsure second level asks the user back."""
        context = make_context(
            '{"group": "analysis", "confidence": 0.9}', '{"intent": "report", "confidence": 0.3}'
        )
        assert run_hierarchical("q", context).route == ["clarify"]

    def test_agent_outside_group_goes_general(self, make_context: ContextFactory) -> None:
        """A level-2 answer outside the group falls back to general chat."""
        context = make_context(
            '{"group": "analysis", "confidence": 0.9}', '{"intent": "tax", "confidence": 0.9}'
        )
        assert run_hierarchical("q", context).route == ["general"]

    def test_unknown_group_asks_back(self, make_context: ContextFactory) -> None:
        """A group name that does not exist asks the user back."""
        assert run_hierarchical("q", make_context('{"group": "x", "confidence": 0.9}')).route == [
            "clarify"
        ]


class TestPeerToPeer:
    """Agents hand the question to each other."""

    @pytest.fixture(autouse=True)
    def _fake_specialists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Answer with the agent name instead of running real agents."""
        _answer_with_agent_name(peer_to_peer, monkeypatch)

    def test_entry_rotates_round_robin(self) -> None:
        """Each of the six agents starts once per six tasks."""
        assert len({entry_agent_for(index) for index in range(6)}) == 6

    def test_handle_at_entry(self, make_context: ContextFactory) -> None:
        """The entry agent accepts."""
        outcome = run_peer_to_peer("q", make_context('{"action": "handle"}', task_index=0))
        assert outcome.route == ["tax"] and outcome.answer == "ans:tax"

    def test_forward_then_handle(self, make_context: ContextFactory) -> None:
        """A forward moves the question to the named peer."""
        context = make_context(
            '{"action": "forward", "to": "expense"}', '{"action": "handle"}', task_index=0
        )
        assert run_peer_to_peer("q", context).route == ["tax", "expense"]

    def test_ask_user(self, make_context: ContextFactory) -> None:
        """ask_user ends with the ask-back message."""
        outcome = run_peer_to_peer("เงิน", make_context('{"action": "ask_user"}', task_index=0))
        assert outcome.route == ["tax", "clarify"]

    def test_hop_limit(self, make_context: ContextFactory) -> None:
        """After MAX_HOPS forwards the last agent must answer."""
        forwards = ['{"action": "forward", "to": "expense"}', '{"action": "forward", "to": "tax"}']
        context = make_context(*(forwards * 2), task_index=0)
        outcome = run_peer_to_peer("q", context)
        assert len(outcome.route) == MAX_HOPS + 1

    def test_invalid_target_stays(self, make_context: ContextFactory) -> None:
        """Forwarding to a non-agent keeps the current agent."""
        context = make_context('{"action": "forward", "to": "nobody"}', task_index=0)
        assert run_peer_to_peer("q", context).route == ["tax"]

    def test_handoff_prompt_lists_peers_not_self(self) -> None:
        """The prompt names the five peers but not the agent itself."""
        prompt = build_handoff_prompt("tax")
        assert '"expense"' in prompt and '- "tax":' not in prompt


class TestSingleAgent:
    """One agent with every tool."""

    def test_has_31_unique_tools(self) -> None:
        """All specialist tools, each once."""
        names = [tool.name for tool in all_tools()]
        assert len(names) == len(set(names)) == 31

    def test_prompt_covers_every_domain_once(self) -> None:
        """Every section appears and the shared rules appear once."""
        prompt = build_single_agent_prompt()
        assert all(title in prompt for title in ("ด้านภาษี", "ด้านรายงานการเงิน"))
        assert prompt.count(ANTI_HALLUCINATION_RULES) == 1

    def test_plain_answer_has_empty_route(
        self, monkeypatch: pytest.MonkeyPatch, make_context: ContextFactory
    ) -> None:
        """A direct answer returns with no route."""
        monkeypatch.setattr(
            "finance_ai.agents.recommendation_agent.get_user_risk_level", lambda *args: None
        )
        outcome = run_single_agent("สวัสดี", make_context("สวัสดีค่ะ"))
        assert not outcome.route and outcome.answer.startswith("สวัสดีค่ะ")
