"""Tests for the router study runner."""

from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import Runnable, RunnableLambda

from finance_ai.agents import router_agent
from finance_ai.evaluation.architectures.runner import StudyPaused
from finance_ai.evaluation.router_thinking.runner import (
    PlannedCall,
    cached_asset_hints,
    classify_once,
    classify_with_retry,
    is_parse_failure,
    plan_calls,
    read_log,
    run_study,
)
from finance_ai.evaluation.router_thinking.questions import Destination
from finance_ai.evaluation.router_thinking.setups import SETUPS

from .conftest import fake_router_model, make_question

SETUP = SETUPS["flash_minimal"]


def _call(expected: Destination = "tax", query: str = "ภาษีปีนี้") -> PlannedCall:
    """A planned call for the production setup."""
    return PlannedCall(
        setup=SETUP, round_index=0, question=make_question(expected=expected, query=query)
    )


def _failing(message: str) -> Runnable[Any, Any]:
    """A model that always raises."""

    def fail(_messages: Any) -> Any:
        raise RuntimeError(message)

    return RunnableLambda(fail)


class TestPlan:
    """Calls are planned in a fixed shuffled order."""

    def test_every_combination_once(self) -> None:
        """Questions × setups × rounds, no duplicates."""
        questions = [make_question("a"), make_question("b")]
        calls = plan_calls(questions, list(SETUPS.values()), rounds=2)
        assert len({call.key for call in calls}) == len(calls) == 16

    def test_order_is_repeatable(self) -> None:
        """Restarts see the same order."""
        questions = [make_question(str(i)) for i in range(5)]
        first = [c.key for c in plan_calls(questions, list(SETUPS.values()), 1)]
        assert first == [c.key for c in plan_calls(questions, list(SETUPS.values()), 1)]


class TestParseFailure:
    """Unreadable replies are told apart from a real 'unknown'."""

    def test_default_decision_from_bad_reply(self) -> None:
        """The router's fallback counts as a parse failure."""
        assert is_parse_failure("unknown", "0", "I cannot answer")

    def test_real_unknown_is_not_a_failure(self) -> None:
        """A model that says unknown is a real answer."""
        assert not is_parse_failure("unknown", "0", '{"intent": "unknown", "confidence": 0}')


class TestClassifyOnce:
    """One call runs the production router and scores the destination."""

    def test_correct_route_with_tokens(self) -> None:
        """Right intent → correct, tokens and thinking tokens logged."""
        row = classify_once(_call(), fake_router_model(thinking_tokens=300))
        assert row.correct and row.destination == "tax"
        assert (row.input_tokens, row.thinking_tokens) == (1200, 300)

    def test_low_confidence_asks_back(self) -> None:
        """Confidence below 0.7 becomes clarify, as in production."""
        model = fake_router_model('{"intent": "expense", "confidence": 0.4}')
        row = classify_once(_call(expected="clarify", query="บันทึกให้หน่อย"), model)
        assert row.destination == "clarify" and row.correct

    def test_unreadable_reply_is_wrong_even_for_general_chat(self) -> None:
        """A parse failure never counts as correct."""
        row = classify_once(_call(expected="general_chat"), fake_router_model("sorry"))
        assert row.parse_failed and not row.correct


class TestRetry:
    """Provider failures pause; other failures are logged."""

    def test_provider_failure_pauses(self) -> None:
        """A rate limit that does not clear raises StudyPaused after the retries."""
        slept: list[float] = []
        with pytest.raises(StudyPaused):
            classify_with_retry(_call(), _failing("429 RESOURCE_EXHAUSTED"), sleep=slept.append)
        assert slept == [5.0, 10.0, 20.0]

    def test_other_failure_logged(self) -> None:
        """A bug becomes an error row instead of stopping the study."""
        row = classify_with_retry(_call(), _failing("boom"), sleep=lambda _s: None)
        assert row.error == "RuntimeError: boom" and not row.correct


class TestRunStudy:
    """The study appends rows and resumes."""

    def test_runs_then_resumes(self, tmp_path: Path) -> None:
        """A second run makes no calls."""
        log = tmp_path / "calls.jsonl"
        calls = plan_calls([make_question("a"), make_question("b")], [SETUP], 1)
        models: dict[str, Runnable[Any, Any]] = {SETUP.name: fake_router_model()}
        assert run_study(calls, models, log) == 2
        assert run_study(calls, models, log) == 0
        assert len(read_log(log)) == 2

    def test_partial_line_skipped(self, tmp_path: Path) -> None:
        """A half-written last line is ignored."""
        log = tmp_path / "calls.jsonl"
        log.write_text('{"setup": "flash', encoding="utf-8")
        assert not read_log(log)

    def test_missing_log_is_empty(self, tmp_path: Path) -> None:
        """No log yet → no rows."""
        assert not read_log(tmp_path / "none.jsonl")


class TestAssetHints:
    """Each query's asset hint is resolved once."""

    def test_hint_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The second lookup reuses the first result."""
        lookups: list[str] = []

        def lookup(query: str) -> str:
            lookups.append(query)
            return "hint"

        monkeypatch.setattr(router_agent, "_resolve_asset_hint", lookup)
        with cached_asset_hints() as cache:
            router_agent._resolve_asset_hint("PTT")  # pylint: disable=protected-access
            router_agent._resolve_asset_hint("PTT")  # pylint: disable=protected-access
        assert lookups == ["PTT"] and cache == {"PTT": "hint"}
