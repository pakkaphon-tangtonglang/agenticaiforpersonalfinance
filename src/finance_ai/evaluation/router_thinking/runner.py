"""Run the router study: every question through every setup, one JSONL row per call.

The log is the source of truth. Restarts skip finished calls, and provider
failures (rate limits, 5xx) pause the study without writing a row.
"""

import json
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, cast
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field, ValidationError

from finance_ai.agents import router_agent
from finance_ai.agents.multi_intent import merge_tasks_by_agent, split_by_confidence
from finance_ai.agents.router_agent import DEFAULT_ABLATION_CONFIG, classify_query
from finance_ai.core.logging import get_logger
from finance_ai.evaluation.architectures.runner import StudyPaused, is_provider_unavailable
from finance_ai.evaluation.router_thinking.questions import (
    RoutingQuestion,
    accepted_destinations,
    destination_for,
    routed_correctly,
)
from finance_ai.evaluation.router_thinking.setups import RouterSetup

logger = get_logger(__name__)

LOG_NAME = "calls.jsonl"
# Production applies the 0.7 ask-back rule in orchestrate_query, after classify_query;
# this flag applies the same rule inside classify_query so the study scores what users see.
PRODUCTION_ROUTING = replace(DEFAULT_ABLATION_CONFIG, apply_confidence_threshold=True)
RETRY_DELAYS_SECONDS = (5.0, 10.0, 20.0)
SHUFFLE_SEED = 20261107


class RouterCallRow(BaseModel):
    """One logged router call."""

    setup: str
    model: str
    thinking_level: str
    round_index: int = Field(ge=0)
    question_id: str
    source: str
    expected: str
    predicted_intent: str = ""
    confidence: str = "0"
    destination: str = ""
    destinations: list[str] = Field(default_factory=list)
    correct: bool = False
    parse_failed: bool = False
    raw_reply: str = ""
    asset_hint: Optional[str] = None
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    error: Optional[str] = None
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def key(self) -> tuple[str, int, str]:
        """Resume key: (setup, round, question)."""
        return (self.setup, self.round_index, self.question_id)


class PlannedCall(BaseModel):
    """One call still to make."""

    setup: RouterSetup
    round_index: int
    question: RoutingQuestion

    @property
    def key(self) -> tuple[str, int, str]:
        """Resume key matching RouterCallRow.key."""
        return (self.setup.name, self.round_index, self.question.question_id)


class CallRecorder:
    """Wraps a router model to keep the last reply and time only the model call."""

    def __init__(self, model: Runnable[Any, Any]) -> None:
        """Store the model to wrap.

        Args:
            model: Gemini model with the thinking level bound.
        """
        self._model = model
        self.reply: Optional[AIMessage] = None
        self.latency_seconds = 0.0

    def invoke(self, messages: Any) -> AIMessage:
        """Call the model, keeping the reply and its wall-clock time.

        Args:
            messages: Router messages.

        Returns:
            The model reply.
        """
        start = time.perf_counter()
        self.reply = cast(AIMessage, self._model.invoke(messages))
        self.latency_seconds = time.perf_counter() - start
        return self.reply

    def as_chat_model(self) -> BaseChatModel:
        """Return this recorder in the shape classify_query accepts."""
        return cast(BaseChatModel, RunnableLambda(self.invoke))


def plan_calls(
    questions: list[RoutingQuestion], setups: list[RouterSetup], rounds: int
) -> list[PlannedCall]:
    """List every (setup, round, question) call in a fixed random order.

    Args:
        questions: Questions to ask.
        setups: Setups to compare.
        rounds: Repeats per question and setup.

    Returns:
        Shuffled calls (same order on every restart).
    """
    calls = [
        PlannedCall(setup=setup, round_index=round_index, question=question)
        for round_index in range(rounds)
        for setup in setups
        for question in questions
    ]
    random.Random(SHUFFLE_SEED).shuffle(calls)
    return calls


def read_log(log_path: Path) -> list[RouterCallRow]:
    """Read logged calls, skipping a partly written last line.

    Args:
        log_path: JSONL log.

    Returns:
        Rows in log order; empty when the file is missing.
    """
    if not log_path.exists():
        return []
    rows: list[RouterCallRow] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(RouterCallRow.model_validate_json(line))
        except ValidationError:
            logger.warning("Skipping unreadable log line")
    return rows


def _usage(reply: Optional[AIMessage]) -> tuple[int, int, int]:
    """Return (input, output, thinking) tokens from a reply.

    Args:
        reply: Model reply, or None.

    Returns:
        Token counts; zeros when the provider sends none.
    """
    usage: dict[str, Any] = dict(getattr(reply, "usage_metadata", None) or {})
    details = usage.get("output_token_details") or {}
    reasoning = int(details.get("reasoning", 0))
    return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)), reasoning


def _reply_text(reply: Optional[AIMessage]) -> str:
    """Return the reply text (Gemini may send a list of content blocks)."""
    if reply is None:
        return ""
    return router_agent._content_to_text(reply.content)  # pylint: disable=protected-access


def is_parse_failure(intent: str, confidence: str, raw_reply: str) -> bool:
    """Tell whether the router fell back to its default because the reply was unreadable.

    Args:
        intent: Parsed intent.
        confidence: Parsed confidence as text.
        raw_reply: Model reply text.

    Returns:
        True when the default "unknown / 0" came from a parse failure.

    Example:
        >>> is_parse_failure("unknown", "0", "sorry")
        True
    """
    return intent == "unknown" and float(confidence) == 0.0 and '"unknown"' not in raw_reply


def production_destinations(decision: Any) -> list[str]:
    """Return every place production sends a message, as the router decided.

    A message with two or more agents in its parts runs one agent per
    confident part (ask-back when none is confident); otherwise the single
    intent decides, as before.

    Args:
        decision: OrchestratorDecision after the confidence threshold.

    Returns:
        One destination, or several for a multi-topic message.

    Example:
        >>> production_destinations(OrchestratorDecision(intent="unknown", confidence=1))
        ['general_chat']
    """
    tasks = merge_tasks_by_agent(decision.tasks)
    if len(tasks) < 2:
        return [destination_for(decision.intent)]
    confident, _unsure = split_by_confidence(tasks)
    return [destination_for(task.intent) for task in confident] or ["clarify"]


def build_row(call: PlannedCall, recorder: CallRecorder, decision: Any) -> RouterCallRow:
    """Turn one finished classification into a log row.

    Args:
        call: The planned call.
        recorder: Recorder holding the reply and model latency.
        decision: OrchestratorDecision after the confidence threshold.

    Returns:
        The scored row.
    """
    raw_reply = _reply_text(recorder.reply)
    confidence = str(decision.confidence)
    parse_failed = is_parse_failure(decision.intent, confidence, raw_reply)
    destinations = production_destinations(decision)
    input_tokens, output_tokens, thinking_tokens = _usage(recorder.reply)
    return RouterCallRow(
        setup=call.setup.name,
        model=call.setup.model,
        thinking_level=call.setup.thinking_level,
        round_index=call.round_index,
        question_id=call.question.question_id,
        source=call.question.source,
        expected=call.question.expected,
        predicted_intent=decision.intent,
        confidence=confidence,
        destination=destinations[0],
        destinations=destinations,
        correct=not parse_failed
        and routed_correctly(destinations, accepted_destinations(call.question)),
        parse_failed=parse_failed,
        raw_reply=raw_reply,
        latency_seconds=recorder.latency_seconds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
    )


def classify_once(call: PlannedCall, model: Runnable[Any, Any]) -> RouterCallRow:
    """Run the production classify_query once for a planned call.

    Args:
        call: Setup, round and question.
        model: The setup's Gemini model.

    Returns:
        The scored row (raises provider errors to the caller).
    """
    recorder = CallRecorder(model)
    history = call.question.chat_history or None
    decision = classify_query(
        call.question.query,
        chat_model=recorder.as_chat_model(),
        chat_history=history,
        config=PRODUCTION_ROUTING,
    )
    return build_row(call, recorder, decision)


def classify_with_retry(
    call: PlannedCall, model: Runnable[Any, Any], sleep: Callable[[float], None] = time.sleep
) -> RouterCallRow:
    """Classify, retrying provider failures, and pause the study if they persist.

    Args:
        call: The planned call.
        model: The setup's Gemini model.
        sleep: Sleep function (replaced in tests).

    Returns:
        The scored row, or a row with ``error`` for a non-provider failure.

    Raises:
        StudyPaused: When the provider keeps failing.
    """
    for delay in (*RETRY_DELAYS_SECONDS, None):
        try:
            return classify_once(call, model)
        except Exception as error:  # noqa: BLE001 - classified below
            if not is_provider_unavailable(error):
                return _error_row(call, error)
            if delay is None:
                raise StudyPaused(f"Provider unavailable at {call.key}: {error}") from error
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def _error_row(call: PlannedCall, error: Exception) -> RouterCallRow:
    """Build the log row for a call that failed for a non-provider reason."""
    return RouterCallRow(
        setup=call.setup.name,
        model=call.setup.model,
        thinking_level=call.setup.thinking_level,
        round_index=call.round_index,
        question_id=call.question.question_id,
        source=call.question.source,
        expected=call.question.expected,
        error=f"{type(error).__name__}: {error}",
    )


@contextmanager
def cached_asset_hints() -> Iterator[dict[str, Optional[str]]]:
    """Resolve each query's asset hint once and reuse it for every setup.

    Every setup then sees the same prompt, and Yahoo Finance latency or
    outages never land in one setup's numbers.

    Yields:
        The cache (query → hint), for logging.
    """
    cache: dict[str, Optional[str]] = {}
    original = router_agent._resolve_asset_hint  # pylint: disable=protected-access

    def cached(query: str) -> Optional[str]:
        if query not in cache:
            cache[query] = original(query)
        return cache[query]

    with patch.object(router_agent, "_resolve_asset_hint", cached):
        yield cache


def run_study(
    calls: list[PlannedCall], models: dict[str, Runnable[Any, Any]], log_path: Path
) -> int:
    """Make every planned call not yet in the log.

    Args:
        calls: Planned calls in run order.
        models: Setup name → Gemini model.
        log_path: JSONL log (appended).

    Returns:
        Number of calls made in this session.

    Raises:
        StudyPaused: When the provider keeps failing (finished rows stay logged).
    """
    done = {row.key for row in read_log(log_path)}
    pending = [call for call in calls if call.key not in done]
    logger.info("Router study: %d done, %d to go", len(done), len(pending))
    with cached_asset_hints() as hints:
        for index, call in enumerate(pending, start=1):
            row = classify_with_retry(call, models[call.setup.name])
            row.asset_hint = hints.get(call.question.query)
            _append_row(log_path, row)
            if index % 25 == 0:
                logger.info("Router study: %d / %d", index, len(pending))
    return len(pending)


def _append_row(log_path: Path, row: RouterCallRow) -> None:
    """Append one row to the log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n")
