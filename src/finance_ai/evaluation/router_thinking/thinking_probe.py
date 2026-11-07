"""Thinking probe: how many thinking tokens each Gemini thinking level spends.

Asks the same routing questions (through the production router) and a few
reasoning questions at every thinking level, and logs the thinking tokens
Gemini reports. It shows whether a level actually changes the model's work
on routing, which decides which levels are worth comparing in the study.

Usage:
    uv run python -m finance_ai.evaluation.router_thinking.thinking_probe
"""

import json
import statistics
import time
from pathlib import Path
from typing import Any, Optional, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from finance_ai.agents.llm_factory import create_google_chat_model
from finance_ai.core.config import get_settings
from finance_ai.evaluation.router_thinking.cli import PILOT_QUESTION_IDS
from finance_ai.evaluation.router_thinking.questions import RoutingQuestion, load_new_questions
from finance_ai.evaluation.router_thinking.runner import (
    CallRecorder,
    PRODUCTION_ROUTING,
    cached_asset_hints,
)

PROBE_MODELS = ("gemini-3.5-flash", "gemini-3.5-flash-lite")
PROBE_LEVELS: tuple[Optional[str], ...] = (None, "minimal", "low", "medium", "high")
REASONING_QUESTIONS = (
    "How many prime numbers are there between 1000 and 1100? Answer with the number only.",
    "A loan of 500,000 baht at 6% a year, compounded monthly, is repaid in 60 equal monthly "
    "payments. What is the payment? Answer with the number only.",
    "If 3 workers build 3 walls in 3 hours, how many workers build 12 walls in 6 hours? "
    "Answer with the number only.",
)
OUTPUT_FOLDER = Path("data/evaluation/results/router_thinking/thinking_probe")


class ProbeRow(BaseModel):
    """One probe call."""

    model: str
    level: str
    task: str
    prompt_id: str
    thinking_tokens: int
    output_tokens: int
    latency_seconds: float
    reply: str


def probe_model(model_name: str, level: Optional[str]) -> Runnable[Any, Any]:
    """Create a Gemini model at a thinking level (None = the model's own default).

    Args:
        model_name: Gemini model id.
        level: Thinking level, or None to send no setting.

    Returns:
        A binding, so the production router does not re-bind its own level.
    """
    settings = get_settings().model_copy(
        update={"google_model": model_name, "llm_thinking_level": None}
    )
    model = create_google_chat_model(settings)
    return model.bind(thinking_level=level) if level else model.bind()


def _row(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    model_name: str,
    level: Optional[str],
    task: str,
    prompt_id: str,
    reply: Optional[AIMessage],
    seconds: float,
) -> ProbeRow:
    """Build a ProbeRow from a reply."""
    usage: dict[str, Any] = dict(getattr(reply, "usage_metadata", None) or {})
    details = usage.get("output_token_details") or {}
    return ProbeRow(
        model=model_name,
        level=level or "default",
        task=task,
        prompt_id=prompt_id,
        thinking_tokens=int(details.get("reasoning", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        latency_seconds=seconds,
        reply=str(getattr(reply, "content", ""))[:200],
    )


def probe_routing(model_name: str, level: Optional[str], question: RoutingQuestion) -> ProbeRow:
    """Ask one routing question through the production router.

    Args:
        model_name: Gemini model id.
        level: Thinking level or None.
        question: RoutingQuestion.

    Returns:
        The probe row.
    """
    # pylint: disable=import-outside-toplevel
    from finance_ai.agents.router_agent import classify_query

    recorder = CallRecorder(probe_model(model_name, level))
    classify_query(
        question.query,
        chat_model=recorder.as_chat_model(),
        chat_history=question.chat_history or None,
        config=PRODUCTION_ROUTING,
    )
    return _row(
        model_name, level, "routing", question.question_id, recorder.reply, recorder.latency_seconds
    )


def probe_reasoning(model_name: str, level: Optional[str], index: int) -> ProbeRow:
    """Ask one reasoning question directly.

    Args:
        model_name: Gemini model id.
        level: Thinking level or None.
        index: Position in REASONING_QUESTIONS.

    Returns:
        The probe row.
    """
    start = time.perf_counter()
    reply = probe_model(model_name, level).invoke([HumanMessage(REASONING_QUESTIONS[index])])
    seconds = time.perf_counter() - start
    return _row(
        model_name, level, "reasoning", f"math_{index + 1}", cast(AIMessage, reply), seconds
    )


def run_probe() -> list[ProbeRow]:
    """Run every model × level × prompt once.

    Returns:
        All probe rows.
    """
    questions = [q for q in load_new_questions() if q.question_id in PILOT_QUESTION_IDS]
    rows: list[ProbeRow] = []
    with cached_asset_hints():
        for model_name in PROBE_MODELS:
            for level in PROBE_LEVELS:
                rows += [probe_routing(model_name, level, question) for question in questions]
                rows += [
                    probe_reasoning(model_name, level, i) for i in range(len(REASONING_QUESTIONS))
                ]
    return rows


def summary_lines(rows: list[ProbeRow]) -> list[str]:
    """Markdown table: thinking tokens per model, level and task.

    Args:
        rows: Probe rows.

    Returns:
        Table lines.
    """
    lines = [
        "| Model | Level | Task | Calls | Calls that thought | Thinking tokens mean (max) "
        "| Latency median (s) |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for model_name in PROBE_MODELS:
        for level in PROBE_LEVELS:
            for task in ("routing", "reasoning"):
                cell = [
                    r
                    for r in rows
                    if r.model == model_name and r.level == (level or "default") and r.task == task
                ]
                if not cell:
                    continue
                tokens = [r.thinking_tokens for r in cell]
                lines.append(
                    f"| {model_name} | {level or 'default (no setting)'} | {task} | {len(cell)} "
                    f"| {sum(t > 0 for t in tokens)} | {statistics.fmean(tokens):.0f} "
                    f"({max(tokens)}) | {statistics.median(r.latency_seconds for r in cell):.2f} |"
                )
    return lines


def main() -> Path:
    """Run the probe and write probe.jsonl and probe.md.

    Returns:
        Path of probe.md.
    """
    rows = run_probe()
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    (OUTPUT_FOLDER / "probe.jsonl").write_text(
        "\n".join(json.dumps(r.model_dump(), ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    header = [
        "# Thinking probe: thinking tokens per Gemini thinking level",
        "",
        "10 routing questions (production router prompt) and 3 reasoning questions, "
        "once per model × level. Thinking tokens = `output_token_details.reasoning` reported "
        "by Gemini (0 = no thinking). 'default' = no thinking setting sent.",
        "",
    ]
    report = OUTPUT_FOLDER / "probe.md"
    report.write_text("\n".join(header + summary_lines(rows)) + "\n", encoding="utf-8")
    print(f"Report: {report}")
    return report


if __name__ == "__main__":
    main()
