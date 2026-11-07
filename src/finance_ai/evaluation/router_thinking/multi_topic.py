"""Multi-topic routing test: does the router split a message into the right agents?

Runs the production router (gemini-3.5-flash, minimal thinking, production
ask-back rule) on the multi-topic questions, reusing the router study's
logging, retries and resume. Scores each message by the set of places
production sends it: exact match, Jaccard similarity and order; for
one-question look-alikes, whether it wrongly split.

Usage:
    uv run python -m finance_ai.evaluation.router_thinking.multi_topic
"""

import argparse
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from finance_ai.evaluation.architectures.runner import StudyPaused
from finance_ai.evaluation.architectures.stats import wilson_interval
from finance_ai.evaluation.router_thinking.questions import Destination, RoutingQuestion
from finance_ai.evaluation.router_thinking.runner import (
    LOG_NAME,
    RouterCallRow,
    plan_calls,
    read_log,
    run_study,
)
from finance_ai.evaluation.router_thinking.setups import SETUPS, build_router_model

QUESTIONS_PATH = Path("data/evaluation/routing_multi_dataset.yaml")
RESULTS_DIR = Path("data/evaluation/results/router_thinking/multi_topic")
PRODUCTION_SETUP = "flash_minimal"


class MultiTopicQuestion(BaseModel):
    """A message with every agent it should reach, in the order asked.

    Example:
        >>> MultiTopicQuestion(question_id="m", query="q", expected=["tax"], kind="single").kind
        'single'
    """

    question_id: str
    query: str
    expected: list[Destination] = Field(min_length=1)
    kind: Literal["multi", "single"]


class MultiTopicScore(BaseModel):
    """How one routed message compares with its expected agents."""

    question_id: str
    kind: str
    expected: list[str]
    predicted: list[str]
    exact: bool
    jaccard: float
    order_ok: bool
    split: bool


class MultiTopicSummary(BaseModel):
    """Headline numbers for the multi-topic routing test."""

    multi_questions: int
    multi_exact: int
    multi_exact_low: float
    multi_exact_high: float
    multi_mean_jaccard: float
    multi_order_ok: int
    single_questions: int
    single_correct: int
    single_wrong_splits: int


def load_multi_topic_questions(path: Path = QUESTIONS_PATH) -> list[MultiTopicQuestion]:
    """Load the multi-topic routing questions.

    Args:
        path: YAML file with a ``questions`` list.

    Returns:
        Validated questions.
    """
    import yaml  # pylint: disable=import-outside-toplevel

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [MultiTopicQuestion.model_validate(item) for item in raw["questions"]]


def as_routing_question(question: MultiTopicQuestion) -> RoutingQuestion:
    """Wrap a question for the router study runner (first agent as ``expected``).

    Args:
        question: Multi-topic question.

    Returns:
        RoutingQuestion with source "multi".
    """
    return RoutingQuestion(
        question_id=question.question_id,
        query=question.query,
        expected=question.expected[0],
        source="multi",
    )


def score_question(question: MultiTopicQuestion, row: RouterCallRow) -> MultiTopicScore:
    """Score one routed message.

    Args:
        question: The question with its expected agents.
        row: The logged router call.

    Returns:
        Exact set match, Jaccard similarity, order and split flags (errors score 0).

    Example:
        >>> score_question(question, row).jaccard  # doctest: +SKIP
        1.0
    """
    predicted = [] if row.error or row.parse_failed else row.destinations or [row.destination]
    expected_set, predicted_set = set(question.expected), set(predicted)
    union = expected_set | predicted_set
    return MultiTopicScore(
        question_id=question.question_id,
        kind=question.kind,
        expected=list(question.expected),
        predicted=predicted,
        exact=expected_set == predicted_set,
        jaccard=len(expected_set & predicted_set) / len(union) if union else 0.0,
        order_ok=predicted == list(question.expected),
        split=len(predicted) > 1,
    )


def summarize(scores: list[MultiTopicScore]) -> MultiTopicSummary:
    """Headline numbers for multi-topic messages and one-question look-alikes.

    Args:
        scores: One score per question.

    Returns:
        Counts, rates and 95% Wilson intervals.
    """
    multi = [score for score in scores if score.kind == "multi"]
    single = [score for score in scores if score.kind == "single"]
    exact = sum(score.exact for score in multi)
    low, high = wilson_interval(exact, len(multi))
    return MultiTopicSummary(
        multi_questions=len(multi),
        multi_exact=exact,
        multi_exact_low=low,
        multi_exact_high=high,
        multi_mean_jaccard=sum(score.jaccard for score in multi) / len(multi) if multi else 0.0,
        multi_order_ok=sum(score.order_ok for score in multi),
        single_questions=len(single),
        single_correct=sum(score.exact and not score.split for score in single),
        single_wrong_splits=sum(score.split for score in single),
    )


def write_report(folder: Path, questions: list[MultiTopicQuestion]) -> Path:
    """Score the log and write report.md and summary.json.

    Args:
        folder: Results folder holding calls.jsonl.
        questions: The questions that were asked.

    Returns:
        Path of report.md.
    """
    rows = {row.question_id: row for row in read_log(folder / LOG_NAME)}
    scores = [score_question(q, rows[q.question_id]) for q in questions if q.question_id in rows]
    summary = summarize(scores)
    (folder / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    report = folder / "report.md"
    report.write_text(format_report(summary, scores), encoding="utf-8")
    return report


def format_report(summary: MultiTopicSummary, scores: list[MultiTopicScore]) -> str:
    """Render the report as markdown.

    Args:
        summary: Output of summarize().
        scores: Per-question scores.

    Returns:
        Markdown text.
    """
    lines = [
        "# Multi-topic routing (gemini-3.5-flash, minimal thinking, production ask-back rule)",
        "",
        f"- Multi-topic messages: exact agent set {summary.multi_exact}/"
        f"{summary.multi_questions} ({summary.multi_exact_low:.1%}–"
        f"{summary.multi_exact_high:.1%} 95% CI), mean Jaccard"
        f" {summary.multi_mean_jaccard:.3f}, same order"
        f" {summary.multi_order_ok}/{summary.multi_questions}",
        f"- One-question look-alikes: correct {summary.single_correct}/"
        f"{summary.single_questions}, wrongly split {summary.single_wrong_splits}",
        "",
        "| Question | Kind | Expected | Predicted | Exact | Jaccard | Order |",
        "|---|---|---|---|:-:|---:|:-:|",
    ]
    for score in scores:
        lines.append(
            f"| {score.question_id} | {score.kind} | {', '.join(score.expected)} "
            f"| {', '.join(score.predicted) or '—'} | {'✓' if score.exact else '✗'} "
            f"| {score.jaccard:.2f} | {'✓' if score.order_ok else '✗'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> Path:
    """Run the multi-topic routing test (or rebuild its report).

    Args:
        argv: Arguments (defaults to sys.argv).

    Returns:
        Path of report.md.
    """
    parser = argparse.ArgumentParser(description="Multi-topic routing test")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR, help="results folder")
    parser.add_argument("--report-only", action="store_true", help="rebuild report from the log")
    args = parser.parse_args(argv)
    questions = load_multi_topic_questions()
    if not args.report_only:
        setup = SETUPS[PRODUCTION_SETUP]
        calls = plan_calls([as_routing_question(q) for q in questions], [setup], rounds=1)
        try:
            run_study(calls, {setup.name: build_router_model(setup)}, args.output / LOG_NAME)
        except StudyPaused as pause:
            print(f"Paused: {pause}")
    report = write_report(args.output, questions)
    print(f"Report: {report}")
    return report


if __name__ == "__main__":
    main()
