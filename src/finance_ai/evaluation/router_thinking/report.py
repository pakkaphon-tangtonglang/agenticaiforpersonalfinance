"""Summarise the router study log into Markdown and JSON."""

import json
import random
import statistics
from pathlib import Path

from pydantic import BaseModel

from finance_ai.evaluation.architectures.stats import (
    bootstrap_median_interval,
    exact_binomial_p,
    percentile,
    wilson_interval,
)
from finance_ai.evaluation.router_thinking.questions import (
    DESTINATIONS,
    RoutingQuestion,
    accepted_destinations,
    load_all_questions,
    routed_correctly,
)
from finance_ai.evaluation.router_thinking.runner import RouterCallRow, read_log
from finance_ai.evaluation.router_thinking.setups import COMPARISONS, SETUPS

NON_INFERIORITY_MARGIN = 0.03


class SetupSummary(BaseModel):
    """Headline numbers for one setup."""

    setup: str
    calls: int
    accuracy: float
    accuracy_low: float
    accuracy_high: float
    accuracy_new: float
    accuracy_sept35: float
    parse_failures: int
    splits: int
    errors: int
    latency_median: float
    latency_p95: float
    input_tokens: float
    output_tokens: float
    thinking_tokens: float
    by_destination: dict[str, float]


class Comparison(BaseModel):
    """Paired comparison of two setups on the same (round, question) pairs."""

    first: str
    second: str
    label: str
    pairs: int
    accuracy_diff: float
    accuracy_diff_low: float
    accuracy_diff_high: float
    only_first: int
    only_second: int
    mcnemar_p: float
    latency_diff_median: float
    latency_diff_low: float
    latency_diff_high: float
    sign_p: float


def _rate(rows: list[RouterCallRow]) -> float:
    """Share of correct rows (0.0 for none)."""
    return sum(row.correct for row in rows) / len(rows) if rows else 0.0


def _mean(values: list[float]) -> float:
    """Mean, 0.0 for an empty list."""
    return statistics.fmean(values) if values else 0.0


def summarize_setup(setup: str, rows: list[RouterCallRow]) -> SetupSummary:
    """Summarise one setup's calls (error rows count as wrong).

    Args:
        setup: Setup name.
        rows: That setup's rows.

    Returns:
        The summary.
    """
    answered = [row for row in rows if row.error is None]
    low, high = wilson_interval(sum(row.correct for row in rows), len(rows))
    latencies = [row.latency_seconds for row in answered]
    return SetupSummary(
        setup=setup,
        calls=len(rows),
        accuracy=_rate(rows),
        accuracy_low=low,
        accuracy_high=high,
        accuracy_new=_rate([row for row in rows if row.source == "new"]),
        accuracy_sept35=_rate([row for row in rows if row.source == "sept35"]),
        parse_failures=sum(row.parse_failed for row in rows),
        splits=sum(len(row.destinations) > 1 for row in rows),
        errors=len(rows) - len(answered),
        latency_median=percentile(latencies, 0.5),
        latency_p95=percentile(latencies, 0.95),
        input_tokens=_mean([float(row.input_tokens) for row in answered]),
        output_tokens=_mean([float(row.output_tokens) for row in answered]),
        thinking_tokens=_mean([float(row.thinking_tokens) for row in answered]),
        by_destination={
            name: _rate([row for row in rows if row.expected == name]) for name in DESTINATIONS
        },
    )


def bootstrap_mean_interval(
    differences: list[float], resamples: int = 2000, seed: int = 11
) -> tuple[float, float]:
    """95 % bootstrap interval for the mean of paired differences.

    Args:
        differences: Paired differences (e.g. +1 / 0 / -1 for correctness).
        resamples: Bootstrap resamples.
        seed: Random seed.

    Returns:
        (low, high); (0, 0) for an empty sample.

    Example:
        >>> bootstrap_mean_interval([0.0, 0.0])
        (0.0, 0.0)
    """
    if not differences:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = [
        statistics.fmean(rng.choices(differences, k=len(differences))) for _ in range(resamples)
    ]
    return (percentile(means, 0.025), percentile(means, 0.975))


def _pairs(
    rows: list[RouterCallRow], first: str, second: str
) -> list[tuple[RouterCallRow, RouterCallRow]]:
    """Match two setups' rows on (round, question); error rows are left out."""
    by_key = {(row.setup, row.round_index, row.question_id): row for row in rows if not row.error}
    return [
        (row, by_key[(second, row.round_index, row.question_id)])
        for key, row in by_key.items()
        if key[0] == first and (second, key[1], key[2]) in by_key
    ]


def compare(rows: list[RouterCallRow], first: str, second: str, label: str) -> Comparison:
    """Paired accuracy and latency comparison (first minus second).

    Args:
        rows: All rows.
        first: Setup name.
        second: Setup name.
        label: What the comparison answers.

    Returns:
        The comparison.
    """
    pairs = _pairs(rows, first, second)
    accuracy_diffs = [float(a.correct) - float(b.correct) for a, b in pairs]
    latency_diffs = [a.latency_seconds - b.latency_seconds for a, b in pairs]
    only_first = sum(a.correct and not b.correct for a, b in pairs)
    only_second = sum(b.correct and not a.correct for a, b in pairs)
    acc_low, acc_high = bootstrap_mean_interval(accuracy_diffs)
    lat_low, lat_high = bootstrap_median_interval(latency_diffs)
    return Comparison(
        first=first,
        second=second,
        label=label,
        pairs=len(pairs),
        accuracy_diff=_mean(accuracy_diffs),
        accuracy_diff_low=acc_low,
        accuracy_diff_high=acc_high,
        only_first=only_first,
        only_second=only_second,
        mcnemar_p=exact_binomial_p(only_first, only_second),
        latency_diff_median=percentile(latency_diffs, 0.5),
        latency_diff_low=lat_low,
        latency_diff_high=lat_high,
        sign_p=exact_binomial_p(
            sum(diff < 0 for diff in latency_diffs), sum(diff > 0 for diff in latency_diffs)
        ),
    )


def verdict(comparison: Comparison) -> str:
    """Read a comparison against the non-inferiority margin.

    Args:
        comparison: Paired comparison (first minus second).

    Returns:
        Plain-language reading of the accuracy difference.

    Example:
        >>> verdict(Comparison(first="a", second="b", label="", pairs=1, accuracy_diff=0,
        ...     accuracy_diff_low=-0.01, accuracy_diff_high=0.01, only_first=0, only_second=0,
        ...     mcnemar_p=1, latency_diff_median=0, latency_diff_low=0, latency_diff_high=0,
        ...     sign_p=1))
        'no loss: whole CI above -3 points'
    """
    margin_points = int(NON_INFERIORITY_MARGIN * 100)
    if comparison.accuracy_diff_low > -NON_INFERIORITY_MARGIN:
        return f"no loss: whole CI above -{margin_points} points"
    if comparison.accuracy_diff_high < 0 and comparison.mcnemar_p < 0.05:
        return "worse (significant)"
    return f"inconclusive: CI reaches below -{margin_points} points"


def _pct(value: float) -> str:
    """Format a share as a percentage."""
    return f"{value * 100:.1f}%"


def _points(value: float) -> str:
    """Format a difference in percentage points."""
    return f"{value * 100:+.1f}"


def _setup_lines(summaries: list[SetupSummary]) -> list[str]:
    """Markdown table of per-setup headline numbers."""
    lines = [
        "| Setup | Model | Thinking | Calls | Accuracy (95% CI) | New 150 | Sept 35 "
        "| Latency median / p95 (s) | Tokens in / out / thinking | Parse fails | Split | Errors |",
        "|---|---|---|---:|---|---:|---:|---|---|---:|---:|---:|",
    ]
    for item in summaries:
        setup = SETUPS[item.setup]
        lines.append(
            f"| {item.setup} | {setup.model} | {setup.thinking_level} | {item.calls} "
            f"| {_pct(item.accuracy)} ({_pct(item.accuracy_low)}–{_pct(item.accuracy_high)}) "
            f"| {_pct(item.accuracy_new)} | {_pct(item.accuracy_sept35)} "
            f"| {item.latency_median:.2f} / {item.latency_p95:.2f} "
            f"| {item.input_tokens:,.0f} / {item.output_tokens:.1f} / {item.thinking_tokens:.1f} "
            f"| {item.parse_failures} | {item.splits} | {item.errors} |"
        )
    return lines


def _destination_lines(summaries: list[SetupSummary]) -> list[str]:
    """Markdown table of accuracy per expected destination."""
    lines = [
        "| Setup | " + " | ".join(DESTINATIONS) + " |",
        "|---|" + "---:|" * len(DESTINATIONS),
    ]
    for item in summaries:
        cells = " | ".join(_pct(item.by_destination[name]) for name in DESTINATIONS)
        lines.append(f"| {item.setup} | {cells} |")
    return lines


def _comparison_lines(comparisons: list[Comparison]) -> list[str]:
    """Markdown table of the paired comparisons."""
    lines = [
        "| Question | First − second | Pairs | Accuracy diff, points (95% CI) | Only first "
        "/ only second | McNemar p | Latency diff median, s (95% CI) | Sign p | Reading |",
        "|---|---|---:|---|---|---:|---|---:|---|",
    ]
    for item in comparisons:
        lines.append(
            f"| {item.label} | {item.first} − {item.second} | {item.pairs} "
            f"| {_points(item.accuracy_diff)} ({_points(item.accuracy_diff_low)} to "
            f"{_points(item.accuracy_diff_high)}) | {item.only_first} / {item.only_second} "
            f"| {item.mcnemar_p:.4f} | {item.latency_diff_median:+.2f} "
            f"({item.latency_diff_low:+.2f} to {item.latency_diff_high:+.2f}) "
            f"| {item.sign_p:.4f} | {verdict(item)} |"
        )
    return lines


def _wrong_lines(rows: list[RouterCallRow]) -> list[str]:
    """Markdown list of every wrong or failed call, for reading by hand."""
    wrong = sorted(
        (row for row in rows if not row.correct),
        key=lambda row: (row.question_id, row.setup, row.round_index),
    )
    lines = [
        "| Question | Setup | Round | Expected | Got | Confidence | Note |",
        "|---|---|---:|---|---|---:|---|",
    ]
    for row in wrong:
        note = row.error or ("unreadable reply" if row.parse_failed else "")
        lines.append(
            f"| {row.question_id} | {row.setup} | {row.round_index + 1} | {row.expected} "
            f"| {row.destination or '—'} | {row.confidence} | {note} |"
        )
    return lines


def build_markdown(
    summaries: list[SetupSummary], comparisons: list[Comparison], rows: list[RouterCallRow]
) -> str:
    """Assemble the full Markdown report.

    Args:
        summaries: Per-setup summaries.
        comparisons: Paired comparisons.
        rows: All rows (for the wrong-call list).

    Returns:
        Markdown text.
    """
    rounds = len({row.round_index for row in rows})
    questions = len({row.question_id for row in rows})
    margin = int(NON_INFERIORITY_MARGIN * 100)
    return "\n".join(
        [
            "# Router study: 2 models × 2 thinking levels",
            "",
            f"{questions} questions × {len(summaries)} setups × {rounds} round(s); "
            "production router prompt (few-shot, history, asset hint, 0.7 clarify threshold). "
            "Correct = the destination the user ends up in (general and unknown both open "
            "General Chat; confidence < 0.7 asks back). Latency = the model call only, same "
            "provider, shuffled order, measured from a laptop (not Render). "
            f"'No loss' = the whole 95% CI of the accuracy difference is above −{margin} points.",
            "",
            "## Setups",
            *_setup_lines(summaries),
            "",
            "## Accuracy by expected destination",
            *_destination_lines(summaries),
            "",
            "## Paired comparisons",
            *_comparison_lines(comparisons),
            "",
            "## Every wrong call",
            *_wrong_lines(rows),
            "",
        ]
    )


def rescore(rows: list[RouterCallRow], questions: list[RoutingQuestion]) -> list[RouterCallRow]:
    """Recompute correctness from the current labels (the log keeps raw predictions).

    Args:
        rows: Logged calls.
        questions: Questions with their accepted destinations.

    Returns:
        Copies of the rows with ``correct`` recomputed; unknown questions keep their value.
    """
    accepted = {question.question_id: accepted_destinations(question) for question in questions}
    rescored: list[RouterCallRow] = []
    for row in rows:
        if row.question_id in accepted and row.error is None:
            destinations = row.destinations or [row.destination]
            ok = not row.parse_failed and routed_correctly(destinations, accepted[row.question_id])
            row = row.model_copy(update={"correct": ok})
        rescored.append(row)
    return rescored


def write_report(log_path: Path, questions: list[RoutingQuestion] | None = None) -> Path:
    """Write report.md and summary.json next to the log.

    Args:
        log_path: The study's calls.jsonl.
        questions: Labels to score against (default: the study's question files).

    Returns:
        Path of report.md.
    """
    labels = load_all_questions() if questions is None else questions
    rows = rescore(read_log(log_path), labels)
    present = [name for name in SETUPS if any(row.setup == name for row in rows)]
    summaries = [summarize_setup(name, [r for r in rows if r.setup == name]) for name in present]
    comparisons = [
        compare(rows, first, second, label)
        for first, second, label in COMPARISONS
        if first in present and second in present
    ]
    folder = log_path.parent
    summary = {
        "setups": [item.model_dump() for item in summaries],
        "comparisons": [item.model_dump() for item in comparisons],
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = folder / "report.md"
    report_path.write_text(build_markdown(summaries, comparisons, rows), encoding="utf-8")
    return report_path
