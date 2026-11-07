"""Summarise a study log into JSON, CSV and a readable Markdown report.

Outputs (next to the log): ``summary.json``, ``runs.csv``, ``report.md``.
"""

import csv
from collections import Counter
from pathlib import Path
from statistics import mean

from pydantic import BaseModel

from finance_ai.evaluation.architectures.audit import (
    AUDIT_NAME,
    AuditVerdict,
    apply_audit,
    load_audit,
)
from finance_ai.evaluation.architectures.runner import StudyLogRow, read_log
from finance_ai.evaluation.architectures.scoring import score_run
from finance_ai.evaluation.architectures.task_models import ArchitectureTask
from finance_ai.evaluation.architectures.stats import (
    bootstrap_median_interval,
    exact_binomial_p,
    pass_hat_k,
    percentile,
    wilson_interval,
)

BASELINE_DESIGN = "hub_spoke"
CATEGORIES = ("single", "cross", "vague")


class DesignSummary(BaseModel):
    """Aggregates for one design."""

    design: str
    runs: int
    success_rate: float
    success_ci: tuple[float, float]
    success_by_category: dict[str, float]
    route_accuracy: float | None
    pass_k: float
    mean_llm_calls: float
    mean_input_tokens: float
    mean_output_tokens: float
    latency_median: float
    latency_p95: float
    errors: int
    failures: dict[str, int]


class PairedComparison(BaseModel):
    """One design against the hub-and-spoke baseline on the same (task, round) pairs."""

    design: str
    pairs: int
    design_only_success: int
    baseline_only_success: int
    mcnemar_p: float
    latency_median_difference: float
    latency_difference_ci: tuple[float, float]
    latency_sign_test_p: float


class StudyReport(BaseModel):
    """Everything the Markdown report shows."""

    model_label: str
    rounds: int
    tasks: int
    designs: list[DesignSummary]
    comparisons: list[PairedComparison]
    audit_notes: list[str] = []


def rescore(rows: list[StudyLogRow], tasks: dict[str, ArchitectureTask]) -> list[StudyLogRow]:
    """Recompute every score from the logged facts with the current scoring rules.

    Args:
        rows: Log rows.
        tasks: task_id → task definition.

    Returns:
        Rows with fresh scores (rows of unknown tasks are dropped).
    """
    return [
        row.model_copy(update={"score": score_run(tasks[row.run.task_id], row.run, row.db_results)})
        for row in rows
        if row.run.task_id in tasks
    ]


def summarize(
    rows: list[StudyLogRow],
    tasks: dict[str, ArchitectureTask],
    verdicts: list[AuditVerdict] | None = None,
) -> StudyReport:
    """Build the study report: rescore, apply manual verdicts, aggregate.

    Args:
        rows: Log rows (one per run).
        tasks: task_id → task definition.
        verdicts: Manual audit verdicts (optional).

    Returns:
        StudyReport with per-design summaries, paired comparisons and audit notes.
    """
    verdicts = verdicts or []
    rows = apply_audit(rescore(rows, tasks), verdicts)
    task_categories: dict[str, str] = {task_id: task.category for task_id, task in tasks.items()}
    rounds = max((row.run.round_index for row in rows), default=-1) + 1
    designs = sorted({row.run.design for row in rows})
    summaries = [
        _summarize_design(
            design, [r for r in rows if r.run.design == design], task_categories, rounds
        )
        for design in designs
    ]
    comparisons = [
        _compare(design, rows)
        for design in designs
        if design != BASELINE_DESIGN and BASELINE_DESIGN in designs
    ]
    return StudyReport(
        model_label=rows[0].model_label if rows else "",
        rounds=rounds,
        tasks=len({row.run.task_id for row in rows}),
        designs=summaries,
        comparisons=comparisons,
        audit_notes=[
            f"{v.task_id} / {v.design} / round {v.round_index}: "
            f"{'pass' if v.success else 'fail'} — {v.reason}"
            for v in verdicts
        ],
    )


def _summarize_design(
    design: str, rows: list[StudyLogRow], task_categories: dict[str, str], rounds: int
) -> DesignSummary:
    """Aggregate one design's rows."""
    successes = sum(row.score.success for row in rows)
    latencies = [row.run.latency_seconds for row in rows]
    routed = [row.score.route_ok for row in rows if row.score.route_ok is not None]
    return DesignSummary(
        design=design,
        runs=len(rows),
        success_rate=successes / len(rows),
        success_ci=wilson_interval(successes, len(rows)),
        success_by_category=_success_by_category(rows, task_categories),
        route_accuracy=sum(routed) / len(routed) if routed else None,
        pass_k=pass_hat_k([(row.run.task_id, row.score.success) for row in rows], rounds),
        mean_llm_calls=mean(row.run.metrics.llm_calls for row in rows),
        mean_input_tokens=mean(row.run.metrics.input_tokens for row in rows),
        mean_output_tokens=mean(row.run.metrics.output_tokens for row in rows),
        latency_median=percentile(latencies, 0.5),
        latency_p95=percentile(latencies, 0.95),
        errors=sum(row.run.error is not None for row in rows),
        failures=dict(Counter(row.score.failure for row in rows if row.score.failure)),
    )


def _success_by_category(
    rows: list[StudyLogRow], task_categories: dict[str, str]
) -> dict[str, float]:
    """Return the success rate per task category (categories with runs only)."""
    result: dict[str, float] = {}
    for category in CATEGORIES:
        picked = [
            row.score.success for row in rows if task_categories.get(row.run.task_id) == category
        ]
        if picked:
            result[category] = sum(picked) / len(picked)
    return result


def _compare(design: str, rows: list[StudyLogRow]) -> PairedComparison:
    """Pair a design with the baseline on identical (task, round) keys."""
    by_key: dict[tuple[str, str, int], StudyLogRow] = {
        (r.run.design, r.run.task_id, r.run.round_index): r for r in rows
    }
    keys = [(t, n) for (d, t, n) in by_key if d == design and (BASELINE_DESIGN, t, n) in by_key]
    pairs = [(by_key[(design, t, n)], by_key[(BASELINE_DESIGN, t, n)]) for t, n in keys]
    design_only = sum(a.score.success and not b.score.success for a, b in pairs)
    baseline_only = sum(b.score.success and not a.score.success for a, b in pairs)
    differences = [a.run.latency_seconds - b.run.latency_seconds for a, b in pairs]
    return PairedComparison(
        design=design,
        pairs=len(pairs),
        design_only_success=design_only,
        baseline_only_success=baseline_only,
        mcnemar_p=exact_binomial_p(design_only, baseline_only),
        latency_median_difference=percentile(differences, 0.5),
        latency_difference_ci=bootstrap_median_interval(differences),
        latency_sign_test_p=exact_binomial_p(
            sum(d > 0 for d in differences), sum(d < 0 for d in differences)
        ),
    )


def write_report(log_path: Path, tasks: dict[str, ArchitectureTask]) -> Path:
    """Write summary.json, runs.csv and report.md next to the log.

    Manual verdicts are read from ``audit.yaml`` in the same folder, if present.

    Args:
        log_path: Study log (JSONL).
        tasks: task_id → task definition (used to rescore).

    Returns:
        Path of report.md.
    """
    verdicts = load_audit(log_path.parent / AUDIT_NAME)
    logged = read_log(log_path)
    rows = apply_audit(rescore(logged, tasks), verdicts)
    report = summarize(logged, tasks, verdicts)
    folder = log_path.parent
    (folder / "summary.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    _write_runs_csv(rows, folder / "runs.csv")
    report_path = folder / "report.md"
    report_path.write_text(format_markdown(report), encoding="utf-8")
    return report_path


def _write_runs_csv(rows: list[StudyLogRow], path: Path) -> None:
    """Write one CSV line per run (for charts in the thesis)."""
    fields = [
        "task_id",
        "design",
        "round",
        "success",
        "failure",
        "route",
        "llm_calls",
        "input_tokens",
        "output_tokens",
        "latency_seconds",
        "tools",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for row in rows:
            run = row.run
            writer.writerow(
                [
                    run.task_id,
                    run.design,
                    run.round_index,
                    row.score.success,
                    row.score.failure or "",
                    ">".join(run.route),
                    run.metrics.llm_calls,
                    run.metrics.input_tokens,
                    run.metrics.output_tokens,
                    f"{run.latency_seconds:.3f}",
                    " ".join(run.metrics.tool_names()),
                    run.error or "",
                ]
            )


def format_markdown(report: StudyReport) -> str:
    """Render the report as Markdown.

    Args:
        report: Study report.

    Returns:
        Markdown text with the main table, categories, failures and paired tests.
    """
    sections = [
        f"# Architecture study — {report.model_label}",
        f"{report.tasks} tasks × {len(report.designs)} designs × {report.rounds} rounds. "
        "Success = right route + required tools + expected values + expected database rows "
        "(vague tasks: asked back without writing). 95 % Wilson CI.",
        _main_table(report.designs),
        _category_table(report.designs),
        _failure_table(report.designs),
        _comparison_table(report.comparisons),
        _audit_section(report.audit_notes),
    ]
    return "\n\n".join(sections) + "\n"


def _audit_section(notes: list[str]) -> str:
    """List the manual audit verdicts applied to this report."""
    if not notes:
        return "## Manual audit overrides\nNone recorded in audit.yaml."
    lines = ["## Manual audit overrides", "Verdicts from audit.yaml (applied above):"]
    return "\n".join(lines + [f"- {note}" for note in notes])


def _percent(value: float | None) -> str:
    """Format a rate as a percentage ('—' for None)."""
    return "—" if value is None else f"{value * 100:.1f}%"


def _main_table(designs: list[DesignSummary]) -> str:
    """Main results table."""
    lines = [
        "## Main results",
        "| Design | Runs | Success (95% CI) | Route acc. | pass^k | LLM calls | Tokens in / out "
        "| Latency median / p95 (s) | Errors |",
        "|---|---:|---|---:|---:|---:|---|---|---:|",
    ]
    for d in designs:
        success = (
            f"{_percent(d.success_rate)} ({_percent(d.success_ci[0])}–{_percent(d.success_ci[1])})"
        )
        tokens = f"{d.mean_input_tokens:,.0f} / {d.mean_output_tokens:,.0f}"
        latency = f"{d.latency_median:.1f} / {d.latency_p95:.1f}"
        lines.append(
            f"| {d.design} | {d.runs} | {success} | {_percent(d.route_accuracy)} "
            f"| {_percent(d.pass_k)} | {d.mean_llm_calls:.2f} | {tokens} | {latency} | {d.errors} |"
        )
    return "\n".join(lines)


def _category_table(designs: list[DesignSummary]) -> str:
    """Success rate per task category."""
    lines = [
        "## Success by task type",
        "| Design | " + " | ".join(CATEGORIES) + " |",
        "|---|" + "---:|" * len(CATEGORIES),
    ]
    for d in designs:
        cells = " | ".join(_percent(d.success_by_category.get(c)) for c in CATEGORIES)
        lines.append(f"| {d.design} | {cells} |")
    return "\n".join(lines)


def _failure_table(designs: list[DesignSummary]) -> str:
    """Count of each failure type per design."""
    kinds = sorted({kind for d in designs for kind in d.failures})
    if not kinds:
        return "## Failure types\nNo failures."
    lines = [
        "## Failure types (runs)",
        "| Design | " + " | ".join(kinds) + " |",
        "|---|" + "---:|" * len(kinds),
    ]
    for d in designs:
        lines.append(
            f"| {d.design} | " + " | ".join(str(d.failures.get(k, 0)) for k in kinds) + " |"
        )
    return "\n".join(lines)


def _comparison_table(comparisons: list[PairedComparison]) -> str:
    """Paired tests against hub-and-spoke."""
    if not comparisons:
        return "## Paired tests vs hub_spoke\nBaseline not in this run."
    lines = [
        "## Paired tests vs hub_spoke (same task and round)",
        "| Design | Pairs | Only design succeeded | Only hub succeeded | McNemar p "
        "| Latency diff median (95% CI) s | Sign test p |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for c in comparisons:
        low, high = c.latency_difference_ci
        lines.append(
            f"| {c.design} | {c.pairs} | {c.design_only_success} | {c.baseline_only_success} "
            f"| {c.mcnemar_p:.4f} | {c.latency_median_difference:+.1f} ({low:+.1f} to {high:+.1f}) "
            f"| {c.latency_sign_test_p:.4f} |"
        )
    return "\n".join(lines)
