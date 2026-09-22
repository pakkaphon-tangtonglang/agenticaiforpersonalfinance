"""Router feature ablation experiment runner.

Runs the router's ``classify_query`` over a grid of ablation variants,
models, and rounds, logging one JSONL row per call. The log is the
single source of truth: aggregation reads it back, and restarts skip
rows that already completed.

Example:
    >>> from finance_ai.evaluation.router_ablation import ABLATION_VARIANTS
    >>> sorted(ABLATION_VARIANTS)
    ['+few_shot', '+history', '+symbol', '+threshold', 'baseline', 'full']
"""

import json
import statistics
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from finance_ai.agents.router_agent import (
    DEFAULT_ABLATION_CONFIG,
    RouterAblationConfig,
    classify_query,
)
from finance_ai.evaluation.datasets import (
    load_routing_dataset,
    load_routing_history_dataset,
)

# One-at-a-time variants: each turns on exactly one helper over the
# baseline; "full" reproduces the production default config exactly.
_BASELINE = RouterAblationConfig(
    include_chat_history=False,
    include_few_shot_examples=False,
    resolve_asset_hint=False,
    apply_confidence_threshold=False,
)

ABLATION_VARIANTS: dict[str, RouterAblationConfig] = {
    "baseline": _BASELINE,
    "+few_shot": replace(_BASELINE, include_few_shot_examples=True),
    "+history": replace(_BASELINE, include_chat_history=True),
    "+symbol": replace(_BASELINE, resolve_asset_hint=True),
    "+threshold": replace(_BASELINE, apply_confidence_threshold=True),
    "full": DEFAULT_ABLATION_CONFIG,
}


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse a "provider:model" string into its two parts.

    Args:
        spec: Model specification, e.g. "ollama:minimax-m3".

    Returns:
        (provider, model_name) tuple.

    Raises:
        ValueError: If the spec is not "provider:model" with both parts
            non-empty.

    Example:
        >>> parse_model_spec("ollama:minimax-m3")
        ('ollama', 'minimax-m3')
    """
    provider, separator, model_name = spec.partition(":")
    if not separator or not provider or not model_name:
        raise ValueError(f"Invalid model spec '{spec}'. Expected 'provider:model'.")
    return provider, model_name


# ============================================================================
# Ablation Call Log
# ============================================================================

BASE_CASE_GROUP = "base"
HISTORY_CASE_GROUP = "history"


class AblationCallRow(BaseModel):
    """One logged ablation call (one JSONL line in the run log).

    Attributes:
        model: "provider:model" spec of the chat model used.
        variant: Ablation variant label, e.g. "baseline" or "+history".
        round: 1-based repetition index within the variant.
        case_id: Dataset case identifier.
        case_group: "base" (single-turn dataset) or "history" (multi-turn).
        query: The final user query.
        expected_intent: Ground-truth intent.
        expected_clarify: True when a "clarify" prediction is desired.
        predicted_intent: Router's predicted intent.
        predicted_confidence: Router confidence, Decimal serialized as str.
        latency_seconds: Wall-clock seconds for the router call.
        error: Exception message when the call failed, else None.
    """

    model: str
    variant: str
    round: int = Field(ge=1)  # noqa: A003
    case_id: str
    case_group: str
    query: str
    expected_intent: str
    expected_clarify: bool = False
    predicted_intent: str = "unknown"
    predicted_confidence: str = "0"
    latency_seconds: float = 0.0
    error: str | None = None


def _row_key(row: AblationCallRow) -> str:
    """Build the resume key for one logged call.

    Args:
        row: The logged row.

    Returns:
        Unique key "model|variant|round|case_id".
    """
    return f"{row.model}|{row.variant}|{row.round}|{row.case_id}"


def _load_ablation_cases(
    data_dir: str,
) -> list[tuple[str, str, str, str, bool, list[tuple[str, str]]]]:
    """Load base + history cases into uniform tuples.

    Args:
        data_dir: Directory holding routing_dataset.yaml and
            routing_history_dataset.yaml.

    Returns:
        Tuples of (case_id, query, expected_intent, case_group,
        expected_clarify, chat_history).

    Example:
        >>> cases = _load_ablation_cases("data/evaluation")  # doctest: +SKIP
    """
    base = load_routing_dataset(f"{data_dir}/routing_dataset.yaml")
    history = load_routing_history_dataset(f"{data_dir}/routing_history_dataset.yaml")
    rows: list[tuple[str, str, str, str, bool, list[tuple[str, str]]]] = [
        (c.case_id, c.query, str(c.expected_intent), BASE_CASE_GROUP, False, []) for c in base.cases
    ]
    rows.extend(
        (
            c.case_id,
            c.query,
            str(c.expected_intent),
            HISTORY_CASE_GROUP,
            c.expected_clarify,
            list(c.chat_history),
        )
        for c in history.cases
    )
    return rows


class AblationRunner:
    """Runs the variant x round x case grid and logs every call."""

    def __init__(self, models: dict[str, BaseChatModel], log_path: Path, data_dir: str) -> None:
        """Store models, log path, and dataset directory.

        Args:
            models: Chat models keyed by "provider:model" spec.
            log_path: JSONL file receiving one line per call.
            data_dir: Directory holding both routing datasets.
        """
        self._models = models
        self._log_path = Path(log_path)
        self._data_dir = data_dir

    def run(
        self,
        variants: dict[str, RouterAblationConfig] | None = None,
        rounds: int = 3,
    ) -> int:
        """Execute all (model, variant, round, case) calls not yet logged.

        Args:
            variants: Variant configs keyed by name (defaults to
                ABLATION_VARIANTS).
            rounds: Number of repetitions per grid cell.

        Returns:
            Number of newly executed calls (rows appended).
        """
        selected = variants if variants is not None else ABLATION_VARIANTS
        cases = _load_ablation_cases(self._data_dir)
        completed = self._completed_keys()
        executed = 0
        for model_spec, chat_model in self._models.items():
            for variant_name, config in selected.items():
                for round_number in range(1, rounds + 1):
                    executed += self._run_round(
                        model_spec,
                        chat_model,
                        variant_name,
                        config,
                        round_number,
                        cases,
                        completed,
                    )
        return executed

    def read_rows(self) -> list[AblationCallRow]:
        """Read every row currently in the log.

        Returns:
            Parsed AblationCallRow list.
        """
        if not self._log_path.exists():
            return []
        text = self._log_path.read_text(encoding="utf-8")
        return [AblationCallRow(**json.loads(line)) for line in text.splitlines() if line]

    def _completed_keys(self) -> set[str]:
        """Keys of rows that finished without an error.

        Returns:
            Set of resume keys; error rows are retried, not skipped.
        """
        return {_row_key(row) for row in self.read_rows() if row.error is None}

    def _run_round(  # noqa: PLR0913
        self,
        model_spec: str,
        chat_model: BaseChatModel,
        variant_name: str,
        config: RouterAblationConfig,
        round_number: int,
        cases: list[tuple[str, str, str, str, bool, list[tuple[str, str]]]],
        completed: set[str],
    ) -> int:
        """Run every case once for one (model, variant, round) cell.

        Args:
            model_spec: Model key, e.g. "ollama:minimax-m3".
            chat_model: The chat model instance.
            variant_name: Variant label.
            config: Ablation config for this variant.
            round_number: 1-based round index.
            cases: Uniform case tuples from _load_ablation_cases.
            completed: Resume keys already done.

        Returns:
            Number of calls executed in this round.
        """
        executed = 0
        for case_id, query, expected, group, expected_clarify, chat_history in cases:
            resume_key = f"{model_spec}|{variant_name}|{round_number}|{case_id}"
            if resume_key in completed:
                continue
            row = _execute_call(
                model_spec,
                chat_model,
                variant_name,
                config,
                round_number,
                case_id,
                query,
                expected,
                group,
                expected_clarify,
                chat_history,
            )
            self._append_row(row)
            executed += 1
        return executed

    def _append_row(self, row: AblationCallRow) -> None:
        """Append one row to the JSONL log, one line per call.

        Args:
            row: The row to persist immediately.
        """
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row.model_dump(), ensure_ascii=False) + "\n")


def _execute_call(  # noqa: PLR0913
    model_spec: str,
    chat_model: BaseChatModel,
    variant_name: str,
    config: RouterAblationConfig,
    round_number: int,
    case_id: str,
    query: str,
    expected: str,
    group: str,
    expected_clarify: bool,
    chat_history: list[tuple[str, str]],
) -> AblationCallRow:
    """Classify one case and build its log row, never raising.

    Args:
        model_spec: Model key for the log row.
        chat_model: Chat model to invoke.
        variant_name: Variant label for the log row.
        config: Ablation config controlling the call.
        round_number: 1-based round index.
        case_id: Case identifier.
        query: Final user query.
        expected: Expected intent.
        group: Case group ("base" or "history").
        expected_clarify: Whether clarify is the desired outcome.
        chat_history: Prior turns (empty for base cases).

    Returns:
        The row with either a prediction or an error message.
    """
    row = AblationCallRow(
        model=model_spec,
        variant=variant_name,
        round=round_number,
        case_id=case_id,
        case_group=group,
        query=query,
        expected_intent=expected,
        expected_clarify=expected_clarify,
    )
    start = time.perf_counter()
    try:
        history = chat_history if chat_history else None
        decision = classify_query(query, chat_model=chat_model, chat_history=history, config=config)
        row.predicted_intent = decision.intent
        row.predicted_confidence = str(decision.confidence)
    except Exception as exc:  # noqa: BLE001 - log and continue by design
        row.error = str(exc)
    row.latency_seconds = time.perf_counter() - start
    return row


# ============================================================================
# Aggregation and Report
# ============================================================================


class VariantSummary(BaseModel):
    """Aggregated metrics for one (model, variant) cell.

    Attributes:
        model: "provider:model" spec of the chat model.
        variant: Ablation variant label.
        rounds_completed: How many rounds produced valid rows.
        total_calls: All logged calls, including errors.
        error_count: Calls that raised an exception.
        base_accuracy_mean: Mean accuracy on the base 35-case group.
        base_accuracy_std: Sample std of base accuracy across rounds.
        history_accuracy_mean: Mean accuracy on the history 10-case group.
        history_accuracy_std: Sample std of history accuracy across rounds.
        clarify_rate_mean: Share of valid calls predicting "clarify".
        mean_latency_seconds: Mean wall-clock seconds per valid call.
    """

    model: str
    variant: str
    rounds_completed: int
    total_calls: int
    error_count: int
    base_accuracy_mean: float
    base_accuracy_std: float
    history_accuracy_mean: float
    history_accuracy_std: float
    clarify_rate_mean: float
    mean_latency_seconds: float


class RouterAblationReport(BaseModel):
    """Complete ablation report across models and variants.

    Attributes:
        report_id: Unique identifier used for the output file names.
        rounds: Planned number of rounds per grid cell.
        generated_at: UTC ISO timestamp of report creation.
        variants: One summary per (model, variant), in canonical order.
    """

    report_id: str
    rounds: int
    generated_at: str
    variants: list[VariantSummary]


def _is_correct(row: AblationCallRow) -> bool:
    """Apply the scoring rule to one row.

    Args:
        row: The logged call.

    Returns:
        True when the prediction matches and 'clarify' is expected.

    Example:
        >>> _is_correct(_row())  # doctest: +SKIP
    """
    if row.expected_clarify:
        return row.predicted_intent == "clarify"
    return row.predicted_intent == row.expected_intent


def _round_accuracies(rows: list[AblationCallRow], group: str, round_number: int) -> float | None:
    """Accuracy of one group in one round, None when no valid calls.

    Args:
        rows: Rows for a single (model, variant).
        group: "base" or "history".
        round_number: 1-based round index.

    Returns:
        Accuracy for that (group, round), or None.
    """
    valid = [
        row
        for row in rows
        if row.case_group == group and row.error is None and row.round == round_number
    ]
    if not valid:
        return None
    return sum(_is_correct(row) for row in valid) / len(valid)


def _group_stats(rows: list[AblationCallRow], group: str, rounds: int) -> tuple[float, float]:
    """Mean and std accuracy for one case group across rounds.

    Args:
        rows: Rows for a single (model, variant).
        group: "base" or "history".
        rounds: Planned round count.

    Returns:
        (mean, std) — std is 0.0 when fewer than two rounds completed.
    """
    per_round = [
        accuracy
        for r in range(1, rounds + 1)
        if (accuracy := _round_accuracies(rows, group, r)) is not None
    ]
    if not per_round:
        return 0.0, 0.0
    if len(per_round) < 2:
        return per_round[0], 0.0
    return float(statistics.mean(per_round)), float(statistics.stdev(per_round))


def _clarify_rate(rows: list[AblationCallRow]) -> float:
    """Share of valid calls that predicted 'clarify'.

    Args:
        rows: Rows for a single (model, variant).

    Returns:
        Clarify rate, 0.0 when there are no valid calls.
    """
    valid = [row for row in rows if row.error is None]
    if not valid:
        return 0.0
    return sum(row.predicted_intent == "clarify" for row in valid) / len(valid)


def _mean_latency(rows: list[AblationCallRow]) -> float:
    """Mean latency over valid calls.

    Args:
        rows: Rows for a single (model, variant).

    Returns:
        Mean seconds, 0.0 when there are no valid calls.
    """
    valid = [row for row in rows if row.error is None]
    if not valid:
        return 0.0
    return float(statistics.mean(row.latency_seconds for row in valid))


def _summarize_cell(
    model_spec: str, variant_name: str, cell_rows: list[AblationCallRow], rounds: int
) -> VariantSummary:
    """Build one VariantSummary from a (model, variant) cell's rows.

    Args:
        model_spec: Model key for the summary.
        variant_name: Variant label for the summary.
        cell_rows: All logged rows of this cell.
        rounds: Planned round count.

    Returns:
        The aggregated VariantSummary.
    """
    base_mean, base_std = _group_stats(cell_rows, BASE_CASE_GROUP, rounds)
    hist_mean, hist_std = _group_stats(cell_rows, HISTORY_CASE_GROUP, rounds)
    valid = [row for row in cell_rows if row.error is None]
    return VariantSummary(
        model=model_spec,
        variant=variant_name,
        rounds_completed=len({row.round for row in valid}),
        total_calls=len(cell_rows),
        error_count=len(cell_rows) - len(valid),
        base_accuracy_mean=base_mean,
        base_accuracy_std=base_std,
        history_accuracy_mean=hist_mean,
        history_accuracy_std=hist_std,
        clarify_rate_mean=_clarify_rate(cell_rows),
        mean_latency_seconds=_mean_latency(cell_rows),
    )


def aggregate_rows(rows: list[AblationCallRow], rounds: int) -> list[VariantSummary]:
    """Aggregate logged rows into per (model, variant) summaries.

    Args:
        rows: All rows from the run log.
        rounds: Planned number of rounds.

    Returns:
        One VariantSummary per (model, variant), sorted by model then
        the ABLATION_VARIANTS order.

    Example:
        >>> summaries = aggregate_rows(rows, rounds=3)  # doctest: +SKIP
    """
    grouped: dict[tuple[str, str], list[AblationCallRow]] = {}
    for row in rows:
        grouped.setdefault((row.model, row.variant), []).append(row)
    summaries = [
        _summarize_cell(model_spec, variant_name, cell_rows, rounds)
        for (model_spec, variant_name), cell_rows in grouped.items()
    ]
    variant_order = {name: i for i, name in enumerate(ABLATION_VARIANTS)}
    return sorted(summaries, key=lambda s: (s.model, variant_order.get(s.variant, 99)))


def format_ablation_markdown(report: RouterAblationReport) -> str:
    """Render the report as markdown tables (one per model).

    Args:
        report: The complete ablation report.

    Returns:
        Markdown string for the .md report file.

    Example:
        >>> text = format_ablation_markdown(report)  # doctest: +SKIP
    """
    lines = [
        "# Router Ablation Report",
        "",
        f"**Generated:** {report.generated_at} | **Rounds:** {report.rounds}",
        "",
    ]
    for model_spec in sorted({summary.model for summary in report.variants}):
        lines.append(f"## Model: {model_spec}")
        lines.append("")
        lines.append(
            "| Variant | base acc (mean ± std) "
            "| history acc (mean ± std) | clarify rate | latency (s) |"
        )
        lines.append("|---|---|---|---|---|")
        lines.extend(_model_table_lines(report, model_spec))
        lines.append("")
    return "\n".join(lines)


def _model_table_lines(report: RouterAblationReport, model_spec: str) -> list[str]:
    """Render the table rows of one model.

    Args:
        report: The complete ablation report.
        model_spec: Model key to render.

    Returns:
        Markdown table lines for the model's variants.
    """
    lines = []
    for summary in report.variants:
        if summary.model != model_spec:
            continue
        lines.append(
            f"| {summary.variant} "
            f"| {summary.base_accuracy_mean:.3f} ± {summary.base_accuracy_std:.3f} "
            f"| {summary.history_accuracy_mean:.3f} ± {summary.history_accuracy_std:.3f} "
            f"| {summary.clarify_rate_mean:.3f} "
            f"| {summary.mean_latency_seconds:.2f} |"
        )
    return lines


def save_ablation_report(report: RouterAblationReport, output_dir: str) -> tuple[str, str]:
    """Write the report as JSON and Markdown files.

    Args:
        report: The complete ablation report.
        output_dir: Directory for the report files.

    Returns:
        (json_path, markdown_path) of the written files.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{report.report_id}.json"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path = directory / f"{report.report_id}.md"
    md_path.write_text(format_ablation_markdown(report), encoding="utf-8")
    return str(json_path), str(md_path)


def build_ablation_report(rows: list[AblationCallRow], rounds: int) -> RouterAblationReport:
    """Build a complete report from logged rows.

    Args:
        rows: All rows from the run log.
        rounds: Planned number of rounds.

    Returns:
        RouterAblationReport ready to save.
    """
    return RouterAblationReport(
        report_id=str(uuid.uuid4()),
        rounds=rounds,
        generated_at=datetime.now(timezone.utc).isoformat(),
        variants=aggregate_rows(rows, rounds),
    )
