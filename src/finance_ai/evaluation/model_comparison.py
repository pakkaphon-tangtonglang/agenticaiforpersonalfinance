"""Multi-model, multi-dimension comparison evaluation for the thesis.

Runs one or more evaluation dimensions across candidate models:
the production baseline plus alternatives from Ollama Cloud and
Google Gemini. Supported dimensions:

- routing: intent classification accuracy + latency
- tax-accuracy: tax-answer correctness + MAE + latency
- tax-accuracy-forced: same, with the tax tool forced (isolates
  tool-argument accuracy from tool-selection accuracy)
- quality: LLM-as-judge response quality (fixed judge model, mean
  judge score 1-5 normalized to 0-1)
- hallucination: anti-hallucination compliance rate
- recommendation-safety: recommendation guardrail compliance rate

The thesis can then justify the production model choice with
measured numbers instead of anecdote. RAG retrieval is excluded
because it evaluates the vector store, not the model.

Every (model, dimension) pair is an independent I/O-bound job, so
`run_multi_dimension_comparison` can execute all of them
concurrently in one shared thread pool.

Models whose credentials are missing (e.g. Gemini before
GOOGLE_API_KEY is configured) are reported as 'skipped' rather than
aborting the whole comparison.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from finance_ai.core.logging import get_logger
from finance_ai.evaluation.models import (
    ModelComparisonEntry,
    ModelComparisonResult,
)

logger = get_logger(__name__)

ModelFactory = Callable[["ModelSpec"], Any]
RunnerFactory = Callable[[Any, "ModelSpec"], Any]

VALID_COMPARISON_DIMENSIONS = (
    "routing",
    "tax-accuracy",
    "tax-accuracy-forced",
    "quality",
    "hallucination",
    "recommendation-safety",
)

# Job = (candidate model, dimension to evaluate it on).
ComparisonJob = tuple["ModelSpec", str]


class ModelSpec(BaseModel):
    """A candidate model to compare.

    Attributes:
        provider: LLM provider (ollama, google, openrouter).
        model_name: Model identifier as passed to the provider.

    Example:
        >>> ModelSpec(provider="ollama", model_name="minimax-m3")
    """

    provider: str
    model_name: str


# Production baseline first, then open-model flagships from Ollama
# Cloud, then Gemini candidates (skipped until GOOGLE_API_KEY is set).
DEFAULT_COMPARISON_MODELS: list[ModelSpec] = [
    ModelSpec(provider="ollama", model_name="minimax-m3"),
    ModelSpec(provider="ollama", model_name="qwen3.5:397b"),
    ModelSpec(provider="ollama", model_name="deepseek-v4-pro:0813"),
    ModelSpec(provider="ollama", model_name="glm-5.3"),
    ModelSpec(provider="ollama", model_name="glm-5.3-flash"),
    ModelSpec(provider="google", model_name="gemini-3.5-flash-lite"),
    ModelSpec(provider="google", model_name="gemini-3.5-flash"),
]


def parse_model_spec(text: str) -> ModelSpec:
    """Parse a 'provider:model' CLI string into a ModelSpec.

    Splits on the FIRST colon only, because Ollama model names
    themselves contain colons (e.g. 'qwen3.5:397b').

    Args:
        text: String like 'ollama:qwen3.5:397b' or 'google:gemini-3.5-flash'.

    Returns:
        Parsed ModelSpec.

    Raises:
        ValueError: If the string has no provider separator.

    Example:
        >>> parse_model_spec("ollama:qwen3.5:397b").model_name
        'qwen3.5:397b'
    """
    provider, separator, model_name = text.partition(":")
    if not separator or not provider or not model_name:
        raise ValueError(
            f"Invalid model spec '{text}'. Expected provider:model "
            "(e.g. ollama:minimax-m3, google:gemini-3.5-flash)."
        )
    return ModelSpec(provider=provider, model_name=model_name)


def run_model_comparison(
    specs: list[ModelSpec],
    data_dir: str = "data/evaluation",
    model_factory: ModelFactory | None = None,
    runner_factory: RunnerFactory | None = None,
    max_workers: int = 1,
    dimension: str = "routing",
) -> ModelComparisonResult:
    """Run one dimension for every candidate model.

    Args:
        specs: Candidate models to evaluate, in run order.
        data_dir: Evaluation dataset directory.
        model_factory: Callable building a chat model from a spec
            (defaults to the provider-aware factory in the CLI module).
        runner_factory: Callable building an EvaluationRunner from
            (model, spec); defaults to the real runner.
        max_workers: When >1, candidates are evaluated concurrently
            (LLM calls are HTTP I/O-bound); results keep spec order.
        dimension: 'routing' or 'tax-accuracy'.

    Returns:
        ModelComparisonResult with one entry per candidate.

    Raises:
        ValueError: If dimension is not a supported comparison dimension.
    """
    results = run_multi_dimension_comparison(
        specs,
        [dimension],
        data_dir=data_dir,
        model_factory=model_factory,
        runner_factory=runner_factory,
        max_workers=max_workers,
    )
    return results[0]


def run_multi_dimension_comparison(
    specs: list[ModelSpec],
    dimensions: list[str],
    data_dir: str = "data/evaluation",
    model_factory: ModelFactory | None = None,
    runner_factory: RunnerFactory | None = None,
    max_workers: int = 1,
    judge_spec: ModelSpec | None = None,
) -> list[ModelComparisonResult]:
    """Run every (model, dimension) pair, optionally all at once.

    Args:
        specs: Candidate models to evaluate.
        dimensions: Comparison dimensions to run for every model.
        data_dir: Evaluation dataset directory.
        model_factory: Optional injected model factory (tests).
        runner_factory: Optional injected runner factory (tests).
        max_workers: When >1, all model x dimension pairs are
            evaluated concurrently in one shared thread pool.
        judge_spec: Optional fixed judge model for LLM-as-judge
            dimensions (quality). The same judge scores every
            candidate, so comparisons stay fair.

    Returns:
        One ModelComparisonResult per requested dimension, in the
        requested dimension order.

    Raises:
        ValueError: If any dimension is not supported.
    """
    _validate_dimensions(dimensions)
    judge_model = _build_judge_model(judge_spec, model_factory)
    jobs: list[ComparisonJob] = [(spec, dimension) for dimension in dimensions for spec in specs]
    if max_workers > 1:
        pairs = _run_jobs_concurrently(
            jobs, data_dir, model_factory, runner_factory, max_workers, judge_model
        )
    else:
        pairs = [
            _evaluate_job(job, data_dir, model_factory, runner_factory, judge_model) for job in jobs
        ]
    return [_assemble_dimension_result(dimension, pairs) for dimension in dimensions]


def _build_judge_model(
    judge_spec: ModelSpec | None,
    model_factory: ModelFactory | None,
) -> Any:
    """Build the shared judge model once per comparison run.

    Args:
        judge_spec: Judge model spec, or None to run without a judge.
        model_factory: Model factory (defaults to the CLI factory).

    Returns:
        A chat model, or None when no judge was requested or it
        could not be built (quality jobs then report 'skipped').
    """
    if judge_spec is None:
        return None
    factory = model_factory or _default_model_factory
    try:
        return factory(judge_spec)
    except ValueError as exc:
        logger.warning("Judge model unavailable, LLM-judge dims skip: %s", exc)
        return None


def _validate_dimensions(dimensions: list[str]) -> None:
    """Reject unsupported dimension names with an actionable message.

    Args:
        dimensions: Requested dimension names.

    Raises:
        ValueError: If any name is not a supported dimension.
    """
    invalid = [name for name in dimensions if name not in VALID_COMPARISON_DIMENSIONS]
    if invalid:
        raise ValueError(
            f"Unsupported comparison dimension(s): {', '.join(invalid)}. "
            f"Expected one of: {', '.join(VALID_COMPARISON_DIMENSIONS)}."
        )


def _assemble_dimension_result(
    dimension: str,
    pairs: list[tuple[str, ModelComparisonEntry]],
) -> ModelComparisonResult:
    """Group evaluated jobs into one result per dimension.

    Args:
        dimension: Dimension to assemble.
        pairs: (dimension, entry) pairs from every evaluated job.

    Returns:
        A result holding that dimension's entries in job order.
    """
    entries = [entry for dim, entry in pairs if dim == dimension]
    return ModelComparisonResult(
        dimension=dimension,
        generated_at=datetime.now(timezone.utc),
        entries=entries,
    )


def _run_jobs_concurrently(
    jobs: list[ComparisonJob],
    data_dir: str,
    model_factory: ModelFactory | None,
    runner_factory: RunnerFactory | None,
    max_workers: int,
    judge_model: Any,
) -> list[tuple[str, ModelComparisonEntry]]:
    """Evaluate all jobs concurrently, preserving submission order.

    Args:
        jobs: (model, dimension) pairs to evaluate.
        data_dir: Evaluation dataset directory.
        model_factory: Optional injected model factory.
        runner_factory: Optional injected runner factory.
        max_workers: Thread pool size.
        judge_model: Shared judge model (None to run without one).

    Returns:
        (dimension, entry) pairs in the same order as jobs.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _evaluate_job, job, data_dir, model_factory, runner_factory, judge_model
            )
            for job in jobs
        ]
        return [future.result() for future in futures]


def _evaluate_job(
    job: ComparisonJob,
    data_dir: str,
    model_factory: ModelFactory | None,
    runner_factory: RunnerFactory | None,
    judge_model: Any,
) -> tuple[str, ModelComparisonEntry]:
    """Evaluate one (model, dimension) job.

    Args:
        job: Candidate model and dimension to evaluate.
        data_dir: Evaluation dataset directory.
        model_factory: Optional injected model factory.
        runner_factory: Optional injected runner factory.
        judge_model: Shared judge model (None to run without one).

    Returns:
        The evaluated dimension and its comparison entry.
    """
    spec, dimension = job
    entry = _compare_single_model(
        spec, data_dir, model_factory, runner_factory, dimension, judge_model
    )
    return dimension, entry


def _compare_single_model(
    spec: ModelSpec,
    data_dir: str,
    model_factory: ModelFactory | None,
    runner_factory: RunnerFactory | None,
    dimension: str,
    judge_model: Any,
) -> ModelComparisonEntry:
    """Evaluate one candidate model on one dimension.

    Args:
        spec: Candidate model.
        data_dir: Evaluation dataset directory.
        model_factory: Optional injected model factory.
        runner_factory: Optional injected runner factory.
        dimension: Comparison dimension to run.
        judge_model: Shared judge model (None to run without one).

    Returns:
        Entry with dimension metrics, or a skipped/error status.
    """
    logger.info("Comparing %s:%s on %s", spec.provider, spec.model_name, dimension)
    try:
        factory = model_factory or _default_model_factory
        model = factory(spec)
        runner = _build_runner(model, spec, data_dir, runner_factory, judge_model)
        aggregate = _run_dimension(runner, dimension)
    except ValidationError as exc:
        # Dataset/config problems are real failures — not missing
        # credentials (ValidationError subclasses ValueError).
        return _entry_with_error(spec, str(exc))
    except ValueError as exc:
        return _skipped_entry(spec, exc)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return _entry_with_error(spec, str(exc))
    return _ok_entry(spec, aggregate, dimension)


def _entry_with_error(spec: ModelSpec, message: str) -> ModelComparisonEntry:
    """Build an error entry for a model that failed to evaluate.

    Args:
        spec: Candidate model.
        message: Failure reason.

    Returns:
        Entry with status 'error'.
    """
    return ModelComparisonEntry(
        provider=spec.provider,
        model_name=spec.model_name,
        status="error",
        error_message=message,
    )


def _run_dimension(runner: Any, dimension: str) -> Any:
    """Run the requested dimension on an evaluation runner.

    Args:
        runner: EvaluationRunner (or test double).
        dimension: Comparison dimension to run.

    Returns:
        The aggregate result for that dimension.
    """
    method_by_dimension = {
        "routing": runner.run_routing,
        "tax-accuracy": runner.run_tax_accuracy,
        "tax-accuracy-forced": runner.run_tax_accuracy_forced,
        "quality": runner.run_quality,
        "hallucination": runner.run_hallucination,
        "recommendation-safety": runner.run_recommendation_safety,
    }
    return method_by_dimension[dimension]()


def _ok_entry(
    spec: ModelSpec,
    aggregate: Any,
    dimension: str,
) -> ModelComparisonEntry:
    """Build a successful entry from a dimension aggregate result.

    Args:
        spec: Candidate model.
        aggregate: Aggregated metrics for the dimension.
        dimension: Comparison dimension that was run.

    Returns:
        Entry with status 'ok' and the dimension metrics.
    """
    return ModelComparisonEntry(
        provider=spec.provider,
        model_name=spec.model_name,
        status="ok",
        per_case_results=_extract_per_case_results(aggregate),
        **_extract_metrics(aggregate, dimension),
    )


def _extract_per_case_results(aggregate: Any) -> list[dict[str, Any]]:
    """Pull per-case detail rows out of a dimension aggregate.

    Aggregates that keep per-case results (quality, hallucination,
    recommendation safety) expose them as a 'results' list of pydantic
    models; the rows are serialized so the comparison JSON preserves the
    agent responses for offline re-judging. Aggregates without detail
    rows (routing, tax) yield an empty list.

    Args:
        aggregate: Aggregated metrics for the dimension.

    Returns:
        List of serialized per-case rows; empty when unavailable.
    """
    raw = getattr(aggregate, "results", None)
    if not isinstance(raw, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, BaseModel):
            rows.append(item.model_dump(mode="json"))
        elif isinstance(item, dict):
            rows.append(item)
    return rows


def _extract_metrics(aggregate: Any, dimension: str) -> dict[str, Any]:
    """Normalize a dimension aggregate into entry metric kwargs.

    Args:
        aggregate: Aggregated metrics for the dimension.
        dimension: Comparison dimension that was run.

    Returns:
        kwargs for ModelComparisonEntry (accuracy, latency, counts).
    """
    if dimension == "routing":
        return _routing_metrics(aggregate)
    if dimension in ("tax-accuracy", "tax-accuracy-forced"):
        return _tax_metrics(aggregate)
    if dimension == "quality":
        return _quality_metrics(aggregate)
    if dimension == "hallucination":
        return _hallucination_metrics(aggregate)
    return _recommendation_safety_metrics(aggregate)


def _routing_metrics(aggregate: Any) -> dict[str, Any]:
    """Routing aggregate -> entry kwargs (accuracy + correct count)."""
    return {
        "accuracy": aggregate.accuracy,
        "mean_latency_seconds": aggregate.mean_latency_seconds,
        "total_cases": aggregate.total_cases,
        "correct_count": aggregate.correct_count,
    }


def _tax_metrics(aggregate: Any) -> dict[str, Any]:
    """Tax aggregate -> entry kwargs (accuracy rate + MAE)."""
    return {
        "accuracy": aggregate.accuracy_rate,
        "mean_latency_seconds": aggregate.mean_latency_seconds,
        "total_cases": aggregate.total_cases,
        "correct_count": aggregate.within_tolerance_count,
        "mean_absolute_error_thb": aggregate.mean_absolute_error_thb,
    }


def _quality_metrics(aggregate: Any) -> dict[str, Any]:
    """Quality aggregate -> entry kwargs (judge mean 1-5 -> 0-1)."""
    return {
        "accuracy": aggregate.mean_overall / Decimal("5"),
        "mean_latency_seconds": aggregate.mean_latency_seconds,
        "total_cases": aggregate.total_cases,
    }


def _hallucination_metrics(aggregate: Any) -> dict[str, Any]:
    """Hallucination aggregate -> entry kwargs (compliance rate)."""
    return {
        "accuracy": aggregate.compliance_rate,
        "mean_latency_seconds": aggregate.mean_latency_seconds,
        "total_cases": aggregate.total_cases,
        "correct_count": aggregate.compliant_count,
    }


def _recommendation_safety_metrics(aggregate: Any) -> dict[str, Any]:
    """Recommendation safety aggregate -> entry kwargs (compliance)."""
    return {
        "accuracy": aggregate.compliance_rate,
        "total_cases": aggregate.total_cases,
        "correct_count": aggregate.passed_count,
    }


def _default_model_factory(spec: ModelSpec) -> Any:
    """Build a chat model for a spec using the CLI settings builder.

    Args:
        spec: Candidate model.

    Returns:
        Configured LangChain chat model.
    """
    from finance_ai.agents.llm_factory import create_chat_model  # noqa: PLC0415
    from finance_ai.evaluation.cli import _build_settings  # noqa: PLC0415

    settings = _build_settings(spec.provider, spec.model_name)
    return create_chat_model(settings=settings)


def _build_runner(
    model: Any,
    spec: ModelSpec,
    data_dir: str,
    runner_factory: RunnerFactory | None,
    judge_model: Any,
) -> Any:
    """Build an EvaluationRunner for the given model.

    Args:
        model: Chat model under evaluation.
        spec: Candidate model spec.
        data_dir: Evaluation dataset directory.
        runner_factory: Optional injected factory (tests).
        judge_model: Shared judge model for LLM-judge dimensions.

    Returns:
        An EvaluationRunner (or test double).
    """
    if runner_factory is not None:
        return runner_factory(model, spec)
    from finance_ai.evaluation.runner import EvaluationRunner  # noqa: PLC0415

    return EvaluationRunner(
        chat_model=model,
        vector_store=None,
        llm_provider=spec.provider,
        llm_model=spec.model_name,
        judge_model=judge_model,
        data_dir=data_dir,
    )


def _skipped_entry(spec: ModelSpec, exc: ValueError) -> ModelComparisonEntry:
    """Build a skipped entry for a model with missing configuration.

    Args:
        spec: Candidate model.
        exc: The ValueError raised by the model factory.

    Returns:
        Entry with status 'skipped' and the reason.
    """
    return ModelComparisonEntry(
        provider=spec.provider,
        model_name=spec.model_name,
        status="skipped",
        error_message=str(exc),
    )


def format_comparison_markdown(result: ModelComparisonResult) -> str:
    """Render a comparison result as a markdown table for the thesis.

    Args:
        result: Comparison result from run_model_comparison.

    Returns:
        Markdown string with one row per model, sorted by accuracy.
        An extra 'MAE (THB)' column appears for accuracy dimensions.

    Example:
        >>> print(format_comparison_markdown(result))  # doctest: +SKIP
    """
    include_mae = any(entry.mean_absolute_error_thb is not None for entry in result.entries)
    header = _table_header(include_mae)
    ranked = sorted(result.entries, key=_accuracy_sort_key, reverse=True)
    rows = [_format_entry_row(entry, include_mae) for entry in ranked]
    notes = _format_skipped_notes(ranked)
    return header + "\n".join(rows) + "\n" + notes


def _table_header(include_mae: bool) -> str:
    """Build the markdown table header for a comparison.

    Args:
        include_mae: Whether to add the MAE (THB) column.

    Returns:
        Two-line markdown header (columns + separator row).
    """
    columns = "| Model | Provider | Status | Accuracy | Mean Latency (s) |"
    separator = "|---|---|---|---|---|"
    if include_mae:
        columns += " MAE (THB) |"
        separator += "---|"
    return f"{columns}\n{separator}\n"


def _accuracy_sort_key(entry: ModelComparisonEntry) -> Decimal:
    """Sort key ranking successful entries by accuracy (others last).

    Args:
        entry: Comparison entry.

    Returns:
        Accuracy, or -1 when the model did not run.
    """
    return entry.accuracy if entry.accuracy is not None else Decimal(-1)


def _format_entry_row(entry: ModelComparisonEntry, include_mae: bool) -> str:
    """Format one comparison entry as a markdown table row.

    Args:
        entry: Comparison entry.
        include_mae: Whether to append the MAE (THB) column.

    Returns:
        Markdown row string.
    """
    accuracy = f"{entry.accuracy:.2f}" if entry.accuracy is not None else "-"
    latency = f"{entry.mean_latency_seconds:.2f}" if entry.mean_latency_seconds is not None else "-"
    row = f"| {entry.model_name} | {entry.provider} | {entry.status} | {accuracy} | {latency} |"
    if include_mae:
        mae = (
            f"{entry.mean_absolute_error_thb:,.2f}"
            if entry.mean_absolute_error_thb is not None
            else "-"
        )
        row += f" {mae} |"
    return row


def _format_skipped_notes(entries: list[ModelComparisonEntry]) -> str:
    """Collect skip/error reasons as markdown footnotes.

    Long messages (e.g. multi-field pydantic errors) are truncated to
    their first line so the table stays readable.

    Args:
        entries: Comparison entries.

    Returns:
        Footnote lines (empty string when everything ran).
    """
    notes = [
        f"- {entry.model_name}: {_first_line(entry.error_message, limit=160)}"
        for entry in entries
        if entry.status in ("skipped", "error")
    ]
    return "\n".join(notes) + "\n" if notes else ""


def _first_line(message: str, limit: int) -> str:
    """Return the first line of a message, truncated to a length cap.

    Args:
        message: Full error/skip message.
        limit: Maximum characters to keep.

    Returns:
        Single-line summary with an ellipsis marker when truncated.
    """
    first = message.splitlines()[0] if message else ""
    if len(first) > limit:
        return first[:limit] + "..."
    return first
