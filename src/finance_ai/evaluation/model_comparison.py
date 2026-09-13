"""Multi-model comparison evaluation for the thesis model selection.

Runs the routing dimension (intent accuracy + latency) across candidate
models — the production baseline plus alternatives from Ollama Cloud
and Google Gemini — so the thesis can justify the production model
choice with measured numbers instead of anecdote.

Models whose credentials are missing (e.g. Gemini before
GOOGLE_API_KEY is configured) are reported as 'skipped' rather than
aborting the whole comparison.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from finance_ai.core.logging import get_logger
from finance_ai.evaluation.models import (
    ModelComparisonEntry,
    ModelComparisonResult,
    RoutingAggregateResult,
)

logger = get_logger(__name__)

ModelFactory = Callable[["ModelSpec"], Any]
RunnerFactory = Callable[[Any, "ModelSpec"], Any]


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
    ModelSpec(provider="google", model_name="gemini-3.5"),
    ModelSpec(provider="google", model_name="gemini-3.5-flash"),
]


def parse_model_spec(text: str) -> ModelSpec:
    """Parse a 'provider:model' CLI string into a ModelSpec.

    Splits on the FIRST colon only, because Ollama model names
    themselves contain colons (e.g. 'qwen3.5:397b').

    Args:
        text: String like 'ollama:qwen3.5:397b' or 'google:gemini-3.5'.

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
            "(e.g. ollama:minimax-m3, google:gemini-3.5)."
        )
    return ModelSpec(provider=provider, model_name=model_name)


def run_model_comparison(
    specs: list[ModelSpec],
    data_dir: str = "data/evaluation",
    model_factory: ModelFactory | None = None,
    runner_factory: RunnerFactory | None = None,
) -> ModelComparisonResult:
    """Run the routing dimension for every candidate model.

    Args:
        specs: Candidate models to evaluate, in run order.
        data_dir: Evaluation dataset directory.
        model_factory: Callable building a chat model from a spec
            (defaults to the provider-aware factory in the CLI module).
        runner_factory: Callable building an EvaluationRunner from
            (model, spec); defaults to the real runner.

    Returns:
        ModelComparisonResult with one entry per candidate.
    """
    entries = [
        _compare_single_model(spec, data_dir, model_factory, runner_factory) for spec in specs
    ]
    return ModelComparisonResult(
        dimension="routing",
        generated_at=datetime.now(timezone.utc),
        entries=entries,
    )


def _compare_single_model(
    spec: ModelSpec,
    data_dir: str,
    model_factory: ModelFactory | None,
    runner_factory: RunnerFactory | None,
) -> ModelComparisonEntry:
    """Evaluate one candidate model on the routing dimension.

    Args:
        spec: Candidate model.
        data_dir: Evaluation dataset directory.
        model_factory: Optional injected model factory.
        runner_factory: Optional injected runner factory.

    Returns:
        Entry with routing metrics, or a skipped/error status.
    """
    logger.info("Comparing model %s:%s", spec.provider, spec.model_name)
    try:
        factory = model_factory or _default_model_factory
        model = factory(spec)
        runner = _build_runner(model, spec, data_dir, runner_factory)
        routing = runner.run_routing()
    except ValidationError as exc:
        # Dataset/config problems are real failures — not missing
        # credentials (ValidationError subclasses ValueError).
        return ModelComparisonEntry(
            provider=spec.provider,
            model_name=spec.model_name,
            status="error",
            error_message=str(exc),
        )
    except ValueError as exc:
        return _skipped_entry(spec, exc)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return ModelComparisonEntry(
            provider=spec.provider,
            model_name=spec.model_name,
            status="error",
            error_message=str(exc),
        )
    return _ok_entry(spec, routing)


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
) -> Any:
    """Build an EvaluationRunner for the given model.

    Args:
        model: Chat model under evaluation.
        spec: Candidate model spec.
        data_dir: Evaluation dataset directory.
        runner_factory: Optional injected factory (tests).

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


def _ok_entry(spec: ModelSpec, routing: RoutingAggregateResult) -> ModelComparisonEntry:
    """Build a successful entry from a routing aggregate result.

    Args:
        spec: Candidate model.
        routing: Aggregated routing metrics for the model.

    Returns:
        Entry with status 'ok' and the routing metrics.
    """
    return ModelComparisonEntry(
        provider=spec.provider,
        model_name=spec.model_name,
        status="ok",
        routing_accuracy=routing.accuracy,
        mean_latency_seconds=routing.mean_latency_seconds,
        total_cases=routing.total_cases,
        correct_count=routing.correct_count,
    )


def format_comparison_markdown(result: ModelComparisonResult) -> str:
    """Render a comparison result as a markdown table for the thesis.

    Args:
        result: Comparison result from run_model_comparison.

    Returns:
        Markdown string with one row per model, sorted by accuracy.

    Example:
        >>> print(format_comparison_markdown(result))  # doctest: +SKIP
    """
    header = (
        "| Model | Provider | Status | Routing Accuracy | Mean Latency (s) |\n"
        "|---|---|---|---|---|\n"
    )
    ranked = sorted(result.entries, key=_accuracy_sort_key, reverse=True)
    rows = [_format_entry_row(entry) for entry in ranked]
    notes = _format_skipped_notes(ranked)
    return header + "\n".join(rows) + "\n" + notes


def _accuracy_sort_key(entry: ModelComparisonEntry) -> Decimal:
    """Sort key ranking successful entries by accuracy (others last).

    Args:
        entry: Comparison entry.

    Returns:
        Accuracy, or -1 when the model did not run.
    """
    return entry.routing_accuracy if entry.routing_accuracy is not None else Decimal(-1)


def _format_entry_row(entry: ModelComparisonEntry) -> str:
    """Format one comparison entry as a markdown table row.

    Args:
        entry: Comparison entry.

    Returns:
        Markdown row string.
    """
    accuracy = f"{entry.routing_accuracy:.2f}" if entry.routing_accuracy is not None else "-"
    latency = f"{entry.mean_latency_seconds:.2f}" if entry.mean_latency_seconds is not None else "-"
    return f"| {entry.model_name} | {entry.provider} | {entry.status} | {accuracy} | {latency} |"


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
