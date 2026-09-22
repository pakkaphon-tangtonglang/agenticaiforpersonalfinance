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
import time
from dataclasses import replace
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
