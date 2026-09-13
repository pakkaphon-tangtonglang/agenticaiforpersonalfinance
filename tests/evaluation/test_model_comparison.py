"""Tests for the multi-model comparison evaluation.

Compares candidate models on the routing dimension (accuracy +
latency) so the thesis can justify the production model choice.
Models with missing credentials (e.g. Gemini before its API key is
configured) are reported as skipped, not crashes.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

from finance_ai.evaluation.models import (
    ModelComparisonEntry,
    RoutingAggregateResult,
)
from finance_ai.evaluation.model_comparison import (
    DEFAULT_COMPARISON_MODELS,
    ModelSpec,
    format_comparison_markdown,
    parse_model_spec,
    run_model_comparison,
)


def _routing_result(accuracy: str, latency: float) -> RoutingAggregateResult:
    """Build a minimal routing aggregate for tests."""
    return RoutingAggregateResult(
        total_cases=10,
        correct_count=9,
        accuracy=Decimal(accuracy),
        per_intent_accuracy={},
        confusion_matrix={},
        mean_latency_seconds=latency,
        results=[],
    )


class TestModelSpec:
    """Tests for ModelSpec parsing."""

    def test_parse_ollama_spec_with_colon_in_model(self) -> None:
        """Model names containing colons (qwen3.5:397b) parse correctly."""
        spec = parse_model_spec("ollama:qwen3.5:397b")

        assert spec.provider == "ollama"
        assert spec.model_name == "qwen3.5:397b"

    def test_parse_google_spec(self) -> None:
        """A provider:model string splits on the first colon."""
        spec = parse_model_spec("google:gemini-3.5")

        assert spec == ModelSpec(provider="google", model_name="gemini-3.5")

    def test_parse_invalid_spec_raises(self) -> None:
        """A string without a provider separator is rejected."""
        with pytest.raises(ValueError, match="provider:model"):
            parse_model_spec("minimax-m3")


class TestDefaultComparisonModels:
    """Tests for the default comparison shortlist."""

    def test_includes_production_baseline(self) -> None:
        """The production model (minimax-m3) is the comparison baseline."""
        assert ModelSpec(provider="ollama", model_name="minimax-m3") in DEFAULT_COMPARISON_MODELS

    def test_includes_gemini_candidates(self) -> None:
        """Gemini 3.5 and 3.5 Flash are included as candidates."""
        names = [(s.provider, s.model_name) for s in DEFAULT_COMPARISON_MODELS]

        assert ("google", "gemini-3.5") in names
        assert ("google", "gemini-3.5-flash") in names

    def test_all_specs_use_valid_providers(self) -> None:
        """Every default spec uses a provider the CLI supports."""
        assert all(
            s.provider in ("google", "ollama", "openrouter") for s in DEFAULT_COMPARISON_MODELS
        )


class TestRunModelComparison:
    """Tests for run_model_comparison with injected factories."""

    def test_collects_routing_accuracy_and_latency(self) -> None:
        """A successful run records accuracy, latency, and counts."""
        runner = MagicMock()
        runner.run_routing.return_value = _routing_result("0.90", 1.5)
        result = run_model_comparison(
            [ModelSpec(provider="ollama", model_name="minimax-m3")],
            model_factory=lambda spec: MagicMock(),
            runner_factory=lambda model, spec: runner,
        )

        entry = result.entries[0]
        assert entry.status == "ok"
        assert entry.model_name == "minimax-m3"
        assert entry.routing_accuracy == Decimal("0.90")
        assert entry.mean_latency_seconds == 1.5
        assert (entry.total_cases, entry.correct_count) == (10, 9)

    def test_missing_credentials_marked_skipped(self) -> None:
        """A ValueError (missing API key) yields a skipped entry."""

        def _fail(spec: ModelSpec) -> Any:
            raise ValueError("google_api_key is required when llm_provider is 'google'.")

        result = run_model_comparison(
            [ModelSpec(provider="google", model_name="gemini-3.5")],
            model_factory=_fail,
            runner_factory=lambda model, spec: MagicMock(),
        )

        entry = result.entries[0]
        assert entry.status == "skipped"
        assert "google_api_key" in entry.error_message
        assert entry.routing_accuracy is None

    def test_model_failure_recorded_and_continues(self) -> None:
        """A crashing model is recorded as an error; later models still run."""
        runners = [MagicMock(), MagicMock()]
        runners[1].run_routing.return_value = _routing_result("0.80", 2.0)
        specs = [
            ModelSpec(provider="ollama", model_name="bad-model"),
            ModelSpec(provider="ollama", model_name="minimax-m3"),
        ]
        models = iter([None, MagicMock()])
        raised = [False]

        def _factory(spec: ModelSpec) -> Any:
            if spec.model_name == "bad-model":
                raised[0] = True
                raise RuntimeError("connection refused")
            return next(models)

        def _runner_factory(model: Any, spec: ModelSpec) -> Any:
            return runners[1] if spec.model_name == "minimax-m3" else MagicMock()

        result = run_model_comparison(
            specs,
            model_factory=_factory,
            runner_factory=_runner_factory,
        )

        assert raised[0]
        assert result.entries[0].status == "error"
        assert "connection refused" in result.entries[0].error_message
        assert result.entries[1].status == "ok"
        assert result.entries[1].routing_accuracy == Decimal("0.80")


class TestComparisonReport:
    """Tests for comparison report formatting."""

    def test_markdown_table_lists_all_models(self) -> None:
        """The markdown table includes every model with its status."""
        result = run_model_comparison(
            [ModelSpec(provider="ollama", model_name="minimax-m3")],
            model_factory=lambda spec: MagicMock(),
            runner_factory=lambda model, spec: MagicMock(
                run_routing=MagicMock(return_value=_routing_result("0.90", 1.5)),
            ),
        )
        markdown = format_comparison_markdown(result)

        assert "minimax-m3" in markdown
        assert "0.90" in markdown

    def test_result_is_serializable_model(self) -> None:
        """The comparison result is a pydantic model with metadata."""
        entry = ModelComparisonEntry(
            provider="ollama",
            model_name="minimax-m3",
            status="ok",
            routing_accuracy=Decimal("0.90"),
        )
        from finance_ai.evaluation.models import ModelComparisonResult  # noqa: PLC0415

        comparison = ModelComparisonResult(
            dimension="routing",
            generated_at=datetime(2026, 9, 14, 12, 0, 0),
            entries=[entry],
        )
        dumped = comparison.model_dump(mode="json")

        assert dumped["entries"][0]["model_name"] == "minimax-m3"
        assert dumped["dimension"] == "routing"


class TestComparisonErrorClassification:
    """Tests distinguishing skipped (missing key) from error (bad data)."""

    def test_validation_error_is_error_not_skipped(self) -> None:
        """Dataset ValidationError must be 'error', not 'skipped'.

        pydantic ValidationError subclasses ValueError, so the skip
        path (missing credentials) must not swallow it.
        """
        import pytest  # noqa: PLC0415
        from typing import cast  # noqa: PLC0415
        from finance_ai.evaluation.models import RoutingCase  # noqa: PLC0415

        def _raising_run_routing() -> None:
            invalid_intent = cast(Any, "bogus")
            RoutingCase(case_id="x", query="y", expected_intent=invalid_intent)

        runner = MagicMock()
        runner.run_routing.side_effect = _raising_run_routing
        result = run_model_comparison(
            [ModelSpec(provider="ollama", model_name="minimax-m3")],
            model_factory=lambda spec: MagicMock(),
            runner_factory=lambda model, spec: runner,
        )

        entry = result.entries[0]
        assert entry.status == "error"
        assert "expected_intent" in entry.error_message

    def test_missing_key_still_skipped(self) -> None:
        """A plain ValueError (missing API key) stays 'skipped'."""

        def _fail(spec: ModelSpec) -> Any:
            raise ValueError("ollama_api_key is required")

        result = run_model_comparison(
            [ModelSpec(provider="ollama", model_name="minimax-m3")],
            model_factory=_fail,
            runner_factory=lambda model, spec: MagicMock(),
        )

        assert result.entries[0].status == "skipped"

    def test_long_error_messages_truncated_in_markdown(self) -> None:
        """Multi-line error messages collapse to one line in markdown."""
        runner = MagicMock()
        runner.run_routing.side_effect = RuntimeError("line one\nline two\nline three")
        result = run_model_comparison(
            [ModelSpec(provider="ollama", model_name="bad-model")],
            model_factory=lambda spec: MagicMock(),
            runner_factory=lambda model, spec: runner,
        )
        markdown = format_comparison_markdown(result)

        assert "line one" in markdown
        assert "line three" not in markdown
