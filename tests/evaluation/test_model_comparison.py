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
    AccuracyAggregateResult,
    ModelComparisonEntry,
    RoutingAggregateResult,
)
from finance_ai.evaluation.model_comparison import (
    DEFAULT_COMPARISON_MODELS,
    ModelSpec,
    format_comparison_markdown,
    parse_model_spec,
    run_model_comparison,
    run_multi_dimension_comparison,
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
        """Gemini 3.5 Flash and Flash Lite are included as candidates."""
        names = [(s.provider, s.model_name) for s in DEFAULT_COMPARISON_MODELS]

        assert ("google", "gemini-3.5-flash") in names
        assert ("google", "gemini-3.5-flash-lite") in names

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
        assert entry.accuracy == Decimal("0.90")
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
        assert entry.accuracy is None

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
        assert result.entries[1].accuracy == Decimal("0.80")


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
            accuracy=Decimal("0.90"),
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


class TestConcurrentComparison:
    """Tests for parallel candidate evaluation."""

    def test_parallel_run_preserves_spec_order(self) -> None:
        """max_workers>1 evaluates concurrently but keeps input order."""
        specs = [ModelSpec(provider="ollama", model_name=f"model-{index}") for index in range(4)]
        runners = {}
        for spec in specs:
            runner = MagicMock()
            runner.run_routing.return_value = _routing_result(
                f"0.{8 - specs.index(spec)}", float(specs.index(spec) + 1)
            )
            runners[spec.model_name] = runner

        result = run_model_comparison(
            specs,
            model_factory=lambda spec: MagicMock(),
            runner_factory=lambda model, spec: runners[spec.model_name],
            max_workers=3,
        )

        assert [e.model_name for e in result.entries] == [s.model_name for s in specs]
        assert all(e.status == "ok" for e in result.entries)

    def test_parallel_run_uses_multiple_threads(self) -> None:
        """Concurrent evaluation spreads work across worker threads."""
        import threading  # noqa: PLC0415

        thread_names: list[str] = []
        barrier = threading.Barrier(3, timeout=10)

        def _factory(spec: ModelSpec) -> Any:
            barrier.wait()
            thread_names.append(threading.current_thread().name)
            return MagicMock()

        def _runner_factory(model: Any, spec: ModelSpec) -> Any:
            runner = MagicMock()
            runner.run_routing.return_value = _routing_result("0.90", 1.0)
            return runner

        specs = [ModelSpec(provider="ollama", model_name=f"m{i}") for i in range(3)]
        result = run_model_comparison(
            specs,
            model_factory=_factory,
            runner_factory=_runner_factory,
            max_workers=3,
        )

        assert len(result.entries) == 3
        assert len(set(thread_names)) > 1


class TestTaxAccuracyDimension:
    """Tests for comparing models on the tax-accuracy dimension."""

    def _tax_result(self, rate: str, mae: str, latency: float) -> AccuracyAggregateResult:
        """Build a tax-accuracy aggregate for tests."""
        return AccuracyAggregateResult(
            agent_type="tax",
            total_cases=20,
            within_tolerance_count=18,
            accuracy_rate=Decimal(rate),
            mean_absolute_error_thb=Decimal(mae),
            mean_latency_seconds=latency,
            results=[],
        )

    def test_tax_dimension_records_rate_and_mae(self) -> None:
        """dimension='tax-accuracy' scores tax-agent correctness + MAE."""
        runner = MagicMock()
        runner.run_tax_accuracy.return_value = self._tax_result("0.90", "150.00", 5.2)
        result = run_model_comparison(
            [ModelSpec(provider="ollama", model_name="minimax-m3")],
            model_factory=lambda spec: MagicMock(),
            runner_factory=lambda model, spec: runner,
            dimension="tax-accuracy",
        )

        entry = result.entries[0]
        assert result.dimension == "tax-accuracy"
        assert entry.status == "ok"
        assert entry.accuracy == Decimal("0.90")
        assert entry.mean_absolute_error_thb == Decimal("150.00")
        assert entry.mean_latency_seconds == 5.2
        assert (entry.total_cases, entry.correct_count) == (20, 18)

    def test_unknown_dimension_raises(self) -> None:
        """An unsupported dimension name is rejected up front."""
        with pytest.raises(ValueError, match="Unsupported comparison dimension"):
            run_model_comparison(
                [ModelSpec(provider="ollama", model_name="minimax-m3")],
                dimension="bogus",
            )

    def test_markdown_includes_mae_column(self) -> None:
        """The report shows MAE for accuracy-dimension comparisons."""
        runner = MagicMock()
        runner.run_tax_accuracy.return_value = self._tax_result("0.90", "150.00", 5.2)
        result = run_model_comparison(
            [ModelSpec(provider="ollama", model_name="minimax-m3")],
            model_factory=lambda spec: MagicMock(),
            runner_factory=lambda model, spec: runner,
            dimension="tax-accuracy",
        )
        markdown = format_comparison_markdown(result)

        assert "MAE (THB)" in markdown
        assert "150.00" in markdown


class TestMultiDimensionComparison:
    """Tests for running every model x dimension pair at once."""

    def test_all_pairs_run_and_group_by_dimension(self) -> None:
        """One shared pool evaluates all pairs; results group per dimension."""
        runner = MagicMock()
        runner.run_routing.return_value = _routing_result("0.90", 1.5)
        runner.run_tax_accuracy.return_value = self._tax_result("0.80", "150.00", 4.0)
        specs = [ModelSpec(provider="ollama", model_name="minimax-m3")]

        results = run_multi_dimension_comparison(
            specs,
            ["routing", "tax-accuracy"],
            model_factory=lambda spec: MagicMock(),
            runner_factory=lambda model, spec: runner,
            max_workers=2,
        )

        assert [result.dimension for result in results] == ["routing", "tax-accuracy"]
        assert results[0].entries[0].accuracy == Decimal("0.90")
        assert results[1].entries[0].accuracy == Decimal("0.80")
        assert results[1].entries[0].mean_absolute_error_thb == Decimal("150.00")

    def _tax_result(self, rate: str, mae: str, latency: float) -> AccuracyAggregateResult:
        """Build a tax-accuracy aggregate for tests."""
        return AccuracyAggregateResult(
            agent_type="tax",
            total_cases=20,
            within_tolerance_count=16,
            accuracy_rate=Decimal(rate),
            mean_absolute_error_thb=Decimal(mae),
            mean_latency_seconds=latency,
            results=[],
        )

    def test_invalid_dimension_raises_before_running(self) -> None:
        """An unsupported dimension is rejected before any model runs."""
        with pytest.raises(ValueError, match="Unsupported comparison dimension"):
            run_multi_dimension_comparison(
                [ModelSpec(provider="ollama", model_name="minimax-m3")],
                ["routing", "bogus"],
            )


class TestQualityDimension:
    """Tests for the LLM-as-judge quality comparison dimension."""

    def _quality_result(self, mean_overall: str, latency: float) -> Any:
        """Build a minimal quality aggregate (judge scores 1-5)."""
        return MagicMock(
            mean_overall=Decimal(mean_overall),
            mean_latency_seconds=latency,
            total_cases=12,
        )

    def test_quality_normalizes_mean_overall_to_unit_scale(self) -> None:
        """Judge mean score (1-5) is normalized to 0-1 accuracy."""
        runner = MagicMock()
        runner.run_quality.return_value = self._quality_result("4.20", 6.5)
        result = run_model_comparison(
            [ModelSpec(provider="ollama", model_name="minimax-m3")],
            model_factory=lambda spec: MagicMock(),
            runner_factory=lambda model, spec: runner,
            dimension="quality",
        )

        entry = result.entries[0]
        assert entry.status == "ok"
        assert entry.accuracy == Decimal("4.20") / Decimal("5")
        assert entry.total_cases == 12
        assert entry.mean_absolute_error_thb is None

    def test_quality_without_judge_is_skipped(self) -> None:
        """A runner without a judge model is reported as skipped."""
        runner = MagicMock()
        runner.run_quality.side_effect = ValueError("Judge model required for quality evaluation")
        result = run_model_comparison(
            [ModelSpec(provider="ollama", model_name="minimax-m3")],
            model_factory=lambda spec: MagicMock(),
            runner_factory=lambda model, spec: runner,
            dimension="quality",
        )

        entry = result.entries[0]
        assert entry.status == "skipped"

    def test_judge_spec_built_once_per_run(self) -> None:
        """The judge model is built once and shared by all quality jobs."""
        runner = MagicMock()
        runner.run_quality.return_value = self._quality_result("4.0", 5.0)
        specs = [
            ModelSpec(provider="ollama", model_name="minimax-m3"),
            ModelSpec(provider="ollama", model_name="glm-5.3"),
        ]
        built: list[ModelSpec] = []

        def _recording_factory(spec: ModelSpec) -> Any:
            built.append(spec)
            return MagicMock()

        run_multi_dimension_comparison(
            specs,
            ["quality"],
            model_factory=_recording_factory,
            runner_factory=lambda model, spec: runner,
            judge_spec=ModelSpec(provider="ollama", model_name="deepseek-v4-pro:0813"),
        )

        judge_builds = [s for s in built if "deepseek" in s.model_name]
        assert len(judge_builds) == 1


class TestHallucinationDimension:
    """Tests for the anti-hallucination comparison dimension."""

    def test_hallucination_uses_compliance_rate(self) -> None:
        """Compliance rate becomes the accuracy metric."""
        aggregate = MagicMock(
            total_cases=16,
            compliant_count=14,
            compliance_rate=Decimal("0.875"),
            mean_latency_seconds=3.1,
        )
        runner = MagicMock()
        runner.run_hallucination.return_value = aggregate
        result = run_model_comparison(
            [ModelSpec(provider="ollama", model_name="minimax-m3")],
            model_factory=lambda spec: MagicMock(),
            runner_factory=lambda model, spec: runner,
            dimension="hallucination",
        )

        entry = result.entries[0]
        assert entry.status == "ok"
        assert entry.accuracy == Decimal("0.875")
        assert (entry.total_cases, entry.correct_count) == (16, 14)


class TestRecommendationSafetyDimension:
    """Tests for the recommendation guardrail compliance dimension."""

    def test_recommendation_safety_uses_compliance_rate(self) -> None:
        """Guardrail compliance rate becomes the accuracy metric."""
        aggregate = MagicMock(
            total_cases=24,
            passed_count=22,
            compliance_rate=Decimal("0.9167"),
        )
        runner = MagicMock()
        runner.run_recommendation_safety.return_value = aggregate
        result = run_model_comparison(
            [ModelSpec(provider="ollama", model_name="minimax-m3")],
            model_factory=lambda spec: MagicMock(),
            runner_factory=lambda model, spec: runner,
            dimension="recommendation-safety",
        )

        entry = result.entries[0]
        assert entry.status == "ok"
        assert entry.accuracy == Decimal("0.9167")
        assert (entry.total_cases, entry.correct_count) == (24, 22)
        assert entry.mean_latency_seconds is None


class TestTaxAccuracyForcedDimension:
    """Tests for the forced-tool tax accuracy dimension."""

    def test_forced_dimension_calls_run_tax_accuracy_forced(self) -> None:
        """dimension='tax-accuracy-forced' isolates tool-argument accuracy."""
        runner = MagicMock()
        runner.run_tax_accuracy_forced.return_value = MagicMock(
            accuracy_rate=Decimal("0.95"),
            within_tolerance_count=19,
            total_cases=20,
            mean_absolute_error_thb=Decimal("80.00"),
            mean_latency_seconds=4.0,
        )
        result = run_model_comparison(
            [ModelSpec(provider="ollama", model_name="minimax-m3")],
            model_factory=lambda spec: MagicMock(),
            runner_factory=lambda model, spec: runner,
            dimension="tax-accuracy-forced",
        )

        entry = result.entries[0]
        runner.run_tax_accuracy_forced.assert_called_once()
        assert entry.accuracy == Decimal("0.95")
        assert entry.mean_absolute_error_thb == Decimal("80.00")
