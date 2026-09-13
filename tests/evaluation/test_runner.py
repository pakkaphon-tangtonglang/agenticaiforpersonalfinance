"""Tests for evaluation runner."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

from finance_ai.evaluation.models import (
    AccuracyAggregateResult,
    HallucinationAggregateResult,
    RAGAggregateResult,
    RecommendationSafetyAggregateResult,
    RoutingAggregateResult,
)
from finance_ai.evaluation.runner import EvaluationRunner


def _make_runner() -> EvaluationRunner:
    """Create a runner with mock dependencies."""
    return EvaluationRunner(
        chat_model=MagicMock(),
        vector_store=MagicMock(),
        llm_provider="google",
        llm_model="gemini-2.0-flash",
        judge_model=MagicMock(),
        data_dir="data/evaluation",
    )


class TestEvaluationRunner:
    """Tests for EvaluationRunner."""

    @patch("finance_ai.evaluation.runner.evaluate_routing_dataset")
    @patch("finance_ai.evaluation.runner.load_routing_dataset")
    def test_run_routing(
        self,
        mock_load: MagicMock,
        mock_eval: MagicMock,
    ) -> None:
        """Test run_routing delegates correctly."""
        expected = RoutingAggregateResult(
            total_cases=5,
            correct_count=4,
            accuracy=Decimal("0.8"),
            per_intent_accuracy={},
            confusion_matrix={},
            mean_latency_seconds=0.5,
            results=[],
        )
        mock_eval.return_value = expected
        runner = _make_runner()
        result = runner.run_routing()
        assert result.total_cases == 5
        mock_load.assert_called_once()

    @patch("finance_ai.evaluation.runner.evaluate_rag_dataset")
    @patch("finance_ai.evaluation.runner.load_rag_dataset")
    def test_run_rag(
        self,
        mock_load: MagicMock,
        mock_eval: MagicMock,
    ) -> None:
        """Test run_rag delegates correctly."""
        expected = RAGAggregateResult(
            total_cases=10,
            mean_precision_at_k=0.8,
            mean_recall_at_k=0.9,
            mean_reciprocal_rank=0.85,
            mean_keyword_hit_rate=0.9,
            per_domain_precision={},
            mean_latency_seconds=0.1,
            results=[],
        )
        mock_eval.return_value = expected
        runner = _make_runner()
        result = runner.run_rag()
        assert result.total_cases == 10

    @patch("finance_ai.evaluation.runner.evaluate_tax_accuracy_dataset")
    @patch("finance_ai.evaluation.runner.load_tax_accuracy_dataset")
    def test_run_tax_accuracy(
        self,
        mock_load: MagicMock,
        mock_eval: MagicMock,
    ) -> None:
        """Test run_tax_accuracy delegates correctly."""
        expected = AccuracyAggregateResult(
            agent_type="tax",
            total_cases=5,
            within_tolerance_count=4,
            accuracy_rate=Decimal("0.8"),
            mean_absolute_error_thb=Decimal("50"),
            mean_latency_seconds=2.0,
            results=[],
        )
        mock_eval.return_value = expected
        runner = _make_runner()
        result = runner.run_tax_accuracy()
        assert result.agent_type == "tax"

    @patch("finance_ai.evaluation.runner.evaluate_hallucination_dataset")
    @patch("finance_ai.evaluation.runner.load_hallucination_dataset")
    def test_run_hallucination(
        self,
        mock_load: MagicMock,
        mock_eval: MagicMock,
    ) -> None:
        """Test run_hallucination delegates correctly."""
        expected = HallucinationAggregateResult(
            total_cases=5,
            compliant_count=4,
            compliance_rate=Decimal("0.8"),
            violations_by_category={},
            mean_latency_seconds=0.5,
            results=[],
        )
        mock_eval.return_value = expected
        runner = _make_runner()
        result = runner.run_hallucination()
        assert result.total_cases == 5

    def test_run_quality_requires_judge(self) -> None:
        """Test that run_quality raises error without judge."""
        runner = EvaluationRunner(
            chat_model=MagicMock(),
            vector_store=MagicMock(),
            llm_provider="google",
            llm_model="gemini",
            judge_model=None,
        )
        import pytest

        with pytest.raises(ValueError, match="Judge model required"):
            runner.run_quality()


class TestRunRecommendationSafety:
    """Tests for run_recommendation_safety."""

    @patch("finance_ai.evaluation.runner.generate_safety_responses")
    @patch("finance_ai.evaluation.runner.evaluate_safety_dataset")
    @patch("finance_ai.evaluation.runner.load_recommendation_safety_dataset")
    def test_generates_then_evaluates(
        self,
        mock_load: MagicMock,
        mock_eval: MagicMock,
        mock_generate: MagicMock,
    ) -> None:
        """run_recommendation_safety generates responses then evaluates."""
        expected = RecommendationSafetyAggregateResult(
            total_cases=1,
            passed_count=1,
            compliance_rate=Decimal("1.0000"),
            failures_by_expected={},
            results=[],
        )
        mock_eval.return_value = expected
        mock_generate.return_value = ({"s1": "ok"}, {"s1": True})
        runner = _make_runner()

        result = runner.run_recommendation_safety()

        assert result is expected
        mock_load.assert_called_once_with("data/evaluation/recommendation_safety_dataset.yaml")
        mock_generate.assert_called_once()
        mock_eval.assert_called_once()
