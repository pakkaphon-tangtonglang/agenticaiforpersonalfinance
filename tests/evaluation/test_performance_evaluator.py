"""Tests for performance evaluator."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

from finance_ai.evaluation.models import QualityCase, QualityDataset
from finance_ai.evaluation.performance_evaluator import (
    _aggregate_performance_results,
    estimate_cost,
    evaluate_performance_dataset,
    evaluate_single_performance,
)


class TestEstimateCost:
    """Tests for estimate_cost."""

    def test_known_model(self) -> None:
        """Test cost estimation for a known model."""
        cost = estimate_cost(1000, 500, "gemini-2.0-flash")
        assert cost > Decimal("0")

    def test_unknown_model_uses_default(self) -> None:
        """Test cost estimation for unknown model uses defaults."""
        cost = estimate_cost(1000, 500, "unknown-model")
        assert cost > Decimal("0")

    def test_zero_tokens(self) -> None:
        """Test cost with zero tokens."""
        cost = estimate_cost(0, 0, "gemini-2.0-flash")
        assert cost == Decimal("0.0000000")


class TestEvaluateSinglePerformance:
    """Tests for evaluate_single_performance."""

    @patch("finance_ai.evaluation.performance_evaluator.orchestrate_query")
    def test_records_latency(self, mock_route: MagicMock) -> None:
        """Test that latency is recorded."""
        mock_route.return_value = {"intent": "tax", "response": "response"}
        model = MagicMock()
        case = QualityCase(
            case_id="perf_001",
            query="คำนวณภาษี",
            expected_agent="tax",
        )
        result = evaluate_single_performance(model, case)
        assert result.total_latency_seconds >= 0.0
        assert result.agent == "tax"
        assert result.case_id == "perf_001"

    @patch("finance_ai.evaluation.performance_evaluator.orchestrate_query")
    def test_captures_agent_intent(self, mock_route: MagicMock) -> None:
        """Test that agent intent is captured from route result."""
        mock_route.return_value = {"intent": "expense", "response": "r"}
        model = MagicMock()
        case = QualityCase(
            case_id="perf_002",
            query="ค่ากาแฟ",
            expected_agent="expense",
        )
        result = evaluate_single_performance(model, case)
        assert result.agent == "expense"


class TestEvaluatePerformanceDataset:
    """Tests for evaluate_performance_dataset."""

    @patch("finance_ai.evaluation.performance_evaluator.orchestrate_query")
    def test_aggregate_results(self, mock_route: MagicMock) -> None:
        """Test dataset aggregation."""
        mock_route.return_value = {"intent": "tax", "response": "r"}
        model = MagicMock()
        dataset = QualityDataset(
            version="1.0",
            cases=[
                QualityCase(case_id="p_001", query="q1", expected_agent="tax"),
                QualityCase(case_id="p_002", query="q2", expected_agent="tax"),
            ],
        )
        agg = evaluate_performance_dataset(model, dataset)
        assert agg.total_queries == 2
        assert agg.mean_total_latency >= 0.0
        assert "tax" in agg.per_agent_latency

    @patch("finance_ai.evaluation.performance_evaluator.orchestrate_query")
    def test_empty_dataset(self, mock_route: MagicMock) -> None:
        """Test with empty dataset."""
        model = MagicMock()
        dataset = QualityDataset(version="1.0", cases=[])
        agg = evaluate_performance_dataset(model, dataset)
        assert agg.total_queries == 0
        assert agg.p50_latency == 0.0


class TestAggregatePerformanceResults:
    """Tests for _aggregate_performance_results."""

    def test_percentile_calculation(self) -> None:
        """Test percentile latency calculations."""
        from finance_ai.evaluation.models import PerformanceResult

        results = [
            PerformanceResult(
                case_id=f"p_{i}",
                query="q",
                agent="tax",
                total_latency_seconds=float(i),
            )
            for i in range(1, 11)
        ]
        agg = _aggregate_performance_results(results)
        assert agg.total_queries == 10
        assert agg.p50_latency > 0.0
        assert agg.p95_latency >= agg.p50_latency
        assert agg.p99_latency >= agg.p95_latency

    def test_cost_summation(self) -> None:
        """Test total cost summation."""
        from finance_ai.evaluation.models import PerformanceResult

        results = [
            PerformanceResult(
                case_id="p_1",
                query="q",
                agent="tax",
                total_latency_seconds=1.0,
                estimated_cost_usd=Decimal("0.001"),
            ),
            PerformanceResult(
                case_id="p_2",
                query="q",
                agent="tax",
                total_latency_seconds=2.0,
                estimated_cost_usd=Decimal("0.002"),
            ),
        ]
        agg = _aggregate_performance_results(results)
        assert agg.total_estimated_cost_usd == Decimal("0.003")
