"""Tests for architecture benchmark helper functions."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from finance_ai.evaluation.architecture_benchmark import (
    N_AGENTS,
    ArchitectureBenchmarkResult,
    ArchitectureSummary,
    QueryBenchmarkResult,
    _build_summary,
    _classify_macro,
    _compute_scalability_data,
)


class TestArchitectureBenchmarkHelpers:
    """Tests for architecture benchmark helper functions."""

    def test_compute_scalability_data(self) -> None:
        """Scalability data contains expected coupling values."""
        data = _compute_scalability_data(range(2, 5))
        assert data["n_agents"] == [2, 3, 4]
        assert data["hub_spoke"] == [2, 3, 4]
        assert data["p2p"] == [1, 3, 6]
        assert data["hierarchical"] == [4, 5, 6]

    def test_build_summary_empty_results(self) -> None:
        """Build summary with empty results gives zero means."""
        summary = _build_summary("hub_spoke", [], N_AGENTS)
        assert summary.architecture == "hub_spoke"
        assert summary.mean_latency_ms == 0.0
        assert summary.routing_llm_calls_per_query == 0.0
        assert summary.coupling_score == N_AGENTS

    def test_build_summary_with_results(self) -> None:
        """Build summary computes mean latency and calls."""
        results = [
            QueryBenchmarkResult(
                architecture="hub_spoke",
                query="q1",
                query_index=0,
                latency_ms=1000.0,
                routing_llm_calls=1,
                agent_intent="tax",
                hops=1,
            ),
            QueryBenchmarkResult(
                architecture="hub_spoke",
                query="q2",
                query_index=1,
                latency_ms=2000.0,
                routing_llm_calls=1,
                agent_intent="expense",
                hops=1,
            ),
        ]
        summary = _build_summary("hub_spoke", results, N_AGENTS)
        assert summary.mean_latency_ms == 1500.0
        assert summary.routing_llm_calls_per_query == 1.0
        assert len(summary.per_query_results) == 2

    def test_classify_macro_returns_macro(self) -> None:
        """_classify_macro parses JSON macro from model response."""
        mock_model = MagicMock()
        mock_model.invoke.return_value = AIMessage(
            content='{"macro": "financial_planning", "confidence": 0.9}',
        )
        result = _classify_macro("test query", mock_model)
        assert result == "financial_planning"

    def test_classify_macro_falls_back_on_bad_json(self) -> None:
        """_classify_macro returns default on malformed JSON."""
        mock_model = MagicMock()
        mock_model.invoke.return_value = AIMessage(content="not json")
        result = _classify_macro("test query", mock_model)
        assert result == "financial_operations"


class TestArchitectureBenchmarkModels:
    """Tests for architecture benchmark models."""

    def test_benchmark_result_defaults(self) -> None:
        """ArchitectureBenchmarkResult has default lists."""
        summary = ArchitectureSummary(
            architecture="hub_spoke",
            mean_latency_ms=1000.0,
            routing_llm_calls_per_query=1.0,
            coupling_score=6,
        )
        result = ArchitectureBenchmarkResult(
            hub_spoke=summary,
            p2p=summary,
            hierarchical=summary,
        )
        assert result.n_agents == N_AGENTS
        assert result.benchmark_queries == []
        assert result.scalability_data == {}
