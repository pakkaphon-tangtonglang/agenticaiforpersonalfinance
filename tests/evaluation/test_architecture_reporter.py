"""Tests for architecture benchmark reporter."""

import os
from decimal import Decimal

from finance_ai.evaluation.architecture_benchmark import (
    ArchitectureBenchmarkResult,
    ArchitectureSummary,
    QueryBenchmarkResult,
)
from finance_ai.evaluation.architecture_reporter import (
    _build_report,
    save_architecture_report,
)


def _make_summary(architecture: str) -> ArchitectureSummary:
    """Create a minimal architecture summary for testing."""
    return ArchitectureSummary(
        architecture=architecture,
        mean_latency_ms=1000.0,
        routing_llm_calls_per_query=1.0,
        coupling_score=6,
        per_query_results=[
            QueryBenchmarkResult(
                architecture=architecture,  # type: ignore[arg-type]
                query="test query",
                query_index=0,
                latency_ms=1200.0,
                routing_llm_calls=1,
                agent_intent="tax",
                hops=1,
            ),
        ],
    )


def _make_result() -> ArchitectureBenchmarkResult:
    """Create a minimal benchmark result for testing."""
    return ArchitectureBenchmarkResult(
        hub_spoke=_make_summary("hub_spoke"),
        p2p=_make_summary("p2p"),
        hierarchical=_make_summary("hierarchical"),
        n_agents=6,
        scalability_data={
            "n_agents": [2, 3],
            "p2p": [1, 3],
            "hierarchical": [4, 5],
            "hub_spoke": [2, 3],
        },
        benchmark_queries=["test query"],
    )


class TestArchitectureReporter:
    """Tests for architecture comparison report generation."""

    def test_build_report_contains_all_sections(self) -> None:
        """Report contains header, summary, per-query, scalability, conclusion."""
        result = _make_result()
        report = _build_report(result)
        assert "เปรียบเทียบสถาปัตยกรรม Multi-Agent" in report
        assert "Hub-and-Spoke" in report
        assert "test query" in report
        assert "สรุป" in report

    def test_save_architecture_report_writes_file(self, tmp_path: str) -> None:
        """save_architecture_report writes markdown to disk."""
        result = _make_result()
        output_path = os.path.join(tmp_path, "architecture_comparison.md")
        written_path = save_architecture_report(result, output_path)
        assert os.path.exists(written_path)
        with open(written_path, encoding="utf-8") as file:
            content = file.read()
        assert "เปรียบเทียบสถาปัตยกรรม Multi-Agent" in content
