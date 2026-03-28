"""Tests for evaluation report generation."""

import tempfile
from decimal import Decimal
from pathlib import Path

from finance_ai.evaluation.models import (
    EvaluationReport,
    RoutingAggregateResult,
)
from finance_ai.evaluation.reporter import (
    _fmt_pct,
    save_comparison_report,
    save_json_report,
    save_markdown_report,
)


def _make_minimal_report() -> EvaluationReport:
    """Create a minimal evaluation report."""
    return EvaluationReport(
        report_id="test-report-001",
        timestamp="2024-01-01T00:00:00",
        llm_provider="google",
        llm_model="gemini-2.0-flash",
    )


def _make_report_with_routing() -> EvaluationReport:
    """Create a report with routing results."""
    return EvaluationReport(
        report_id="test-report-002",
        timestamp="2024-01-01T00:00:00",
        llm_provider="google",
        llm_model="gemini-2.0-flash",
        routing=RoutingAggregateResult(
            total_cases=40,
            correct_count=37,
            accuracy=Decimal("0.925"),
            per_intent_accuracy={"tax": Decimal("1.0")},
            confusion_matrix={"tax": {"tax": 5}},
            mean_latency_seconds=0.5,
            results=[],
        ),
    )


class TestSaveJsonReport:
    """Tests for save_json_report."""

    def test_creates_json_file(self) -> None:
        """Test that JSON file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = _make_minimal_report()
            path = save_json_report(report, tmpdir)
            assert Path(path).exists()
            content = Path(path).read_text(encoding="utf-8")
            assert "test-report-001" in content

    def test_json_contains_all_fields(self) -> None:
        """Test that JSON contains all report fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = _make_report_with_routing()
            path = save_json_report(report, tmpdir)
            content = Path(path).read_text(encoding="utf-8")
            assert "routing" in content
            assert "0.925" in content


class TestSaveMarkdownReport:
    """Tests for save_markdown_report."""

    def test_creates_md_file(self) -> None:
        """Test that Markdown file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = _make_minimal_report()
            path = save_markdown_report(report, tmpdir)
            assert Path(path).exists()
            assert path.endswith(".md")

    def test_md_contains_header(self) -> None:
        """Test that Markdown contains report header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = _make_minimal_report()
            path = save_markdown_report(report, tmpdir)
            content = Path(path).read_text(encoding="utf-8")
            assert "gemini-2.0-flash" in content

    def test_md_contains_routing_table(self) -> None:
        """Test that Markdown contains routing metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = _make_report_with_routing()
            path = save_markdown_report(report, tmpdir)
            content = Path(path).read_text(encoding="utf-8")
            assert "Routing Accuracy" in content
            assert "37/40" in content


class TestSaveComparisonReport:
    """Tests for save_comparison_report."""

    def test_creates_comparison_file(self) -> None:
        """Test that comparison file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reports = [
                _make_report_with_routing(),
                EvaluationReport(
                    report_id="test-report-003",
                    timestamp="2024-01-01T00:00:00",
                    llm_provider="openrouter",
                    llm_model="gpt-4o",
                    routing=RoutingAggregateResult(
                        total_cases=40,
                        correct_count=36,
                        accuracy=Decimal("0.9"),
                        per_intent_accuracy={},
                        confusion_matrix={},
                        mean_latency_seconds=0.4,
                        results=[],
                    ),
                ),
            ]
            path = save_comparison_report(reports, tmpdir)
            assert Path(path).exists()
            content = Path(path).read_text(encoding="utf-8")
            assert "gemini-2.0-flash" in content
            assert "gpt-4o" in content

    def test_empty_reports(self) -> None:
        """Test comparison with empty reports list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_comparison_report([], tmpdir)
            content = Path(path).read_text(encoding="utf-8")
            assert "No reports" in content


class TestFmtPct:
    """Tests for _fmt_pct helper."""

    def test_typical_percentage(self) -> None:
        """Test formatting a typical percentage."""
        assert _fmt_pct(Decimal("0.925")) == "92.50%"

    def test_zero_percentage(self) -> None:
        """Test formatting zero."""
        assert _fmt_pct(Decimal("0")) == "0.00%"

    def test_full_percentage(self) -> None:
        """Test formatting 100%."""
        assert _fmt_pct(Decimal("1")) == "100.00%"
