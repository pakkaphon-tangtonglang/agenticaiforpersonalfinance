"""Tests for tool-bypass benchmark reporter."""

import os
from decimal import Decimal

from finance_ai.evaluation.tool_bypass_benchmark import (
    ToolBypassBenchmarkResult,
    ToolBypassCaseResult,
)
from finance_ai.evaluation.tool_bypass_reporter import (
    _build_report,
    save_tool_bypass_report,
)


def _make_result() -> ToolBypassBenchmarkResult:
    """Create a minimal tool-bypass benchmark result."""
    return ToolBypassBenchmarkResult(
        model_name="deepseek/deepseek-chat",
        cases=[
            ToolBypassCaseResult(
                case_id="bypass_001",
                description="test case",
                query="test query",
                expected_tax=Decimal("1000"),
                notool_response="tax 1000",
                notool_extracted_tax=Decimal("1000"),
                notool_within_tolerance=True,
                notool_latency_seconds=0.5,
                tool_tax=Decimal("1000"),
                tool_within_tolerance=True,
            ),
        ],
        notool_accuracy_pct=100.0,
        tool_accuracy_pct=100.0,
    )


class TestToolBypassReporter:
    """Tests for tool-bypass markdown report generation."""

    def test_build_report_contains_all_sections(self) -> None:
        """Report contains header, summary, per-case, conclusion."""
        result = _make_result()
        report = _build_report(result)
        assert "ปัญหา Tool Bypass" in report
        assert "deepseek-chat" in report
        assert "test case" in report
        assert "100%" in report

    def test_save_tool_bypass_report_writes_file(self, tmp_path: str) -> None:
        """save_tool_bypass_report writes markdown to disk."""
        result = _make_result()
        output_path = os.path.join(tmp_path, "deepseek_tool_bypass.md")
        written_path = save_tool_bypass_report(result, output_path)
        assert os.path.exists(written_path)
        with open(written_path, encoding="utf-8") as file:
            content = file.read()
        assert "ปัญหา Tool Bypass" in content
