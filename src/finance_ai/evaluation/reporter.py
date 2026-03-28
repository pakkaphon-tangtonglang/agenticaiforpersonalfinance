"""Report generation for evaluation results.

Outputs evaluation reports as JSON files, Markdown tables,
and console summaries using the rich library.
"""

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

from finance_ai.evaluation.models import EvaluationReport


def save_json_report(report: EvaluationReport, output_dir: str) -> str:
    """Save evaluation report as a JSON file.

    Args:
        report: Complete evaluation report.
        output_dir: Directory to save the report.

    Returns:
        Path to the saved JSON file.

    Example:
        >>> path = save_json_report(report, "data/evaluation/results")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{report.report_id}.json"
    file_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return str(file_path)


def save_markdown_report(report: EvaluationReport, output_dir: str) -> str:
    """Save evaluation report as a Markdown file with tables.

    Args:
        report: Complete evaluation report.
        output_dir: Directory to save the report.

    Returns:
        Path to the saved Markdown file.

    Example:
        >>> path = save_markdown_report(report, "data/evaluation/results")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{report.report_id}.md"
    content = _build_markdown_content(report)
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


def _build_markdown_content(report: EvaluationReport) -> str:
    """Build Markdown content from an evaluation report.

    Args:
        report: Complete evaluation report.

    Returns:
        Markdown string with tables and summary.
    """
    lines = [
        f"# Evaluation Report: {report.llm_model}",
        f"\n**Provider:** {report.llm_provider}",
        f"**Timestamp:** {report.timestamp}",
        f"**Report ID:** {report.report_id}\n",
        "## Summary\n",
        "| Metric | Value | Details |",
        "|--------|-------|---------|",
    ]
    lines.extend(_routing_row(report))
    lines.extend(_rag_row(report))
    lines.extend(_accuracy_row(report))
    lines.extend(_hallucination_row(report))
    lines.extend(_quality_row(report))
    lines.extend(_performance_row(report))
    return "\n".join(lines) + "\n"


def _routing_row(report: EvaluationReport) -> list[str]:
    """Build routing summary row.

    Args:
        report: Evaluation report.

    Returns:
        List of Markdown table row strings.
    """
    if report.routing is None:
        return []
    r = report.routing
    pct = _fmt_pct(r.accuracy)
    return [f"| Routing Accuracy | {pct} | {r.correct_count}/{r.total_cases} correct |"]


def _rag_row(report: EvaluationReport) -> list[str]:
    """Build RAG retrieval summary rows.

    Args:
        report: Evaluation report.

    Returns:
        List of Markdown table row strings.
    """
    if report.rag_retrieval is None:
        return []
    r = report.rag_retrieval
    return [
        f"| RAG Precision@k | {r.mean_precision_at_k:.2f} | {r.total_cases} queries |",
        f"| RAG MRR | {r.mean_reciprocal_rank:.2f} | {r.total_cases} queries |",
    ]


def _accuracy_row(report: EvaluationReport) -> list[str]:
    """Build tax accuracy summary row.

    Args:
        report: Evaluation report.

    Returns:
        List of Markdown table row strings.
    """
    if report.tax_accuracy is None:
        return []
    r = report.tax_accuracy
    pct = _fmt_pct(r.accuracy_rate)
    detail = f"{r.within_tolerance_count}/{r.total_cases} within tolerance"
    return [f"| Tax Accuracy | {pct} | {detail} |"]


def _hallucination_row(report: EvaluationReport) -> list[str]:
    """Build hallucination compliance summary row.

    Args:
        report: Evaluation report.

    Returns:
        List of Markdown table row strings.
    """
    if report.hallucination is None:
        return []
    r = report.hallucination
    pct = _fmt_pct(r.compliance_rate)
    return [f"| Hallucination Compliance | {pct} | {r.compliant_count}/{r.total_cases} compliant |"]


def _quality_row(report: EvaluationReport) -> list[str]:
    """Build quality score summary row.

    Args:
        report: Evaluation report.

    Returns:
        List of Markdown table row strings.
    """
    if report.quality is None:
        return []
    r = report.quality
    return [f"| Quality (overall) | {r.mean_overall}/5 | {r.total_cases} queries |"]


def _performance_row(report: EvaluationReport) -> list[str]:
    """Build performance summary rows.

    Args:
        report: Evaluation report.

    Returns:
        List of Markdown table row strings.
    """
    if report.performance is None:
        return []
    r = report.performance
    return [
        f"| Mean Latency | {r.mean_total_latency:.2f}s | p95={r.p95_latency:.2f}s |",
        f"| Total Cost | ${r.total_estimated_cost_usd} | {r.total_queries} queries |",
    ]


def _fmt_pct(value: Decimal) -> str:
    """Format a decimal as a percentage string.

    Args:
        value: Decimal value (0.0 to 1.0).

    Returns:
        Percentage string like '92.50%'.
    """
    return f"{float(value) * 100:.2f}%"


def save_comparison_report(
    reports: list[EvaluationReport],
    output_dir: str,
) -> str:
    """Save a multi-model comparison report as Markdown.

    Args:
        reports: List of evaluation reports (one per model).
        output_dir: Directory to save the report.

    Returns:
        Path to the saved comparison Markdown file.

    Example:
        >>> path = save_comparison_report(reports, "data/evaluation/results")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prefix = reports[0].report_id[:8] if reports else "empty"
    file_path = output_path / f"comparison_{prefix}.md"
    content = _build_comparison_content(reports)
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


def _build_comparison_content(
    reports: list[EvaluationReport],
) -> str:
    """Build comparison Markdown content.

    Args:
        reports: List of evaluation reports to compare.

    Returns:
        Markdown string with comparison table.
    """
    if not reports:
        return "# No reports to compare\n"
    models = [r.llm_model for r in reports]
    header = "| Metric | " + " | ".join(models) + " |"
    separator = "|--------|" + "|".join("--------" for _ in models) + "|"
    rows = [
        _comparison_row("Routing Accuracy", reports, _get_routing_accuracy),
        _comparison_row("RAG MRR", reports, _get_rag_mrr),
        _comparison_row("Tax Accuracy", reports, _get_tax_accuracy),
        _comparison_row("Hallucination Compliance", reports, _get_hallucination_rate),
        _comparison_row("Quality (overall)", reports, _get_quality_overall),
        _comparison_row("Mean Latency", reports, _get_mean_latency),
    ]
    lines = ["# Model Comparison Report\n", header, separator]
    lines.extend(row for row in rows if row)
    return "\n".join(lines) + "\n"


def _comparison_row(
    metric: str,
    reports: list[EvaluationReport],
    extractor: "Callable[[EvaluationReport], str]",
) -> str:
    """Build a single comparison table row.

    Args:
        metric: Name of the metric.
        reports: List of reports.
        extractor: Function to extract metric value from a report.

    Returns:
        Markdown table row string.
    """
    values = [extractor(r) for r in reports]
    return f"| {metric} | " + " | ".join(values) + " |"


# Extractor functions for comparison table


def _get_routing_accuracy(report: EvaluationReport) -> str:
    """Extract routing accuracy string from report."""
    if report.routing is None:
        return "N/A"
    return _fmt_pct(report.routing.accuracy)


def _get_rag_mrr(report: EvaluationReport) -> str:
    """Extract RAG MRR string from report."""
    if report.rag_retrieval is None:
        return "N/A"
    return f"{report.rag_retrieval.mean_reciprocal_rank:.2f}"


def _get_tax_accuracy(report: EvaluationReport) -> str:
    """Extract tax accuracy string from report."""
    if report.tax_accuracy is None:
        return "N/A"
    return _fmt_pct(report.tax_accuracy.accuracy_rate)


def _get_hallucination_rate(report: EvaluationReport) -> str:
    """Extract hallucination compliance string from report."""
    if report.hallucination is None:
        return "N/A"
    return _fmt_pct(report.hallucination.compliance_rate)


def _get_quality_overall(report: EvaluationReport) -> str:
    """Extract quality overall score string from report."""
    if report.quality is None:
        return "N/A"
    return f"{report.quality.mean_overall}/5"


def _get_mean_latency(report: EvaluationReport) -> str:
    """Extract mean latency string from report."""
    if report.performance is None:
        return "N/A"
    return f"{report.performance.mean_total_latency:.2f}s"
