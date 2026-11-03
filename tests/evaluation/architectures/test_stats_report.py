"""Tests for study statistics and the report writer."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from finance_ai.evaluation.architectures.report import format_markdown, summarize, write_report
from finance_ai.evaluation.architectures.run_result import DesignName
from finance_ai.evaluation.architectures.runner import StudyLogRow
from finance_ai.evaluation.architectures.scoring import TaskScore
from finance_ai.evaluation.architectures.task_models import ArchitectureTask
from finance_ai.evaluation.architectures.stats import (
    bootstrap_median_interval,
    exact_binomial_p,
    pass_hat_k,
    percentile,
    wilson_interval,
)

from .conftest import make_run


class TestStats:
    """Pure statistics helpers."""

    def test_wilson_known_value(self) -> None:
        """8/10 gives the textbook Wilson interval."""
        low, high = wilson_interval(8, 10)
        assert (round(low, 3), round(high, 3)) == (0.49, 0.943)

    @pytest.mark.parametrize(("successes", "total"), [(0, 10), (10, 10), (0, 0)])
    def test_wilson_stays_in_bounds(self, successes: int, total: int) -> None:
        """Edge cases stay inside [0, 1]."""
        low, high = wilson_interval(successes, total)
        assert 0.0 <= low <= high <= 1.0

    def test_binomial_known_value(self) -> None:
        """9 wins vs 1 loss → p ≈ 0.0215."""
        assert round(exact_binomial_p(9, 1), 4) == 0.0215

    def test_binomial_no_pairs(self) -> None:
        """No informative pairs → p = 1."""
        assert exact_binomial_p(0, 0) == 1.0

    def test_binomial_symmetric(self) -> None:
        """Swapping wins and losses gives the same p."""
        assert exact_binomial_p(3, 7) == exact_binomial_p(7, 3)

    def test_percentile_interpolates(self) -> None:
        """Median of four values is the midpoint of the middle two."""
        assert percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.5
        assert percentile([], 0.5) == 0.0

    def test_bootstrap_repeatable_and_bounded(self) -> None:
        """Same seed → same interval, inside the data range."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert bootstrap_median_interval(data) == bootstrap_median_interval(data)
        low, high = bootstrap_median_interval(data)
        assert 1.0 <= low <= high <= 5.0
        assert bootstrap_median_interval([]) == (0.0, 0.0)

    def test_pass_hat_k(self) -> None:
        """Only tasks solved in every round count; incomplete tasks are skipped."""
        outcomes = [("a", True), ("a", True), ("b", True), ("b", False), ("c", True)]
        assert pass_hat_k(outcomes, 2) == 0.5
        assert pass_hat_k([], 2) == 0.0


def _row(
    design: DesignName,
    task_id: str,
    latency: float,
    answer: str = "a",
    route: list[str] | None = None,
) -> StudyLogRow:
    """Build one log row (scores are recomputed by the report)."""
    return StudyLogRow(
        model_label="fake:model",
        finished_at=datetime(2026, 11, 2, tzinfo=timezone.utc),
        run=make_run(answer, route, None, design, task_id=task_id, latency=latency),
        score=TaskScore(success=False),
    )


TASKS = {
    "t1": ArchitectureTask(
        task_id="t1", query="ภาษี", category="single", acceptable_agents=["tax"]
    ),
    "t2": ArchitectureTask(task_id="t2", query="เงิน", category="vague"),
}
ROWS = [
    _row("hub_spoke", "t1", 2.0, route=["tax"]),
    _row("hub_spoke", "t2", 3.0, route=["tax"]),
    _row("single_agent", "t1", 5.0),
    _row("single_agent", "t2", 6.0, answer="ต้องการอะไรคะ"),
]


class TestReport:
    """Summaries, paired comparison and written files."""

    def test_design_summaries(self) -> None:
        """Rates, failures and route accuracy per design."""
        report = summarize(ROWS, TASKS)
        hub = next(d for d in report.designs if d.design == "hub_spoke")
        single = next(d for d in report.designs if d.design == "single_agent")
        assert hub.success_rate == 0.5 and hub.failures == {"did_not_ask_back": 1}
        assert hub.success_by_category == {"single": 1.0, "vague": 0.0}
        assert single.route_accuracy is None and single.mean_llm_calls == 2

    def test_paired_comparison_against_hub(self) -> None:
        """single_agent is paired with hub_spoke on the same task and round."""
        comparison = summarize(ROWS, TASKS).comparisons[0]
        assert comparison.design == "single_agent" and comparison.pairs == 2
        assert (comparison.design_only_success, comparison.baseline_only_success) == (1, 0)
        assert comparison.latency_median_difference == 3.0

    def test_markdown_has_all_sections(self) -> None:
        """Every section heading is rendered."""
        text = format_markdown(summarize(ROWS, TASKS))
        for heading in ("Main results", "Success by task type", "Failure types", "Paired tests"):
            assert heading in text

    def test_no_baseline_no_comparison(self) -> None:
        """Without hub_spoke rows there is nothing to pair."""
        report = summarize([row for row in ROWS if row.run.design != "hub_spoke"], TASKS)
        assert not report.comparisons
        assert "Baseline not in this run" in format_markdown(report)

    def test_scores_recomputed_from_facts(self) -> None:
        """Stored scores are ignored; the current rules decide."""
        report = summarize(ROWS, TASKS)
        single = next(d for d in report.designs if d.design == "single_agent")
        assert single.success_rate == 1.0

    def test_write_report_files(self, tmp_path: Path) -> None:
        """summary.json, runs.csv and report.md are written next to the log."""
        log_path = tmp_path / "runs.jsonl"
        log_path.write_text("".join(row.model_dump_json() + "\n" for row in ROWS), encoding="utf-8")
        report_path = write_report(log_path, TASKS)
        assert report_path.read_text(encoding="utf-8").startswith("# Architecture study")
        assert (tmp_path / "summary.json").exists()
        assert len((tmp_path / "runs.csv").read_text(encoding="utf-8").splitlines()) == 5
