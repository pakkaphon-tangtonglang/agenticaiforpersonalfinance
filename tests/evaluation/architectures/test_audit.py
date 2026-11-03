"""Tests for manual audit verdicts."""

from datetime import datetime, timezone
from pathlib import Path

from finance_ai.evaluation.architectures.audit import AuditVerdict, apply_audit, load_audit
from finance_ai.evaluation.architectures.report import format_markdown, summarize, write_report
from finance_ai.evaluation.architectures.runner import StudyLogRow
from finance_ai.evaluation.architectures.scoring import TaskScore
from finance_ai.evaluation.architectures.task_models import ArchitectureTask

from .conftest import make_run

TASKS = {
    "rpt_05": ArchitectureTask(
        task_id="rpt_05",
        query="รายจ่ายเดือนที่แล้ว",
        category="single",
        acceptable_agents=["report"],
    )
}
VERDICT = AuditVerdict(
    task_id="rpt_05",
    design="hub_spoke",
    round_index=0,
    success=False,
    reason="Reported September instead of last month",
)


def _row() -> StudyLogRow:
    """A run the automatic checks pass (right route, no requirements)."""
    return StudyLogRow(
        model_label="fake:model",
        finished_at=datetime(2026, 11, 2, tzinfo=timezone.utc),
        run=make_run("รายจ่ายเดือนกันยายน", ["report"], task_id="rpt_05"),
        score=TaskScore(success=True, route_ok=True),
    )


class TestAudit:
    """Verdicts override scores and show up in the report."""

    def test_missing_file_means_no_verdicts(self, tmp_path: Path) -> None:
        """No audit file → no overrides."""
        assert not load_audit(tmp_path / "audit.yaml")

    def test_loads_verdicts(self, tmp_path: Path) -> None:
        """Verdicts are read from the verdicts list."""
        path = tmp_path / "audit.yaml"
        path.write_text(
            "verdicts:\n  - {task_id: rpt_05, design: hub_spoke, round_index: 0, "
            "success: false, reason: wrong month}\n",
            encoding="utf-8",
        )
        assert load_audit(path)[0].key == ("rpt_05", "hub_spoke", 0)

    def test_override_keeps_route_result(self) -> None:
        """A failing verdict replaces success but keeps the measured route."""
        audited = apply_audit([_row()], [VERDICT])[0]
        assert not audited.score.success
        assert audited.score.failure == "audit_fail" and audited.score.route_ok is True

    def test_unaudited_rows_unchanged(self) -> None:
        """Rows without a verdict keep their score."""
        assert apply_audit([_row()], [])[0].score.success

    def test_report_applies_and_lists_verdicts(self) -> None:
        """The summary counts the override and the Markdown lists its reason."""
        report = summarize([_row()], TASKS, [VERDICT])
        assert report.designs[0].success_rate == 0.0
        assert "Reported September" in format_markdown(report)

    def test_write_report_reads_audit_file(self, tmp_path: Path) -> None:
        """write_report picks up audit.yaml next to the log."""
        (tmp_path / "runs.jsonl").write_text(_row().model_dump_json() + "\n", encoding="utf-8")
        (tmp_path / "audit.yaml").write_text(
            "verdicts:\n  - {task_id: rpt_05, design: hub_spoke, round_index: 0, "
            "success: false, reason: wrong month}\n",
            encoding="utf-8",
        )
        text = write_report(tmp_path / "runs.jsonl", TASKS).read_text(encoding="utf-8")
        assert "wrong month" in text and "| hub_spoke | 1 | 0.0%" in text
