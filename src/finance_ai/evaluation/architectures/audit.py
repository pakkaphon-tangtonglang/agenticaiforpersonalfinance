"""Manual audit verdicts that override the automatic score of specific runs.

Every failed run is read by hand. When the automatic checks cannot see a
problem (or wrongly report one), a verdict in ``audit.yaml`` next to the log
records the decision and the reason; the report applies and lists them.
"""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from finance_ai.evaluation.architectures.run_result import DesignName
from finance_ai.evaluation.architectures.runner import StudyLogRow
from finance_ai.evaluation.architectures.scoring import FailureCategory, TaskScore

AUDIT_NAME = "audit.yaml"


class AuditVerdict(BaseModel):
    """A hand-checked decision for one run.

    Example:
        >>> AuditVerdict(task_id="rpt_05", design="single_agent", round_index=0,
        ...              success=False, reason="Reported September, not last month").success
        False
    """

    task_id: str
    design: DesignName
    round_index: int = Field(ge=0)
    success: bool
    reason: str = Field(min_length=1)

    @property
    def key(self) -> tuple[str, str, int]:
        """Return the run key (task id, design, round)."""
        return (self.task_id, self.design, self.round_index)


def load_audit(path: Path) -> list[AuditVerdict]:
    """Load verdicts from an audit file (missing file → no verdicts).

    Args:
        path: ``audit.yaml`` with a ``verdicts`` list.

    Returns:
        Validated verdicts.
    """
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [AuditVerdict.model_validate(item) for item in raw.get("verdicts", [])]


def apply_audit(rows: list[StudyLogRow], verdicts: list[AuditVerdict]) -> list[StudyLogRow]:
    """Replace the score of every audited run with the hand-checked verdict.

    Args:
        rows: Scored log rows.
        verdicts: Manual verdicts.

    Returns:
        Rows with audited scores; route correctness is kept as measured.
    """
    by_key = {verdict.key: verdict for verdict in verdicts}
    return [_override(row, by_key.get(_key(row))) for row in rows]


def _key(row: StudyLogRow) -> tuple[str, str, int]:
    """Return a row's run key."""
    return (row.run.task_id, row.run.design, row.run.round_index)


def _override(row: StudyLogRow, verdict: Optional[AuditVerdict]) -> StudyLogRow:
    """Apply one verdict to one row (no verdict → unchanged)."""
    if verdict is None:
        return row
    failure: Optional[FailureCategory] = None if verdict.success else "audit_fail"
    score = TaskScore(success=verdict.success, route_ok=row.score.route_ok, failure=failure)
    return row.model_copy(update={"score": score})
