"""Tests for the architecture study runner (fake designs, real temp databases)."""

from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from finance_ai.evaluation.architectures import runner
from finance_ai.evaluation.architectures.design_context import (
    DesignContext,
    DesignError,
    DesignOutcome,
)
from finance_ai.evaluation.architectures.runner import (
    StudyConfig,
    StudyPaused,
    is_provider_unavailable,
    plan_runs,
    read_log,
    run_study,
)
from finance_ai.evaluation.architectures.task_models import ArchitectureTask

from .conftest import fake_model

TASKS = [
    ArchitectureTask(
        task_id="exp_01",
        query="กาแฟ 80",
        category="single",
        acceptable_agents=["expense"],
    ),
    ArchitectureTask(task_id="vag_01", query="เงิน", category="vague"),
]


def _config(tmp_path: Path, rounds: int = 1) -> StudyConfig:
    """Build a small study config writing into tmp_path."""
    return StudyConfig(
        model_label="fake:model",
        output_dir=tmp_path,
        rounds=rounds,
        designs=["hub_spoke", "single_agent"],
        as_of=date(2026, 11, 2),
    )


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make retries instant."""
    monkeypatch.setattr(
        "finance_ai.evaluation.architectures.runner.time.sleep", lambda _seconds: None
    )


class TestPlanRuns:
    """Every (task, design, round) appears once, shuffled per round."""

    def test_covers_every_combination(self) -> None:
        """2 tasks × 2 designs × 3 rounds = 12 unique runs."""
        planned = plan_runs(TASKS, ["hub_spoke", "single_agent"], rounds=3, seed=1)
        assert len({run.key for run in planned}) == 12

    def test_rounds_stay_in_order(self) -> None:
        """All of round 0 comes before round 1."""
        rounds = [run.round_index for run in plan_runs(TASKS, ["hub_spoke"], rounds=2, seed=1)]
        assert rounds == sorted(rounds)

    def test_same_seed_same_order(self) -> None:
        """The shuffle is repeatable."""
        first = [run.key for run in plan_runs(TASKS, ["hub_spoke", "single_agent"], 2, 7)]
        second = [run.key for run in plan_runs(TASKS, ["hub_spoke", "single_agent"], 2, 7)]
        assert first == second

    def test_task_index_kept(self) -> None:
        """Each run remembers its task's position for P2P rotation."""
        planned = plan_runs(TASKS, ["hub_spoke"], rounds=1, seed=1)
        assert {(run.task.task_id, run.task_index) for run in planned} == {
            ("exp_01", 0),
            ("vag_01", 1),
        }


class TestRunStudy:
    """End-to-end with fake designs."""

    def test_writes_one_row_per_run_and_resumes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second call skips finished runs."""
        calls: Counter[str] = Counter()

        def fake_design(query: str, _context: DesignContext) -> DesignOutcome:
            calls[query] += 1
            return DesignOutcome(answer="ต้องการอะไรคะ", route=["expense"])

        monkeypatch.setitem(runner.DESIGNS, "hub_spoke", fake_design)
        monkeypatch.setitem(runner.DESIGNS, "single_agent", fake_design)
        log_path = run_study(_config(tmp_path), TASKS, fake_model())
        assert len(read_log(log_path)) == 4
        run_study(_config(tmp_path), TASKS, fake_model())
        assert len(read_log(log_path)) == 4 and sum(calls.values()) == 4

    def test_errors_are_logged_not_raised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crashing design produces an error row with a failed score."""

        def broken(_query: str, _context: DesignContext) -> DesignOutcome:
            raise ValueError("bad reply")

        monkeypatch.setitem(runner.DESIGNS, "hub_spoke", broken)
        config = _config(tmp_path).model_copy(update={"designs": ["hub_spoke"]})
        rows = read_log(run_study(config, TASKS, fake_model()))
        assert all(row.run.error == "ValueError: bad reply" for row in rows)
        assert all(row.score.failure == "error" for row in rows)

    def test_transient_error_retried_from_scratch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 429 is retried; the logged run is the clean final attempt."""
        attempts: Counter[str] = Counter()

        def flaky(query: str, _context: DesignContext) -> DesignOutcome:
            attempts[query] += 1
            if attempts[query] == 1:
                raise RuntimeError("HTTP 429 rate limit")
            return DesignOutcome(answer="ok", route=["expense"])

        monkeypatch.setitem(runner.DESIGNS, "hub_spoke", flaky)
        config = _config(tmp_path).model_copy(update={"designs": ["hub_spoke"]})
        rows = read_log(run_study(config, TASKS, fake_model()))
        assert all(row.run.error is None for row in rows)
        assert attempts == {"กาแฟ 80": 2, "เงิน": 2}

    def test_missing_log_reads_empty(self, tmp_path: Path) -> None:
        """No log file means no rows."""
        assert not read_log(tmp_path / "none.jsonl")


class TestProviderFailures:
    """Provider outages pause the study instead of logging bad rows."""

    def test_server_error_retried(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An Ollama 500 is retried and the clean attempt is logged."""
        attempts: Counter[str] = Counter()

        def flaky(query: str, _context: DesignContext) -> DesignOutcome:
            attempts[query] += 1
            if attempts[query] == 1:
                raise RuntimeError("Internal Server Error (status code: 500)")
            return DesignOutcome(answer="ok", route=["expense"])

        monkeypatch.setitem(runner.DESIGNS, "hub_spoke", flaky)
        config = _config(tmp_path).model_copy(update={"designs": ["hub_spoke"]})
        rows = read_log(run_study(config, TASKS, fake_model()))
        assert len(rows) == 2 and all(row.run.error is None for row in rows)

    def test_persistent_rate_limit_pauses_without_logging(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After every retry fails, the study pauses and writes nothing for that run."""

        def limited(_query: str, _context: DesignContext) -> DesignOutcome:
            raise RuntimeError("HTTP 429 rate limit")

        monkeypatch.setitem(runner.DESIGNS, "hub_spoke", limited)
        config = _config(tmp_path).model_copy(update={"designs": ["hub_spoke"]})
        with pytest.raises(StudyPaused, match="Provider unavailable"):
            run_study(config, TASKS, fake_model())
        assert not read_log(tmp_path / "runs.jsonl")

    def test_persistent_design_error_is_logged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed answer from production code is retried, then logged as an error."""

        def failing(_query: str, _context: DesignContext) -> DesignOutcome:
            raise DesignError("ระบบเกิดข้อผิดพลาด")

        monkeypatch.setitem(runner.DESIGNS, "hub_spoke", failing)
        config = _config(tmp_path).model_copy(update={"designs": ["hub_spoke"]})
        rows = read_log(run_study(config, TASKS, fake_model()))
        assert len(rows) == 2 and all(row.score.failure == "error" for row in rows)

    def test_incomplete_last_line_skipped(self, tmp_path: Path) -> None:
        """A line cut short by a crash is ignored, so its run is repeated on resume."""
        log = tmp_path / "runs.jsonl"
        log.write_text('{"model_label": "x", "finished_at": "2026-11-02T00:00', encoding="utf-8")
        assert not read_log(log)

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("Internal Server Error (status code: 500)", True),
            ("429 Too Many Requests", True),
            ("ValueError: bad date", False),
        ],
    )
    def test_provider_unavailable_detection(self, message: str, expected: bool) -> None:
        """Rate limits and 5xx count as provider outages; other errors do not."""
        assert is_provider_unavailable(RuntimeError(message)) is expected
