"""Run the architecture study: every (task, design, round) once, logged to JSONL.

Runs are shuffled per round with a fixed seed, each run gets a fresh
database copy, market data comes from the snapshot, and the log is the
single source of truth: restarting skips rows already written.
"""

import random
import time
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field, ValidationError

from finance_ai.agents.graph_cache import clear_graph_cache
from finance_ai.core.logging import get_logger
from finance_ai.evaluation.architectures.design_context import (
    DesignContext,
    DesignError,
    DesignOutcome,
)
from finance_ai.evaluation.architectures.environment import (
    SessionFactory,
    build_template_database,
    check_database,
    fresh_database,
    open_template,
)
from finance_ai.evaluation.architectures.hierarchical import run_hierarchical
from finance_ai.evaluation.architectures.hub_spoke import run_hub_multi, run_hub_spoke
from finance_ai.evaluation.architectures.market_snapshot import market_snapshot
from finance_ai.evaluation.architectures.peer_to_peer import run_peer_to_peer
from finance_ai.evaluation.architectures.run_metrics import collect_run_metrics
from finance_ai.evaluation.architectures.run_result import ArchitectureRun, DesignName
from finance_ai.evaluation.architectures.scoring import TaskScore, score_run
from finance_ai.evaluation.architectures.single_agent import run_single_agent
from finance_ai.evaluation.architectures.task_models import ArchitectureTask
from finance_ai.evaluation.llm_retry import _is_retryable  # pylint: disable=protected-access
from finance_ai.tools.demo_seed_service import DEMO_USER_ID

logger = get_logger(__name__)

DesignFunction = Callable[[str, DesignContext], DesignOutcome]
DESIGNS: dict[DesignName, DesignFunction] = {
    "single_agent": run_single_agent,
    "hub_spoke": run_hub_spoke,
    "hub_multi": run_hub_multi,
    "hierarchical": run_hierarchical,
    "peer_to_peer": run_peer_to_peer,
}
# The four designs of the original study; hub_multi runs only when asked for.
ORIGINAL_DESIGNS: tuple[DesignName, ...] = (
    "single_agent",
    "hub_spoke",
    "hierarchical",
    "peer_to_peer",
)
LOG_NAME = "runs.jsonl"
RETRY_MAX_ATTEMPTS = 4
RETRY_INITIAL_DELAY_SECONDS = 5.0
_PROVIDER_DOWN_MARKERS = (
    "status code: 500",
    "status code: 502",
    "status code: 503",
    "status code: 504",
    "internal server error",
    "service unavailable",
    "bad gateway",
)


class StudyPaused(RuntimeError):
    """The provider stayed rate-limited or down after every retry; resume later."""


def is_provider_unavailable(error: Exception) -> bool:
    """Return True for rate limits and provider server errors (not our code's fault).

    Args:
        error: Exception raised during a run.

    Returns:
        True when the run should be retried and, if it keeps failing, paused.

    Example:
        >>> is_provider_unavailable(RuntimeError("Internal Server Error (status code: 500)"))
        True
    """
    message = str(error).lower()
    return _is_retryable(error) or any(marker in message for marker in _PROVIDER_DOWN_MARKERS)


def _should_retry(error: Exception) -> bool:
    """Retry provider failures and errors production code reported as a failed answer."""
    return is_provider_unavailable(error) or isinstance(error, DesignError)


class StudyConfig(BaseModel):
    """Settings for one study run.

    Attributes:
        model_label: ``provider:model`` used for every design (logged).
        output_dir: Folder for the log, databases and reports.
        rounds: Repeats per (task, design).
        designs: Designs to run.
        seed: Shuffle seed (round r uses seed + r).
        as_of: Fixed reference date for the demo data.
        task_ids: Run only these tasks (None = all). Filtered after planning, so each
            task keeps its file position and peer-to-peer entry agent.
    """

    model_label: str
    output_dir: Path
    rounds: int = Field(default=3, ge=1)
    designs: list[DesignName] = Field(default_factory=lambda: list(ORIGINAL_DESIGNS))
    seed: int = 20261102
    as_of: date = Field(default_factory=date.today)
    task_ids: Optional[list[str]] = None


class StudyLogRow(BaseModel):
    """One line of the study log."""

    model_label: str
    finished_at: datetime
    run: ArchitectureRun
    db_results: list[bool] = Field(default_factory=list)
    score: TaskScore


class PlannedRun(BaseModel):
    """One (task, design, round) to execute."""

    task: ArchitectureTask
    task_index: int
    design: DesignName
    round_index: int

    @property
    def key(self) -> tuple[str, str, int]:
        """Return the resume key (task id, design, round)."""
        return (self.task.task_id, self.design, self.round_index)


def plan_runs(
    tasks: list[ArchitectureTask], designs: list[DesignName], rounds: int, seed: int
) -> list[PlannedRun]:
    """List every run, shuffled within each round with a fixed seed.

    Args:
        tasks: Tasks in file order (their position drives P2P entry rotation).
        designs: Designs to compare.
        rounds: Number of repeats.
        seed: Base shuffle seed.

    Returns:
        Planned runs, round 0 first.

    Example:
        >>> len(plan_runs(tasks, ["hub_spoke"], rounds=2, seed=1))  # doctest: +SKIP
        120
    """
    planned: list[PlannedRun] = []
    for round_index in range(rounds):
        batch = [
            PlannedRun(task=task, task_index=index, design=design, round_index=round_index)
            for index, task in enumerate(tasks)
            for design in designs
        ]
        random.Random(seed + round_index).shuffle(batch)
        planned.extend(batch)
    return planned


def read_log(log_path: Path) -> list[StudyLogRow]:
    """Read every complete row from a study log (missing file → empty list).

    A line cut short by a crash (e.g. disk full) is skipped with a warning,
    so the run it belonged to is simply executed again on resume.

    Args:
        log_path: JSONL file.

    Returns:
        Parsed rows in file order.
    """
    if not log_path.exists():
        return []
    rows: list[StudyLogRow] = []
    for number, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(StudyLogRow.model_validate_json(line))
        except ValidationError:
            logger.warning("Skipping incomplete line %d in %s", number, log_path)
    return rows


def execute_run(
    planned: PlannedRun, chat_model: BaseChatModel, template: Path, template_factory: SessionFactory
) -> tuple[ArchitectureRun, list[bool], TaskScore]:
    """Execute one planned run and score it, retrying transient provider errors.

    Every attempt starts over on a fresh database copy with fresh counters,
    so a retried run is counted and checked exactly like a clean one.

    Args:
        planned: The run to execute.
        chat_model: Model shared by every design.
        template: Template database path.
        template_factory: Sessions on the template (for database checks).

    Returns:
        The last attempt's run record, its database check results and its score.

    Raises:
        StudyPaused: When the provider is still rate-limited or down after every
            retry; nothing is logged for this run, so a resume repeats it.
    """
    delay = RETRY_INITIAL_DELAY_SECONDS
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        run, db_results, exc = _attempt(planned, chat_model, template, template_factory)
        if exc is None or not _should_retry(exc) or attempt == RETRY_MAX_ATTEMPTS:
            break
        logger.warning("Retrying %s after transient error (attempt %d)", planned.key, attempt)
        time.sleep(delay)
        delay *= 2
    if exc is not None and is_provider_unavailable(exc):
        raise StudyPaused(f"Provider unavailable at {planned.key}: {exc}")
    return run, db_results, score_run(planned.task, run, db_results)


def _attempt(
    planned: PlannedRun, chat_model: BaseChatModel, template: Path, template_factory: SessionFactory
) -> tuple[ArchitectureRun, list[bool], Optional[Exception]]:
    """Run once on a fresh database copy; return run, db results and any error."""
    run_id = "_".join(str(part) for part in planned.key)
    with fresh_database(template, run_id) as factory:
        context = DesignContext(
            chat_model, DEMO_USER_ID, factory, planned.task.task_id, planned.task_index
        )
        run, exc = _timed_run(planned, context)
        db_results = (
            check_database(factory, template_factory, planned.task.db_checks)
            if exc is None
            else [False] * len(planned.task.db_checks)
        )
    return run, db_results, exc


def _timed_run(
    planned: PlannedRun, context: DesignContext
) -> tuple[ArchitectureRun, Optional[Exception]]:
    """Run one design with counters, timing and error capture."""
    design = DESIGNS[planned.design]
    failure: Optional[Exception] = None
    outcome = DesignOutcome(answer="", route=[])
    started = time.perf_counter()
    with collect_run_metrics() as metrics:
        try:
            outcome = design(planned.task.query, context)
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            failure = exc
            logger.warning("Run %s failed: %s", planned.key, exc)
    run = ArchitectureRun(
        task_id=planned.task.task_id,
        design=planned.design,
        round_index=planned.round_index,
        answer=outcome.answer,
        route=outcome.route,
        metrics=metrics,
        latency_seconds=time.perf_counter() - started,
        error=f"{type(failure).__name__}: {failure}" if failure else None,
    )
    return run, failure


def run_study(
    config: StudyConfig, tasks: list[ArchitectureTask], chat_model: BaseChatModel
) -> Path:
    """Run (or resume) the study and return the log path.

    Args:
        config: Study settings.
        tasks: Tasks to run, in file order.
        chat_model: Model shared by every design.

    Returns:
        Path of the JSONL log.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.output_dir / LOG_NAME
    done = {(row.run.task_id, row.run.design, row.run.round_index) for row in read_log(log_path)}
    todo = [
        planned
        for planned in plan_runs(tasks, config.designs, config.rounds, config.seed)
        if planned.key not in done
        and (not config.task_ids or planned.task.task_id in config.task_ids)
    ]
    logger.info("Architecture study: %d runs done, %d to go", len(done), len(todo))
    template = build_template_database(config.output_dir / "db", config.as_of)
    clear_graph_cache()
    with market_snapshot(), open_template(template) as template_factory:
        for number, planned in enumerate(todo, start=1):
            run, db_results, score = execute_run(planned, chat_model, template, template_factory)
            _append_row(
                log_path,
                StudyLogRow(
                    model_label=config.model_label,
                    finished_at=datetime.now(timezone.utc),
                    run=run,
                    db_results=db_results,
                    score=score,
                ),
            )
            logger.info("[%d/%d] %s → success=%s", number, len(todo), planned.key, score.success)
    return log_path


def _append_row(log_path: Path, row: StudyLogRow) -> None:
    """Append one row to the log and flush it to disk."""
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(row.model_dump_json() + "\n")
