"""Command line for the architecture study.

Usage:
    uv run python -m finance_ai.evaluation.architectures.study_cli --pilot
    uv run python -m finance_ai.evaluation.architectures.study_cli --rounds 3
    uv run python -m finance_ai.evaluation.architectures.study_cli --report-only

Results go to ``data/evaluation/results/architecture_v2/<model>[_pilot]/``:
``runs.jsonl`` (raw log), ``report.md`` (readable), ``summary.json``, ``runs.csv``.
"""

import argparse
import re
from pathlib import Path
from typing import Any, Optional

from finance_ai.evaluation.architectures.report import write_report
from finance_ai.evaluation.architectures.runner import (
    LOG_NAME,
    StudyConfig,
    StudyPaused,
    run_study,
)
from finance_ai.evaluation.architectures.task_models import (
    ArchitectureTask,
    load_architecture_tasks,
)

DEFAULT_MODEL = "ollama:gpt-oss:20b"
RESULTS_ROOT = Path("data/evaluation/results/architecture_v2")
PILOT_TASK_IDS = ("tax_01", "exp_01", "ast_01", "crs_06", "vag_01")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Architecture comparison study (4.11 A)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="provider:model for every design")
    parser.add_argument("--rounds", type=int, default=3, help="repeats per task and design")
    parser.add_argument("--pilot", action="store_true", help="5 tasks × 4 designs × 1 round")
    parser.add_argument("--report-only", action="store_true", help="rebuild report from the log")
    parser.add_argument("--output", type=Path, default=None, help="override the output folder")
    parser.add_argument("--tasks", nargs="+", default=None, help="only these task ids")
    return parser


def output_folder(model_label: str, pilot: bool, override: Optional[Path]) -> Path:
    """Return the results folder for a model.

    Args:
        model_label: ``provider:model``.
        pilot: Whether this is the pilot run.
        override: Explicit folder, if given.

    Returns:
        Folder path.

    Example:
        >>> output_folder("ollama:gpt-oss:20b", True, None).name
        'ollama_gpt-oss_20b_pilot'
    """
    if override is not None:
        return override
    safe = re.sub(r"[^A-Za-z0-9.\-]+", "_", model_label)
    return RESULTS_ROOT / (f"{safe}_pilot" if pilot else safe)


def select_tasks(tasks: list[ArchitectureTask], pilot: bool) -> list[ArchitectureTask]:
    """Return all tasks, or the five pilot tasks.

    Args:
        tasks: Every task in file order.
        pilot: Keep only PILOT_TASK_IDS.

    Returns:
        Tasks to run (file order kept, so P2P rotation matches the full run).
    """
    if not pilot:
        return tasks
    return [task for task in tasks if task.task_id in PILOT_TASK_IDS]


def build_chat_model(model_label: str) -> Any:
    """Create the shared chat model from ``provider:model``.

    Args:
        model_label: e.g. ``ollama:gpt-oss:20b``.

    Returns:
        LangChain chat model.
    """
    # pylint: disable=import-outside-toplevel
    from finance_ai.agents.llm_factory import create_chat_model
    from finance_ai.evaluation.cli import _build_settings  # pylint: disable=protected-access
    from finance_ai.evaluation.model_comparison import parse_model_spec

    spec = parse_model_spec(model_label)
    return create_chat_model(settings=_build_settings(spec.provider, spec.model_name))


def main(argv: Optional[list[str]] = None) -> Path:
    """Run the study (or rebuild its report) and print where the report is.

    Args:
        argv: Arguments (defaults to sys.argv).

    Returns:
        Path of report.md.
    """
    args = build_parser().parse_args(argv)
    task_set = load_architecture_tasks()
    tasks_by_id = {task.task_id: task for task in task_set.tasks}
    folder = output_folder(args.model, args.pilot, args.output)
    if not args.report_only:
        config = StudyConfig(
            model_label=args.model,
            output_dir=folder,
            rounds=1 if args.pilot else args.rounds,
            task_ids=args.tasks,
        )
        try:
            run_study(
                config,
                select_tasks(task_set.tasks, args.pilot),
                build_chat_model(args.model),
            )
        except StudyPaused as pause:
            print(f"Paused: {pause}")
            print("Run the same command again later to resume.")
    report_path = write_report(folder / LOG_NAME, tasks_by_id)
    print(f"Report: {report_path}")
    return report_path


if __name__ == "__main__":
    main()
