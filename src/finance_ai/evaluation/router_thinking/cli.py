"""Command line for the router study.

Usage:
    uv run python -m finance_ai.evaluation.router_thinking.cli --pilot
    uv run python -m finance_ai.evaluation.router_thinking.cli --rounds 1
    uv run python -m finance_ai.evaluation.router_thinking.cli --report-only

Results: ``data/evaluation/results/router_thinking/<pilot|main>/``
(``calls.jsonl`` raw log, ``report.md``, ``summary.json``).
"""

import argparse
from pathlib import Path
from typing import Any, Optional

from finance_ai.evaluation.architectures.runner import StudyPaused
from finance_ai.evaluation.router_thinking.questions import RoutingQuestion, load_all_questions
from finance_ai.evaluation.router_thinking.report import write_report
from finance_ai.evaluation.router_thinking.runner import LOG_NAME, plan_calls, run_study
from finance_ai.evaluation.router_thinking.setups import SETUPS, build_router_model

RESULTS_ROOT = Path("data/evaluation/results/router_thinking")
PILOT_QUESTION_IDS = (
    "tax_16",
    "exp_13",
    "ast_16",
    "pln_16",
    "rec_05",
    "rpt_16",
    "gen_05",
    "oos_10",
    "vag_03",
    "his_04",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Router study: models × thinking levels")
    parser.add_argument("--rounds", type=int, default=1, help="repeats per question and setup")
    parser.add_argument("--pilot", action="store_true", help="10 hard questions × 4 setups")
    parser.add_argument("--report-only", action="store_true", help="rebuild report from the log")
    parser.add_argument("--output", type=Path, default=None, help="override the output folder")
    parser.add_argument(
        "--setups", nargs="+", choices=list(SETUPS), default=list(SETUPS), help="setups to run"
    )
    return parser


def select_questions(questions: list[RoutingQuestion], pilot: bool) -> list[RoutingQuestion]:
    """Return every question, or the ten pilot questions.

    Args:
        questions: All questions.
        pilot: Keep only PILOT_QUESTION_IDS.

    Returns:
        Questions to ask.

    Example:
        >>> select_questions([], pilot=True)
        []
    """
    if not pilot:
        return questions
    return [question for question in questions if question.question_id in PILOT_QUESTION_IDS]


def main(argv: Optional[list[str]] = None) -> Path:
    """Run the study (or rebuild its report) and print where the report is.

    Args:
        argv: Arguments (defaults to sys.argv).

    Returns:
        Path of report.md.
    """
    args = build_parser().parse_args(argv)
    folder = args.output or RESULTS_ROOT / ("pilot" if args.pilot else "main")
    log_path = folder / LOG_NAME
    if not args.report_only:
        questions = select_questions(load_all_questions(), args.pilot)
        setups = [SETUPS[name] for name in args.setups]
        models: dict[str, Any] = {setup.name: build_router_model(setup) for setup in setups}
        try:
            run_study(plan_calls(questions, setups, args.rounds), models, log_path)
        except StudyPaused as pause:
            print(f"Paused: {pause}")
            print("Run the same command again later to resume.")
    report_path = write_report(log_path)
    print(f"Report: {report_path}")
    return report_path


if __name__ == "__main__":
    main()
