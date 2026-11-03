"""Score one architecture run against its task (deterministic, no LLM judge).

Success needs the right route (routed designs), every required tool group,
every expected answer value and every expected database row. Vague tasks
succeed only when the system asks the user back and writes nothing.
"""

import re
from typing import Literal, Optional

from pydantic import BaseModel

from finance_ai.evaluation.architectures.run_result import ArchitectureRun
from finance_ai.evaluation.architectures.task_models import ArchitectureTask

FailureCategory = Literal[
    "error",
    "acted_without_asking",
    "did_not_ask_back",
    "wrong_route",
    "empty_answer",
    "missing_tool",
    "wrong_value",
    "wrong_write",
    "audit_fail",
]

WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "add_expense",
        "add_income",
        "add_portfolio_holding",
        "create_financial_goal",
        "update_goal_progress",
        "update_savings_goal_cross",
    }
)

_DIGIT_GROUP_SEPARATOR = re.compile(r"(?<=\d)[,\s\u00a0\u2009\u202f](?=\d)")
_HYPHEN_VARIANTS = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212]")
_QUESTION_PATTERN = re.compile(r"\?|ไหม|อะไร|หรือเปล่า|แบบไหน|ด้านไหน|เรื่องไหน|เลือก|ต้องการ")


class TaskScore(BaseModel):
    """Outcome of one run.

    Example:
        >>> TaskScore(success=True, route_ok=True).failure is None
        True
    """

    success: bool
    route_ok: Optional[bool] = None
    failure: Optional[FailureCategory] = None


def normalize_text(text: str) -> str:
    """Remove thousands separators between digits and unify hyphen variants.

    Models write 21,500, 21 500 or 21<narrow no-break space>500, and KT-RMFX with a
    plain or non-breaking hyphen; all forms match.

    Args:
        text: Answer or check text.

    Returns:
        Normalized text.

    Example:
        >>> normalize_text("ภาษี 21 500 บาท KT\u2011RMFX")
        'ภาษี 21500 บาท KT-RMFX'
    """
    return _HYPHEN_VARIANTS.sub("-", _DIGIT_GROUP_SEPARATOR.sub("", text))


def failed_answer_checks(answer: str, checks: list[str]) -> list[str]:
    """Return the checks whose values do not appear in the answer.

    Args:
        answer: Final answer text.
        checks: Values; ``|`` separates alternatives; digit grouping is ignored.

    Returns:
        Checks with no matching alternative.

    Example:
        >>> failed_answer_checks("ภาษี 21 500 บาท", ["21,500"])
        []
    """
    text = normalize_text(answer)
    return [
        check
        for check in checks
        if not any(normalize_text(option) in text for option in check.split("|"))
    ]


def missing_tool_groups(called: list[str], groups: list[list[str]]) -> list[list[str]]:
    """Return the tool groups with no call.

    Args:
        called: Tool names called during the run.
        groups: Each group needs at least one of its tools.

    Returns:
        Groups not satisfied.

    Example:
        >>> missing_tool_groups(["a"], [["a"], ["b"]])
        [['b']]
    """
    called_set = set(called)
    return [group for group in groups if not called_set.intersection(group)]


def answer_asks_back(answer: str) -> bool:
    """Return True when the answer asks the user a question.

    Args:
        answer: Final answer text.

    Returns:
        True if a question mark or a Thai question word appears.

    Example:
        >>> answer_asks_back("ต้องการให้ช่วยเรื่องอะไรคะ")
        True
    """
    return bool(_QUESTION_PATTERN.search(answer))


def score_run(task: ArchitectureTask, run: ArchitectureRun, db_results: list[bool]) -> TaskScore:
    """Score one run.

    Args:
        task: The task definition.
        run: What the design did.
        db_results: One pass/fail per ``task.db_checks`` entry, in order.

    Returns:
        TaskScore with success, route correctness and the first failure found.

    Example:
        >>> from finance_ai.evaluation.architectures.task_models import ArchitectureTask
        >>> task = ArchitectureTask(task_id="v", query="เงิน", category="vague")
        >>> run = ArchitectureRun(task_id="v", design="hub_spoke", round_index=0,
        ...                       route=["clarify"])
        >>> score_run(task, run, []).success
        True
    """
    if run.error:
        return TaskScore(success=False, failure="error")
    if task.expects_clarify:
        return _score_vague(run)
    route_ok = _route_ok(task, run)
    failure = _first_failure(task, run, db_results, route_ok)
    return TaskScore(success=failure is None, route_ok=route_ok, failure=failure)


def _score_vague(run: ArchitectureRun) -> TaskScore:
    """Score a task whose right outcome is asking back."""
    routed = run.design != "single_agent"
    if WRITE_TOOLS.intersection(run.metrics.tool_names()):
        return TaskScore(
            success=False, route_ok=False if routed else None, failure="acted_without_asking"
        )
    asked = run.asked_back_by_router or answer_asks_back(run.answer)
    return TaskScore(
        success=asked,
        route_ok=asked if routed else None,
        failure=None if asked else "did_not_ask_back",
    )


def _route_ok(task: ArchitectureTask, run: ArchitectureRun) -> Optional[bool]:
    """Return route correctness; None for the single agent (no routing)."""
    if run.design == "single_agent":
        return None
    return run.final_agent in task.acceptable_agents


def _first_failure(
    task: ArchitectureTask,
    run: ArchitectureRun,
    db_results: list[bool],
    route_ok: Optional[bool],
) -> Optional[FailureCategory]:
    """Return the first failed requirement, checked in pipeline order."""
    if route_ok is False:
        return "wrong_route"
    if not run.answer.strip():
        return "empty_answer"
    if missing_tool_groups(run.metrics.tool_names(), task.required_tools):
        return "missing_tool"
    if failed_answer_checks(run.answer, task.answer_checks):
        return "wrong_value"
    if not all(db_results):
        return "wrong_write"
    return None
