"""Tests for scoring one architecture run against its task."""

import pytest

from finance_ai.evaluation.architectures.run_result import ArchitectureRun, DesignName
from finance_ai.evaluation.architectures.scoring import (
    answer_asks_back,
    failed_answer_checks,
    missing_tool_groups,
    score_run,
)
from finance_ai.evaluation.architectures.task_models import ArchitectureTask, DbCheck

from .conftest import make_run

TAX_TASK = ArchitectureTask(
    task_id="tax_01",
    query="ภาษี",
    category="single",
    acceptable_agents=["tax"],
    required_tools=[["calculate_thai_tax"]],
    answer_checks=["21,500|21500"],
)
VAGUE_TASK = ArchitectureTask(task_id="vag_01", query="เงิน", category="vague")
WRITE_TASK = ArchitectureTask(
    task_id="exp_01",
    query="กาแฟ 80",
    category="single",
    acceptable_agents=["expense"],
    required_tools=[["add_expense"]],
    db_checks=[DbCheck(kind="expense", amount="80")],  # type: ignore[arg-type]
)


def _run(
    answer: str = "",
    route: list[str] | None = None,
    tools: list[str] | None = None,
    design: DesignName = "hub_spoke",
    error: str | None = None,
) -> ArchitectureRun:
    """Build a run with the given answer, route and tool calls."""
    return make_run(answer, route, tools, design, error=error)


class TestHelpers:
    """Small matching helpers."""

    def test_answer_check_ignores_commas(self) -> None:
        """21500 matches a check written as 21,500."""
        assert not failed_answer_checks("ภาษี 21500 บาท", ["21,500"])

    @pytest.mark.parametrize("written", ["21 500", "21 500", "21 500", "21 500"])
    def test_answer_check_ignores_space_separators(self, written: str) -> None:
        """Space-like thousands separators match too."""
        assert not failed_answer_checks(f"ภาษี {written} บาท", ["21,500"])

    def test_spaces_between_separate_numbers_kept(self) -> None:
        """Two separate numbers are not glued into one big match by accident."""
        assert failed_answer_checks("อายุ 30 ปี 500 บาท", ["30500"]) == ["30500"]

    def test_hyphen_variants_match(self) -> None:
        """A non-breaking hyphen matches a normal hyphen."""
        assert not failed_answer_checks("กองทุน KT‑RMFX", ["KT-RMFX"])

    def test_answer_check_accepts_alternatives(self) -> None:
        """Any alternative separated by | is enough."""
        assert not failed_answer_checks("ภาษี 21,500 บาท", ["99|21500"])

    def test_answer_check_reports_missing_value(self) -> None:
        """A missing value is returned as failed."""
        assert failed_answer_checks("ไม่มีตัวเลข", ["21,500"]) == ["21,500"]

    def test_tool_group_satisfied_by_any_member(self) -> None:
        """One call from a group satisfies it."""
        assert not missing_tool_groups(["b"], [["a", "b"]])

    def test_missing_tool_group_reported(self) -> None:
        """A group with no call is returned."""
        assert missing_tool_groups(["a"], [["a"], ["c", "d"]]) == [["c", "d"]]

    def test_question_detected(self) -> None:
        """A Thai question asking the user to choose counts as asking back."""
        assert answer_asks_back("ต้องการให้ช่วยเรื่องอะไรคะ")
        assert answer_asks_back("Which one?")

    def test_statement_is_not_a_question(self) -> None:
        """A plain statement does not count as asking back."""
        assert not answer_asks_back("บันทึกรายจ่าย 80 บาทแล้ว")


class TestScoreRun:
    """score_run combines tools, answer values, database rows and routing."""

    def test_correct_run_succeeds(self) -> None:
        """Right route, tool and value → success."""
        score = score_run(TAX_TASK, _run("ภาษี 21,500 บาท", ["tax"], ["calculate_thai_tax"]), [])
        assert score.success and score.route_ok and score.failure is None

    def test_wrong_route_is_the_failure(self) -> None:
        """A wrong agent is reported as wrong_route even if text looks right."""
        score = score_run(TAX_TASK, _run("21,500", ["planning"], ["calculate_thai_tax"]), [])
        assert not score.success and score.failure == "wrong_route"

    def test_empty_answer(self) -> None:
        """Right agent but a blank answer → empty_answer (model gave nothing)."""
        score = score_run(TAX_TASK, _run("  ", ["tax"], []), [])
        assert score.failure == "empty_answer"

    def test_missing_tool(self) -> None:
        """Right agent but no required tool → missing_tool."""
        score = score_run(TAX_TASK, _run("21,500", ["tax"], []), [])
        assert score.failure == "missing_tool"

    def test_wrong_value(self) -> None:
        """Tool called but the value is absent → wrong_value."""
        score = score_run(TAX_TASK, _run("ภาษี 20,000", ["tax"], ["calculate_thai_tax"]), [])
        assert score.failure == "wrong_value"

    def test_failed_db_check(self) -> None:
        """Tool called but the row was not written → wrong_write."""
        score = score_run(WRITE_TASK, _run("ok", ["expense"], ["add_expense"]), [False])
        assert score.failure == "wrong_write"

    def test_error_wins(self) -> None:
        """A crashed run is an error, whatever else happened."""
        score = score_run(TAX_TASK, _run(route=["tax"], error="boom"), [])
        assert score.failure == "error"

    def test_single_agent_has_no_route_score(self) -> None:
        """The single agent is scored on tools and values only."""
        run = _run("21,500", [], ["calculate_thai_tax"], design="single_agent")
        score = score_run(TAX_TASK, run, [])
        assert score.success and score.route_ok is None


class TestVagueTasks:
    """Vague tasks succeed only when the system asks back and writes nothing."""

    def test_router_clarify_succeeds(self) -> None:
        """A routing step that asks back passes."""
        assert score_run(VAGUE_TASK, _run("เลือกหัวข้อ", ["clarify"]), []).success

    def test_agent_question_succeeds(self) -> None:
        """An agent that answers with a question passes, with no route score."""
        run = _run("ต้องการให้ช่วยเรื่องอะไรคะ", [], design="single_agent")
        score = score_run(VAGUE_TASK, run, [])
        assert score.success and score.route_ok is None

    def test_writing_data_fails(self) -> None:
        """Writing data instead of asking fails."""
        run = _run("บันทึกแล้ว ต้องการอะไรเพิ่มไหม", ["expense"], ["add_expense"])
        assert score_run(VAGUE_TASK, run, []).failure == "acted_without_asking"

    def test_plain_answer_fails(self) -> None:
        """Answering without asking fails."""
        assert score_run(VAGUE_TASK, _run("เงินคือสื่อกลาง", ["general"]), []).failure == (
            "did_not_ask_back"
        )
