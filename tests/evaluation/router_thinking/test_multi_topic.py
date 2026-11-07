"""Tests for the multi-topic routing test."""

from pathlib import Path

import pytest

from finance_ai.evaluation.router_thinking.multi_topic import (
    MultiTopicQuestion,
    as_routing_question,
    load_multi_topic_questions,
    score_question,
    summarize,
    write_report,
)
from finance_ai.evaluation.router_thinking.runner import LOG_NAME

from .conftest import make_row


def question(
    expected: list[str], kind: str = "multi", question_id: str = "m"
) -> MultiTopicQuestion:
    """A multi-topic question."""
    return MultiTopicQuestion(question_id=question_id, query="q", expected=expected, kind=kind)  # type: ignore[arg-type]


def routed(*destinations: str, question_id: str = "m") -> object:
    """A logged call that went to the given places."""
    return make_row(question_id=question_id).model_copy(
        update={"destination": destinations[0], "destinations": list(destinations)}
    )


class TestDataset:
    """The question file."""

    def test_loads_with_both_kinds(self) -> None:
        """Multi-topic messages (up to five agents) and look-alikes are present."""
        questions = load_multi_topic_questions()
        assert {q.kind for q in questions} == {"multi", "single"}
        assert max(len(q.expected) for q in questions) == 5
        assert len({q.question_id for q in questions}) == len(questions)

    def test_single_questions_expect_one_agent(self) -> None:
        """Look-alikes expect exactly one agent."""
        assert all(len(q.expected) == 1 for q in load_multi_topic_questions() if q.kind == "single")

    def test_wrapped_for_router_runner(self) -> None:
        """The runner sees the first agent as expected, source 'multi'."""
        wrapped = as_routing_question(question(["tax", "planning"]))
        assert wrapped.expected == "tax" and wrapped.source == "multi"


class TestScoring:
    """Set match, Jaccard, order and splits."""

    def test_exact_in_order(self) -> None:
        """Same agents in the same order."""
        score = score_question(question(["tax", "planning"]), routed("tax", "planning"))  # type: ignore[arg-type]
        assert score.exact and score.order_ok and score.jaccard == 1.0 and score.split

    def test_same_set_other_order(self) -> None:
        """Order does not affect the set match."""
        score = score_question(question(["tax", "planning"]), routed("planning", "tax"))  # type: ignore[arg-type]
        assert score.exact and not score.order_ok

    def test_partial(self) -> None:
        """One of two agents found: Jaccard 0.5, not exact."""
        score = score_question(question(["tax", "planning"]), routed("tax"))  # type: ignore[arg-type]
        assert not score.exact and score.jaccard == pytest.approx(0.5)

    def test_error_scores_zero(self) -> None:
        """A failed call predicts nothing."""
        row = make_row(error="boom")
        score = score_question(question(["tax"]), row)
        assert score.predicted == [] and score.jaccard == 0.0

    def test_summary(self) -> None:
        """Look-alikes that split are counted."""
        scores = [
            score_question(question(["tax", "planning"]), routed("tax", "planning")),  # type: ignore[arg-type]
            score_question(question(["tax"], "single"), routed("tax", "expense")),  # type: ignore[arg-type]
            score_question(question(["tax"], "single"), routed("tax")),  # type: ignore[arg-type]
        ]
        summary = summarize(scores)
        assert (summary.multi_exact, summary.multi_questions) == (1, 1)
        assert (summary.single_correct, summary.single_wrong_splits) == (1, 1)


class TestReport:
    """The report is built from the log."""

    def test_writes_report(self, tmp_path: Path) -> None:
        """report.md and summary.json are written from calls.jsonl."""
        row = routed("tax", "planning", question_id="m2_x")
        (tmp_path / LOG_NAME).write_text(row.model_dump_json() + "\n", encoding="utf-8")  # type: ignore[attr-defined]
        report = write_report(tmp_path, [question(["tax", "planning"], question_id="m2_x")])
        assert "exact agent set 1/1" in report.read_text(encoding="utf-8")
        assert (tmp_path / "summary.json").exists()
