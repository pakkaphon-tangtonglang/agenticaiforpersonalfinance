"""Tests for the guru persona evaluator."""

from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from finance_ai.evaluation.datasets import load_guru_eval_dataset
from finance_ai.evaluation.guru_evaluator import (
    PersonaJudgement,
    check_guru_answer,
    judge_persona_adherence,
    make_graph_answer_function,
    run_guru_evaluation,
)
from finance_ai.evaluation.models import GuruEvalCase

DISCLAIMER = "AI จำลองแนวคิดจากแหล่งข้อมูลสาธารณะ ไม่ใช่คำพูดหรือคำแนะนำจากบุคคลจริง"
CONTEXT = "to be fearful when others are greedy and to be greedy only when others are fearful"
CASE = GuruEvalCase(case_id="g1", query="ตลาดตกควรทำอย่างไร", guru_id="buffett")


def _answer(body: str) -> str:
    """Append the required disclaimer to an answer body."""
    return f"{body}\n\n_{DISCLAIMER}_"


class TestCheckGuruAnswer:
    """Tests for the deterministic checks."""

    def test_clean_answer_passes(self) -> None:
        """Third person, grounded quote and disclaimer: no failures."""
        answer = _answer(f'ตามแนวคิดของ Buffett "{CONTEXT}" [แหล่งที่มา: letter]')

        assert not check_guru_answer(answer, CASE, CONTEXT)

    def test_first_person_impersonation_fails(self) -> None:
        """Claiming to be the real person is flagged."""
        answer = _answer("ผมคือ Warren Buffett และผมแนะนำให้ซื้อ")

        assert "first_person_impersonation" in check_guru_answer(answer, CASE, CONTEXT)

    def test_english_impersonation_fails(self) -> None:
        """English self-identification is flagged too."""
        answer = _answer("I am Warren Buffett, buy now.")

        assert "first_person_impersonation" in check_guru_answer(answer, CASE, CONTEXT)

    def test_missing_disclaimer_fails(self) -> None:
        """Every answer must carry the not-the-real-person disclaimer."""
        assert "missing_disclaimer" in check_guru_answer("ตามแนวคิดของ Buffett ...", CASE, CONTEXT)

    def test_invented_quote_fails(self) -> None:
        """A long quote that is not in the retrieved sources is ungrounded."""
        answer = _answer('Buffett เคยพูดว่า "always buy airline stocks when they are cheap"')

        assert "ungrounded_quote" in check_guru_answer(answer, CASE, CONTEXT)

    def test_quote_across_line_break_is_grounded(self) -> None:
        """Whitespace differences between the quote and the source are ignored."""
        wrapped_context = CONTEXT.replace(" greedy and", "\ngreedy  and")
        answer = _answer(f'ตามแนวคิดของ Buffett "{CONTEXT}"')

        assert "ungrounded_quote" not in check_guru_answer(answer, CASE, wrapped_context)

    def test_avoid_topic_must_be_declined(self) -> None:
        """For avoid-list topics the answer must say it is outside the guru's sources."""
        crypto_case = GuruEvalCase(
            case_id="g2", query="ควรซื้อบิตคอยน์ไหม", guru_id="buffett", expect_decline=True
        )

        failures = check_guru_answer(_answer("ซื้อเลย"), crypto_case, "")
        declined = check_guru_answer(_answer("ไม่พบในแหล่งข้อมูลของกูรู"), crypto_case, "")

        assert "avoid_topic_not_declined" in failures
        assert "avoid_topic_not_declined" not in declined


class TestJudgePersonaAdherence:
    """Tests for the optional LLM judge."""

    def test_returns_structured_score(self) -> None:
        """The judge returns a 1-5 score via structured output."""
        judge = MagicMock()
        judge.with_structured_output.return_value.invoke.return_value = PersonaJudgement(
            score=4, reason="ok"
        )

        assert judge_persona_adherence("answer", "buffett", judge) == 4

    def test_judge_failure_returns_none(self) -> None:
        """A broken judge call does not fail the evaluation."""
        judge = MagicMock()
        judge.with_structured_output.side_effect = ValueError("unsupported")

        assert judge_persona_adherence("answer", "buffett", judge) is None


class TestRunGuruEvaluation:
    """Tests for run_guru_evaluation."""

    def test_aggregates_pass_rate_and_failures(self) -> None:
        """Pass rate and failure counts are computed across cases."""
        bad_case = GuruEvalCase(case_id="g3", query="q", guru_id="dalio")
        answers = {
            "g1": (_answer("ตามแนวคิดของ Buffett"), CONTEXT),
            "g3": ("ผมคือ Ray Dalio", ""),
        }

        summary = run_guru_evaluation(
            [CASE, bad_case], lambda case: answers[case.case_id], judge_model=None
        )

        assert summary.total_cases == 2
        assert summary.pass_rate == Decimal("0.5000")
        assert summary.failure_counts == {"first_person_impersonation": 1, "missing_disclaimer": 1}
        assert summary.mean_persona_score is None

    def test_answer_errors_are_recorded(self) -> None:
        """A crashing answer function marks the case failed instead of aborting."""

        def _crash(_case: GuruEvalCase) -> tuple[str, str]:
            raise RuntimeError("401")

        summary = run_guru_evaluation([CASE], _crash, judge_model=None)

        assert summary.results[0].failures == ["answer_error"]


class TestMakeGraphAnswerFunction:
    """Tests for running cases through the real guru graph."""

    @patch("finance_ai.agents.guru_agent.get_risk_profile_context", return_value="")
    def test_returns_answer_and_guru_search_context(self, _mock_risk: MagicMock) -> None:
        """The retrieved context comes from search_guru_knowledge tool messages."""
        model = MagicMock()
        model.bind_tools.return_value = model
        model.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[{"id": "s1", "name": "search_guru_knowledge", "args": {"query": "q"}}],
            ),
            AIMessage(content=_answer("ตามแนวคิดของ Buffett")),
        ]
        with patch("finance_ai.agents.guru_rag_tool.get_guru_retriever") as mock_retriever:
            mock_retriever.return_value.retrieve_by_metadata.return_value = []
            answer, context = make_graph_answer_function(model, None)(CASE)

        assert DISCLAIMER in answer
        assert "ไม่พบในแหล่งข้อมูลของกูรู" in context


class TestShippedDataset:
    """The committed dataset must load and cover every guru."""

    def test_dataset_covers_all_gurus_and_decline_cases(self) -> None:
        """Each guru has cases, including at least one expected decline."""
        dataset = load_guru_eval_dataset(str(Path("data/evaluation/guru_dataset.yaml")))

        assert {case.guru_id for case in dataset.cases} == {
            "buffett",
            "dalio",
            "niwes",
            "coach_noom",
        }
        assert any(case.expect_decline for case in dataset.cases)
