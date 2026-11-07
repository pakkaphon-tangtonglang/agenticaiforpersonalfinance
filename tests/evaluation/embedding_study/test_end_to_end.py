"""Tests for the end-to-end answer and judge step."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from finance_ai.evaluation.architectures.runner import StudyPaused
from finance_ai.evaluation.embedding_study import end_to_end as e2e
from finance_ai.evaluation.embedding_study.corpus import Chunk, RagQuestion

CHUNKS = [Chunk(chunk_id="c1", source_file="tax.md", text="ค่าลดหย่อนส่วนตัว 60,000 บาท")]
QUESTION = RagQuestion(
    question_id="q1", query="ลดหย่อนส่วนตัวเท่าไร", gold_files=["tax.md"], keywords=["60,000"]
)


class FakeModel:
    """Returns a fixed reply and records prompts."""

    def __init__(self, reply: str) -> None:
        """Store the reply."""
        self.reply = reply
        self.prompts: list[list[object]] = []

    def invoke(self, messages: list[object]) -> AIMessage:
        """Record the prompt and reply."""
        self.prompts.append(messages)
        return AIMessage(content=self.reply)


class TestHelpers:
    """Passage numbering, reply text and verdict parsing."""

    def test_numbered_passages(self) -> None:
        """Passages are numbered from 1 with their file name."""
        assert e2e.numbered_passages(CHUNKS).startswith("[1] tax.md\n")

    def test_text_of_content_blocks(self) -> None:
        """Gemini content blocks are joined."""
        reply = AIMessage(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
        assert e2e.text_of(reply) == "ab"

    def test_parse_verdict_and_errors(self) -> None:
        """JSON inside text is found; missing or invalid JSON raises."""
        verdict = e2e.parse_verdict('ok {"correct": true, "faithful": true, "reason": "r"} done')
        assert verdict.correct and verdict.reason == "r"
        with pytest.raises(ValueError):
            e2e.parse_verdict("no json")
        with pytest.raises(ValueError):
            e2e.parse_verdict('{"correct": "maybe"}')


class TestAnswerAndJudge:
    """One question through both models."""

    def test_row_and_prompts(self) -> None:
        """The answer uses production rules; the judge sees the reference and answer."""
        answer = FakeModel("ลดหย่อนได้ 60,000 บาท [1]")
        judge = FakeModel('{"correct": true, "faithful": true, "reason": "matches"}')
        row = e2e.answer_and_judge(answer, judge, "gemini", QUESTION, CHUNKS, "60,000 บาท")
        assert row.correct and row.faithful and row.passage_ids == ["c1"]
        assert "search_finance_knowledge" in str(answer.prompts[0][0].content)  # type: ignore[attr-defined]
        assert "60,000 บาท [1]" in str(judge.prompts[0][0].content)  # type: ignore[attr-defined]


class TestRetry:
    """Provider failures pause; other failures are logged."""

    def test_pause_after_retries(self) -> None:
        """A rate limit that persists raises StudyPaused."""

        def fail() -> e2e.EndToEndRow:
            raise RuntimeError("429 Too Many Requests")

        with pytest.raises(StudyPaused):
            e2e.with_retry(fail, ("gemini", "q1"), sleep=lambda _s: None)

    def test_other_error_logged(self) -> None:
        """A bad judge reply becomes an error row."""

        def fail() -> e2e.EndToEndRow:
            raise ValueError("Bad judge reply")

        row = e2e.with_retry(fail, ("gemini", "q1"), sleep=lambda _s: None)
        assert row.error == "Bad judge reply"


class TestRunAndSummary:
    """Full loop with fakes, resume, and summary table."""

    def test_run_resume_and_report(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rows are logged once; the summary pairs against gemini."""
        monkeypatch.setattr(e2e, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(e2e, "load_chunks", lambda: CHUNKS)
        monkeypatch.setattr(e2e, "load_questions", lambda: [QUESTION])
        monkeypatch.setattr(
            e2e, "load_rankings", lambda models: {(m, "q1"): ["c1"] for m in models}
        )
        answer = FakeModel("60,000 บาท [1]")
        judge = FakeModel('{"correct": true, "faithful": true}')
        log = e2e.run(["gemini", "bm25"], answer, judge)
        e2e.run(["gemini", "bm25"], answer, judge)
        rows = e2e.read_rows(log)
        assert len(rows) == 2 and len(answer.prompts) == 2
        text = e2e.summarize_rows(rows)
        assert "| bm25 | 1 | 100.0%" in text

    def test_read_rows_missing(self, tmp_path: Path) -> None:
        """No log → no rows."""
        assert not e2e.read_rows(tmp_path / "none.jsonl")

    def test_main_report_only(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--report-only rebuilds the table without calling models."""
        monkeypatch.setattr(e2e, "RESULTS_DIR", tmp_path)
        row = e2e.EndToEndRow(
            model="gemini", question_id="q1", passage_ids=[], correct=True, faithful=False
        )
        (tmp_path / e2e.LOG_NAME).write_text(row.model_dump_json() + "\n", encoding="utf-8")
        report = e2e.main(["--models", "gemini", "--report-only"])
        assert "| gemini | 1 | 100.0%" in report.read_text(encoding="utf-8")

    def test_load_rankings_reads_retrieval_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the chosen models' rankings are returned."""
        # pylint: disable=import-outside-toplevel
        from finance_ai.evaluation.embedding_study.study import QuestionResult

        monkeypatch.setattr(e2e, "RESULTS_DIR", tmp_path)
        rows = [
            QuestionResult(
                model=m,
                question_id="q1",
                group="thai_guide",
                ranked=["c1"],
                relevant_count=1,
                recall_1=1,
                recall_3=1,
                recall_5=1,
                mrr=1,
                ndcg_10=1,
                latency_ms=1,
            )
            for m in ("gemini", "minilm")
        ]
        (tmp_path / "retrieval.jsonl").write_text(
            "\n".join(r.model_dump_json() for r in rows), encoding="utf-8"
        )
        assert e2e.load_rankings(["gemini"]) == {("gemini", "q1"): ["c1"]}
