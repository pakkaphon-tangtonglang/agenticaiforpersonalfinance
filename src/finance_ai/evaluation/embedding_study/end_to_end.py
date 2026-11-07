"""End-to-end check for the top retrievers: answer with Gemini, judge with DeepSeek.

For each question and model, the top 3 chunks (production top_k) are numbered
like the production search tool and gemini-3.5-flash answers under the
production anti-hallucination and citation rules. DeepSeek V4.1 Flash, a
different model family (no self-preference), judges each answer against the
gold passage: correct (states the reference fact) and faithful (every claim is
supported by the passages it was given).

Usage:
    uv run python -m finance_ai.evaluation.embedding_study.end_to_end --models gemini bge_m3 bm25
"""

import argparse
import time
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from finance_ai.agents.prompts import ANTI_HALLUCINATION_RULES
from finance_ai.core.logging import get_logger
from finance_ai.evaluation.architectures.runner import StudyPaused, is_provider_unavailable
from finance_ai.evaluation.architectures.stats import exact_binomial_p, wilson_interval
from finance_ai.evaluation.embedding_study.corpus import (
    Chunk,
    RagQuestion,
    load_chunks,
    load_questions,
    relevant_chunk_ids,
)
from finance_ai.evaluation.embedding_study.study import RESULTS_DIR, QuestionResult

logger = get_logger(__name__)

ANSWER_MODEL = ("google", "gemini-3.5-flash")
JUDGE_MODEL = ("ollama", "deepseek-v4.1-flash")
LOG_NAME = "end_to_end.jsonl"
PASSAGES = 3
RETRY_DELAYS_SECONDS = (5.0, 15.0, 30.0)

ANSWER_SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยการเงินส่วนบุคคลภาษาไทย ตอบคำถามโดยใช้ผลค้นหาจาก search_finance_knowledge"
    " ที่ให้ไว้ด้านล่าง ตอบสั้นและตรงคำถาม" + ANTI_HALLUCINATION_RULES + "การอ้างอิงแหล่งที่มา:\n"
    "- ผลค้นหาแต่ละช่วงขึ้นต้นด้วยเลข เช่น [1] หรือ [2]\n"
    "- เมื่อใช้ข้อมูลจากช่วงใด ให้ใส่เลขนั้นท้ายประโยค\n"
)
JUDGE_PROMPT = (
    "You grade an answer from a Thai personal-finance assistant. Reply with JSON only:\n"
    '{{"correct": true|false, "faithful": true|false, "reason": "<one short sentence>"}}\n'
    "correct = the answer states the fact that answers the question, as given in the REFERENCE "
    "(numbers must match; extra correct detail is fine; saying the information is unavailable "
    "is not correct).\n"
    "faithful = every factual claim in the answer is supported by the PASSAGES it was given "
    "(not the reference). An answer that only says it cannot find the information is faithful.\n\n"
    "QUESTION: {question}\n\nREFERENCE (gold passage):\n{reference}\n\n"
    "PASSAGES given to the assistant:\n{passages}\n\nANSWER:\n{answer}\n"
)


class EndToEndRow(BaseModel):
    """One answered and judged question for one retriever."""

    model: str
    question_id: str
    passage_ids: list[str]
    answer: str = ""
    correct: Optional[bool] = None
    faithful: Optional[bool] = None
    reason: str = ""
    error: Optional[str] = None


class Verdict(BaseModel):
    """The judge's JSON reply."""

    correct: bool
    faithful: bool
    reason: str = ""


def build_model(provider: str, model_name: str) -> Any:
    """Create a chat model without changing the shared settings object.

    Args:
        provider: "google" or "ollama".
        model_name: Model id.

    Returns:
        LangChain chat model.
    """
    # pylint: disable=import-outside-toplevel
    from finance_ai.agents.llm_factory import create_chat_model
    from finance_ai.core.config import get_settings

    field = {"google": "google_model", "ollama": "ollama_model"}[provider]
    settings = get_settings().model_copy(update={"llm_provider": provider, field: model_name})
    return create_chat_model(settings=settings)


def numbered_passages(chunks: list[Chunk]) -> str:
    """Number passages like the production search tool.

    Example:
        >>> numbered_passages([Chunk(chunk_id="c", source_file="a.md", text="x")])
        '[1] a.md\\nx'
    """
    return "\n\n".join(f"[{i}] {c.source_file}\n{c.text}" for i, c in enumerate(chunks, start=1))


def text_of(reply: Any) -> str:
    """Plain text of a chat reply (Gemini may return content blocks)."""
    content = getattr(reply, "content", reply)
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content)


def parse_verdict(raw: str) -> Verdict:
    """Parse the judge reply, tolerating code fences and surrounding text.

    Raises:
        ValueError: When no JSON object with the two fields is found.

    Example:
        >>> parse_verdict('```json\\n{"correct": true, "faithful": false}\\n```').faithful
        False
    """
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"No JSON in judge reply: {raw[:120]}")
    try:
        return Verdict.model_validate_json(raw[start : end + 1])
    except ValidationError as error:
        raise ValueError(f"Bad judge reply: {raw[:120]}") from error


def answer_and_judge(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    answer_model: Any,
    judge_model: Any,
    model: str,
    question: RagQuestion,
    passages: list[Chunk],
    reference: str,
) -> EndToEndRow:
    """Answer one question from its passages, then judge the answer.

    Args:
        answer_model: Gemini chat model.
        judge_model: DeepSeek chat model.
        model: Retriever name (for the log).
        question: The question.
        passages: The retriever's top chunks.
        reference: Gold passage text.

    Returns:
        The judged row (provider errors propagate).
    """
    context = numbered_passages(passages)
    reply = answer_model.invoke(
        [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=f"ผลค้นหา:\n{context}\n\nคำถาม: {question.query}"),
        ]
    )
    answer = text_of(reply)
    prompt = JUDGE_PROMPT.format(
        question=question.query, reference=reference, passages=context, answer=answer
    )
    verdict = parse_verdict(text_of(judge_model.invoke([HumanMessage(content=prompt)])))
    return EndToEndRow(
        model=model,
        question_id=question.question_id,
        passage_ids=[c.chunk_id for c in passages],
        answer=answer,
        correct=verdict.correct,
        faithful=verdict.faithful,
        reason=verdict.reason,
    )


def with_retry(call: Any, key: tuple[str, str], sleep: Any = time.sleep) -> EndToEndRow:
    """Run a call, retrying provider failures; other errors become an error row.

    Args:
        call: Zero-argument function returning an EndToEndRow.
        key: (model, question_id) for the error row.
        sleep: Sleep function (replaced in tests).

    Returns:
        The row.

    Raises:
        StudyPaused: When a provider keeps failing.
    """
    for delay in (*RETRY_DELAYS_SECONDS, None):
        try:
            return call()  # type: ignore[no-any-return]
        except Exception as error:  # noqa: BLE001 - classified below
            if not is_provider_unavailable(error):
                return EndToEndRow(
                    model=key[0], question_id=key[1], passage_ids=[], error=str(error)
                )
            if delay is None:
                raise StudyPaused(f"Provider unavailable at {key}: {error}") from error
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def read_rows(path: Path) -> list[EndToEndRow]:
    """Read logged rows (missing file → empty)."""
    if not path.exists():
        return []
    return [EndToEndRow.model_validate_json(line) for line in path.read_text("utf-8").splitlines()]


def load_rankings(models: list[str]) -> dict[tuple[str, str], list[str]]:
    """Read each model's ranked chunk ids from the retrieval log.

    Args:
        models: Retriever names.

    Returns:
        (model, question_id) → ranked ids.
    """
    rankings: dict[tuple[str, str], list[str]] = {}
    for line in (RESULTS_DIR / "retrieval.jsonl").read_text("utf-8").splitlines():
        result = QuestionResult.model_validate_json(line)
        if result.model in models:
            rankings[(result.model, result.question_id)] = result.ranked
    return rankings


def run(models: list[str], answer_model: Any, judge_model: Any, limit: int = 0) -> Path:
    """Answer and judge every question for every model, resuming from the log.

    Args:
        models: Retrievers to check.
        answer_model: Gemini chat model.
        judge_model: DeepSeek chat model.
        limit: Only the first N questions (0 = all), for the judge check.

    Returns:
        Path of the log.
    """
    chunks = {c.chunk_id: c for c in load_chunks()}
    questions = load_questions()[: limit or None]
    rankings = load_rankings(models)
    log = RESULTS_DIR / LOG_NAME
    done = {(r.model, r.question_id) for r in read_rows(log) if r.error is None}
    for question in questions:
        reference = "\n---\n".join(
            chunks[i].text for i in sorted(relevant_chunk_ids(question, list(chunks.values())))
        )[:3000]
        for model in models:
            key = (model, question.question_id)
            if key in done:
                continue
            passages = [chunks[i] for i in rankings[key][:PASSAGES]]
            row = with_retry(
                lambda: answer_and_judge(
                    answer_model, judge_model, model, question, passages, reference
                ),
                key,
            )
            with log.open("a", encoding="utf-8") as handle:
                handle.write(row.model_dump_json() + "\n")
    return log


def summarize_rows(rows: list[EndToEndRow], baseline: str = "gemini") -> str:
    """Markdown summary: correct and faithful rates, paired against the baseline.

    Args:
        rows: Judged rows (latest row per model and question is used).
        baseline: Production retriever.

    Returns:
        Markdown text.
    """
    latest = {(r.model, r.question_id): r for r in rows if r.error is None}
    models = sorted({m for m, _ in latest})
    lines = [
        "# Embedding study: end-to-end answers",
        "",
        f"Answers: {ANSWER_MODEL[1]} from each retriever's top {PASSAGES} chunks, production "
        f"rules. Judge: {JUDGE_MODEL[1]} (different model family), checked against hand labels "
        "before use. 95% Wilson CIs.",
        "",
        "| Retriever | Answers | Correct (95% CI) | Faithful (95% CI) | Correct: only this / only "
        f"{baseline} | McNemar p |",
        "|---|---:|---|---|---|---:|",
    ]
    for model in models:
        own = {q: r for (m, q), r in latest.items() if m == model}
        lines.append(_summary_line(model, own, latest, baseline))
    return "\n".join(lines) + "\n"


def _summary_line(
    model: str,
    own: dict[str, EndToEndRow],
    latest: dict[tuple[str, str], EndToEndRow],
    baseline: str,
) -> str:
    """One table row of the end-to-end summary."""
    total = len(own)
    correct = sum(bool(r.correct) for r in own.values())
    faithful = sum(bool(r.faithful) for r in own.values())
    c_low, c_high = wilson_interval(correct, total)
    f_low, f_high = wilson_interval(faithful, total)
    pairs = [(r, latest[(baseline, q)]) for q, r in own.items() if (baseline, q) in latest]
    only_model = sum(bool(a.correct) and not b.correct for a, b in pairs)
    only_base = sum(bool(b.correct) and not a.correct for a, b in pairs)
    return (
        f"| {model} | {total} | {correct / total:.1%} ({c_low:.1%}–{c_high:.1%}) "
        f"| {faithful / total:.1%} ({f_low:.1%}–{f_high:.1%}) | {only_model} / {only_base} "
        f"| {exact_binomial_p(only_model, only_base):.4f} |"
    )


def main(argv: Optional[list[str]] = None) -> Path:
    """Run the end-to-end check and write end_to_end.md.

    Args:
        argv: Arguments (defaults to sys.argv).

    Returns:
        Path of end_to_end.md.
    """
    parser = argparse.ArgumentParser(description="End-to-end answers for the top retrievers")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="first N questions only")
    args = parser.parse_args(argv)
    log = RESULTS_DIR / LOG_NAME
    if not args.report_only:
        try:
            run(args.models, build_model(*ANSWER_MODEL), build_model(*JUDGE_MODEL), args.limit)
        except StudyPaused as pause:
            print(f"Paused: {pause}\nRun the same command again later to resume.")
    report = RESULTS_DIR / "end_to_end.md"
    report.write_text(summarize_rows(read_rows(log)), encoding="utf-8")
    print(f"Report: {report}")
    return report


if __name__ == "__main__":
    main()
