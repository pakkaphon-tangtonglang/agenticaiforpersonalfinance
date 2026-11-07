"""Run the retrieval comparison and write its report.

Usage:
    uv run python -m finance_ai.evaluation.embedding_study.study
    uv run python -m finance_ai.evaluation.embedding_study.study --models bm25 minilm

Results: ``data/evaluation/results/embedding_study/`` (``retrieval.jsonl``,
``report.md``, ``summary.json``; dense vectors cached in ``vectors/``).
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from finance_ai.evaluation.architectures.stats import exact_binomial_p, percentile
from finance_ai.evaluation.embedding_study.corpus import (
    Chunk,
    RagQuestion,
    load_chunks,
    load_questions,
    relevant_chunk_ids,
)
from finance_ai.evaluation.embedding_study.metrics import (
    lexical_overlap,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from finance_ai.evaluation.embedding_study.retrievers import (
    MODEL_SPECS,
    Retriever,
    build_retriever,
)
from finance_ai.evaluation.router_thinking.report import bootstrap_mean_interval

RESULTS_DIR = Path("data/evaluation/results/embedding_study")
TOP_K = 10
PRODUCTION_K = 3
BASELINE = "gemini"
LEAKAGE_THRESHOLD = 0.5


class QuestionResult(BaseModel):
    """One model's ranking for one question."""

    model: str
    question_id: str
    group: str
    ranked: list[str]
    relevant_count: int
    recall_1: float
    recall_3: float
    recall_5: float
    mrr: float
    ndcg_10: float
    latency_ms: float


def question_group(question: RagQuestion) -> str:
    """Group a question by its source: Thai guide, Thai PDF, or English PDF.

    Example:
        >>> question_group(RagQuestion(question_id="q", query="What?", gold_files=["x.pdf"],
        ...     keywords=["k"]))
        'english_pdf'
    """
    if question.query.isascii():
        return "english_pdf"
    if all(name.endswith(".pdf") for name in question.gold_files):
        return "thai_pdf"
    return "thai_guide"


def score_question(
    retriever: Retriever, question: RagQuestion, relevant: set[str]
) -> QuestionResult:
    """Search once and score the ranking.

    Args:
        retriever: An indexed retriever.
        question: The question.
        relevant: Ids of chunks that answer it.

    Returns:
        The scored result.
    """
    start = time.perf_counter()
    ranked = retriever.search(question.query, TOP_K)
    latency_ms = (time.perf_counter() - start) * 1000
    return QuestionResult(
        model=retriever.name,
        question_id=question.question_id,
        group=question_group(question),
        ranked=ranked,
        relevant_count=len(relevant),
        recall_1=recall_at_k(ranked, relevant, 1),
        recall_3=recall_at_k(ranked, relevant, PRODUCTION_K),
        recall_5=recall_at_k(ranked, relevant, 5),
        mrr=reciprocal_rank(ranked, relevant, TOP_K),
        ndcg_10=ndcg_at_k(ranked, relevant, TOP_K),
        latency_ms=latency_ms,
    )


def run_model(
    model: str, chunks: list[Chunk], questions: list[RagQuestion], relevant: dict[str, set[str]]
) -> tuple[list[QuestionResult], float]:
    """Index the chunks with one model and score every question.

    Args:
        model: Key in MODEL_SPECS.
        chunks: All chunks.
        questions: All questions.
        relevant: Question id → relevant chunk ids.

    Returns:
        (results, index build seconds).
    """
    retriever = build_retriever(MODEL_SPECS[model], RESULTS_DIR / "vectors")
    retriever.index(chunks)
    results = [score_question(retriever, q, relevant[q.question_id]) for q in questions]
    return results, float(getattr(retriever, "index_seconds", 0.0))


class ModelSummary(BaseModel):
    """Mean metrics for one model, with 95 % bootstrap CIs over questions."""

    model: str
    ndcg_10: float
    ndcg_low: float
    ndcg_high: float
    recall_1: float
    recall_3: float
    recall_5: float
    mrr: float
    latency_ms_median: float
    index_seconds: float
    by_group: dict[str, float]


def summarize(model: str, results: list[QuestionResult], index_seconds: float) -> ModelSummary:
    """Average one model's per-question scores.

    Args:
        model: Model key.
        results: Its results.
        index_seconds: Index build time (0 when vectors came from cache).

    Returns:
        The summary.
    """
    ndcg = [r.ndcg_10 for r in results]
    low, high = bootstrap_mean_interval(ndcg)
    groups = sorted({r.group for r in results})
    return ModelSummary(
        model=model,
        ndcg_10=statistics.fmean(ndcg),
        ndcg_low=low,
        ndcg_high=high,
        recall_1=statistics.fmean(r.recall_1 for r in results),
        recall_3=statistics.fmean(r.recall_3 for r in results),
        recall_5=statistics.fmean(r.recall_5 for r in results),
        mrr=statistics.fmean(r.mrr for r in results),
        latency_ms_median=percentile([r.latency_ms for r in results], 0.5),
        index_seconds=index_seconds,
        by_group={g: statistics.fmean(r.ndcg_10 for r in results if r.group == g) for g in groups},
    )


class PairedComparison(BaseModel):
    """A model against the production model on the same questions."""

    model: str
    ndcg_diff: float
    ndcg_diff_low: float
    ndcg_diff_high: float
    better: int
    worse: int
    sign_p: float
    hit3_only_model: int
    hit3_only_baseline: int
    mcnemar_p: float


def compare_to_baseline(
    model: str, results: list[QuestionResult], baseline: list[QuestionResult]
) -> PairedComparison:
    """Paired nDCG and Recall@3 comparison against the baseline model.

    Args:
        model: Model key.
        results: Its results.
        baseline: Baseline results (same questions).

    Returns:
        The comparison.
    """
    base = {r.question_id: r for r in baseline}
    pairs = [(r, base[r.question_id]) for r in results if r.question_id in base]
    diffs = [a.ndcg_10 - b.ndcg_10 for a, b in pairs]
    low, high = bootstrap_mean_interval(diffs)
    better, worse = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    only_model = sum(a.recall_3 > b.recall_3 for a, b in pairs)
    only_base = sum(b.recall_3 > a.recall_3 for a, b in pairs)
    return PairedComparison(
        model=model,
        ndcg_diff=statistics.fmean(diffs) if diffs else 0.0,
        ndcg_diff_low=low,
        ndcg_diff_high=high,
        better=better,
        worse=worse,
        sign_p=exact_binomial_p(better, worse),
        hit3_only_model=only_model,
        hit3_only_baseline=only_base,
        mcnemar_p=exact_binomial_p(only_model, only_base),
    )


def leakage_scores(
    questions: list[RagQuestion], chunks: list[Chunk], relevant: dict[str, set[str]]
) -> dict[str, float]:
    """Highest lexical overlap between each question and any of its relevant chunks.

    Args:
        questions: All questions.
        chunks: All chunks.
        relevant: Question id → relevant chunk ids.

    Returns:
        Question id → overlap in [0, 1].
    """
    text = {chunk.chunk_id: chunk.text for chunk in chunks}
    return {
        q.question_id: max(
            (lexical_overlap(q.query, text[c]) for c in relevant[q.question_id]), default=0.0
        )
        for q in questions
    }


def _fmt(value: float) -> str:
    """Three decimals."""
    return f"{value:.3f}"


def build_markdown(
    summaries: list[ModelSummary], comparisons: list[PairedComparison], leakage: dict[str, float]
) -> str:
    """Assemble report.md.

    Args:
        summaries: Per-model summaries (best nDCG first).
        comparisons: Paired comparisons against the production model.
        leakage: Question id → lexical overlap.

    Returns:
        Markdown text.
    """
    groups = sorted({g for s in summaries for g in s.by_group})
    leaky = sorted(q for q, v in leakage.items() if v > LEAKAGE_THRESHOLD)
    lines = [
        "# Embedding study: retrieval quality",
        "",
        f"{len(leakage)} questions, production chunks (500/50), exact cosine search, top {TOP_K}. "
        "A chunk is relevant when it comes from a gold file and contains every answer keyword. "
        f"Recall@{PRODUCTION_K} is what production sends to the agent (top_k = 3). "
        "95% CIs: bootstrap over questions. Latency = one query (local CPU for open models, "
        "API round trip for Gemini), so it is not comparable across the two.",
        "",
        "## Models",
        "| Model | nDCG@10 (95% CI) | Recall@1 | Recall@3 | Recall@5 | MRR@10 "
        "| Query latency median (ms) | Index build (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.model} | {_fmt(s.ndcg_10)} ({_fmt(s.ndcg_low)}–{_fmt(s.ndcg_high)}) "
            f"| {_fmt(s.recall_1)} | {_fmt(s.recall_3)} | {_fmt(s.recall_5)} | {_fmt(s.mrr)} "
            f"| {s.latency_ms_median:.0f} | {s.index_seconds:.1f} |"
        )
    lines += [
        "",
        "## nDCG@10 by question source",
        "| Model | " + " | ".join(groups) + " |",
        "|---|" + "---:|" * len(groups),
    ]
    for s in summaries:
        lines.append(
            f"| {s.model} | " + " | ".join(_fmt(s.by_group.get(g, 0)) for g in groups) + " |"
        )
    lines += [
        "",
        f"## Paired against {BASELINE} (production)",
        "| Model | nDCG@10 diff (95% CI) | Better / worse questions | Sign p "
        "| Recall@3 only model / only gemini | McNemar p |",
        "|---|---|---|---:|---|---:|",
    ]
    for c in comparisons:
        lines.append(
            f"| {c.model} | {c.ndcg_diff:+.3f} ({c.ndcg_diff_low:+.3f} to {c.ndcg_diff_high:+.3f}) "
            f"| {c.better} / {c.worse} | {c.sign_p:.4f} | {c.hit3_only_model} / "
            f"{c.hit3_only_baseline} | {c.mcnemar_p:.4f} |"
        )
    lines += [
        "",
        "## Leakage check",
        f"Share of each question's 6-character sequences that also appear in its relevant chunk. "
        f"Median {percentile(list(leakage.values()), 0.5):.2f}; "
        f"{len(leaky)} questions above {LEAKAGE_THRESHOLD}: {', '.join(leaky) or 'none'}.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(
    results: dict[str, list[QuestionResult]],
    index_seconds: dict[str, float],
    leakage: dict[str, float],
) -> Path:
    """Write retrieval.jsonl, summary.json and report.md.

    Args:
        results: Model → per-question results.
        index_seconds: Model → index build time.
        leakage: Question id → lexical overlap.

    Returns:
        Path of report.md.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [r.model_dump_json() for model_results in results.values() for r in model_results]
    (RESULTS_DIR / "retrieval.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    summaries = sorted(
        (summarize(m, r, index_seconds[m]) for m, r in results.items()), key=lambda s: -s.ndcg_10
    )
    comparisons = [
        compare_to_baseline(m, r, results[BASELINE])
        for m, r in results.items()
        if m != BASELINE and BASELINE in results
    ]
    summary = {
        "models": [s.model_dump() for s in summaries],
        "comparisons": [c.model_dump() for c in comparisons],
        "leakage": leakage,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = RESULTS_DIR / "report.md"
    report.write_text(build_markdown(summaries, comparisons, leakage), encoding="utf-8")
    return report


def main(argv: Optional[list[str]] = None) -> Path:
    """Run every chosen model and write the report.

    Args:
        argv: Arguments (defaults to sys.argv).

    Returns:
        Path of report.md.
    """
    parser = argparse.ArgumentParser(description="Embedding retrieval comparison")
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS), choices=list(MODEL_SPECS))
    args = parser.parse_args(argv)
    chunks, questions = load_chunks(), load_questions()
    relevant = {q.question_id: relevant_chunk_ids(q, chunks) for q in questions}
    results: dict[str, list[QuestionResult]] = {}
    index_seconds: dict[str, float] = {}
    for model in args.models:
        print(f"Running {model} …", flush=True)
        results[model], index_seconds[model] = run_model(model, chunks, questions, relevant)
    report = write_outputs(results, index_seconds, leakage_scores(questions, chunks, relevant))
    print(f"Report: {report}")
    return report


if __name__ == "__main__":
    main()
