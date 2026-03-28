"""RAG retrieval evaluation module.

Evaluates retrieval quality by measuring precision, recall, MRR,
and keyword hit rate against expected source documents.
"""

import time

from finance_ai.evaluation.metrics import (
    compute_keyword_hit_rate,
    compute_mrr,
    compute_precision_at_k,
    compute_recall_at_k,
)
from finance_ai.evaluation.models import (
    RAGAggregateResult,
    RAGCase,
    RAGDataset,
    RAGRetrievalResult,
)
from finance_ai.rag.vector_store import FinanceVectorStore


def evaluate_single_rag_case(
    store: FinanceVectorStore,
    case: RAGCase,
    top_k: int = 3,
) -> RAGRetrievalResult:
    """Evaluate a single RAG retrieval case.

    Args:
        store: Vector store to search.
        case: RAG test case with expected results.
        top_k: Number of results to retrieve.

    Returns:
        RAGRetrievalResult with computed metrics.

    Example:
        >>> result = evaluate_single_rag_case(store, case, top_k=3)
    """
    start = time.perf_counter()
    results = store.search(case.query, top_k=top_k, domain_filter=case.domain_filter)
    latency = time.perf_counter() - start

    retrieved_sources = [r.source_file for r in results]
    combined_text = " ".join(r.content for r in results)

    return RAGRetrievalResult(
        case_id=case.case_id,
        query=case.query,
        retrieved_sources=retrieved_sources,
        expected_sources=case.expected_source_files,
        precision_at_k=compute_precision_at_k(retrieved_sources, case.expected_source_files, top_k),
        recall_at_k=compute_recall_at_k(retrieved_sources, case.expected_source_files, top_k),
        reciprocal_rank=compute_mrr(retrieved_sources, case.expected_source_files),
        keyword_hit_rate=compute_keyword_hit_rate(combined_text, case.expected_keywords),
        latency_seconds=latency,
    )


def evaluate_rag_dataset(
    store: FinanceVectorStore,
    dataset: RAGDataset,
    top_k: int = 3,
) -> RAGAggregateResult:
    """Evaluate all cases in a RAG dataset.

    Args:
        store: Vector store to search.
        dataset: RAG evaluation dataset.
        top_k: Number of results to retrieve per query.

    Returns:
        RAGAggregateResult with aggregated metrics.

    Example:
        >>> agg = evaluate_rag_dataset(store, dataset)
    """
    results = [evaluate_single_rag_case(store, case, top_k) for case in dataset.cases]
    per_domain = _compute_per_domain_precision(results, dataset.cases)

    return RAGAggregateResult(
        total_cases=len(results),
        mean_precision_at_k=_safe_mean([r.precision_at_k for r in results]),
        mean_recall_at_k=_safe_mean([r.recall_at_k for r in results]),
        mean_reciprocal_rank=_safe_mean([r.reciprocal_rank for r in results]),
        mean_keyword_hit_rate=_safe_mean([r.keyword_hit_rate for r in results]),
        per_domain_precision=per_domain,
        mean_latency_seconds=_safe_mean([r.latency_seconds for r in results]),
        results=results,
    )


def _safe_mean(values: list[float]) -> float:
    """Compute mean of floats, returning 0.0 for empty list.

    Args:
        values: List of float values.

    Returns:
        Mean value or 0.0 if empty.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def _compute_per_domain_precision(
    results: list[RAGRetrievalResult],
    cases: list[RAGCase],
) -> dict[str, float]:
    """Compute per-domain mean precision.

    Args:
        results: List of evaluation results.
        cases: Original test cases (for domain info).

    Returns:
        Dict mapping domain to mean precision.
    """
    domain_scores: dict[str, list[float]] = {}
    for result, case in zip(results, cases):
        domain = case.domain_filter or "general"
        if domain not in domain_scores:
            domain_scores[domain] = []
        domain_scores[domain].append(result.precision_at_k)
    return {d: _safe_mean(scores) for d, scores in domain_scores.items()}
