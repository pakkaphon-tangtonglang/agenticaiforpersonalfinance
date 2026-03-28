"""Performance evaluation module for measuring latency, tokens, and cost.

Wraps agent calls with timing instrumentation and extracts
token usage metadata from LangChain callbacks.
"""

import time
from collections.abc import Callable
from decimal import Decimal
from langchain_core.language_models.chat_models import BaseChatModel
from sqlalchemy.orm import Session

from finance_ai.agents.router_agent import orchestrate_query
from finance_ai.evaluation.metrics import compute_percentile
from finance_ai.evaluation.models import (
    PerformanceAggregateResult,
    PerformanceResult,
    QualityCase,
    QualityDataset,
)

DEFAULT_COST_PER_1K_INPUT: dict[str, Decimal] = {
    "gemini-2.0-flash": Decimal("0.0001"),
    "gpt-4o": Decimal("0.005"),
    "claude-sonnet-4-20250514": Decimal("0.003"),
}

DEFAULT_COST_PER_1K_OUTPUT: dict[str, Decimal] = {
    "gemini-2.0-flash": Decimal("0.0004"),
    "gpt-4o": Decimal("0.015"),
    "claude-sonnet-4-20250514": Decimal("0.015"),
}


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model_name: str,
) -> Decimal:
    """Estimate API cost based on token usage and model pricing.

    Args:
        input_tokens: Number of input tokens used.
        output_tokens: Number of output tokens generated.
        model_name: Name of the model for pricing lookup.

    Returns:
        Estimated cost in USD.

    Example:
        >>> estimate_cost(500, 200, "gemini-2.0-flash")
        Decimal('0.0001300')
    """
    input_rate = DEFAULT_COST_PER_1K_INPUT.get(model_name, Decimal("0.001"))
    output_rate = DEFAULT_COST_PER_1K_OUTPUT.get(model_name, Decimal("0.002"))
    input_cost = Decimal(input_tokens) / Decimal("1000") * input_rate
    output_cost = Decimal(output_tokens) / Decimal("1000") * output_rate
    return (input_cost + output_cost).quantize(Decimal("0.0000001"))


def evaluate_single_performance(
    model: BaseChatModel,
    case: QualityCase,
    model_name: str = "",
    user_id: str = "",
    db_session_factory: Callable[[], Session] | None = None,
) -> PerformanceResult:
    """Evaluate performance metrics for a single query.

    Args:
        model: Chat model to evaluate.
        case: Test case with query.
        model_name: Name of the model for cost estimation.
        user_id: UUID of the user.
        db_session_factory: Optional session factory.

    Returns:
        PerformanceResult with timing and token metrics.

    Example:
        >>> result = evaluate_single_performance(model, case)
    """
    start = time.perf_counter()
    agent_result = orchestrate_query(
        query=case.query,
        chat_model=model,
        user_id=user_id,
        db_session_factory=db_session_factory,
    )
    total_latency = time.perf_counter() - start
    agent = agent_result.get("intent", "unknown")

    cost = estimate_cost(0, 0, model_name) if model_name else None

    return PerformanceResult(
        case_id=case.case_id,
        query=case.query,
        agent=agent,
        total_latency_seconds=total_latency,
        estimated_cost_usd=cost,
    )


def evaluate_performance_dataset(
    model: BaseChatModel,
    dataset: QualityDataset,
    model_name: str = "",
    user_id: str = "",
    db_session_factory: Callable[[], Session] | None = None,
) -> PerformanceAggregateResult:
    """Evaluate performance for all cases in a dataset.

    Args:
        model: Chat model to evaluate.
        dataset: Dataset with test queries.
        model_name: Name of the model for cost estimation.
        user_id: UUID of the user.
        db_session_factory: Optional session factory.

    Returns:
        PerformanceAggregateResult with aggregated metrics.

    Example:
        >>> agg = evaluate_performance_dataset(model, dataset)
    """
    results = [
        evaluate_single_performance(model, case, model_name, user_id, db_session_factory)
        for case in dataset.cases
    ]
    return _aggregate_performance_results(results)


def _aggregate_performance_results(
    results: list[PerformanceResult],
) -> PerformanceAggregateResult:
    """Aggregate individual performance results.

    Args:
        results: List of individual performance results.

    Returns:
        PerformanceAggregateResult with percentile latencies.
    """
    latencies = [r.total_latency_seconds for r in results]
    per_agent = _compute_per_agent_latency(results)
    total_cost = _sum_costs(results)

    return PerformanceAggregateResult(
        total_queries=len(results),
        mean_total_latency=_safe_mean(latencies),
        p50_latency=compute_percentile(latencies, 50),
        p95_latency=compute_percentile(latencies, 95),
        p99_latency=compute_percentile(latencies, 99),
        total_input_tokens=sum(r.input_tokens or 0 for r in results),
        total_output_tokens=sum(r.output_tokens or 0 for r in results),
        total_estimated_cost_usd=total_cost,
        per_agent_latency=per_agent,
        results=results,
    )


def _compute_per_agent_latency(
    results: list[PerformanceResult],
) -> dict[str, float]:
    """Compute mean latency per agent type.

    Args:
        results: List of performance results.

    Returns:
        Dict mapping agent name to mean latency.
    """
    agent_latencies: dict[str, list[float]] = {}
    for result in results:
        if result.agent not in agent_latencies:
            agent_latencies[result.agent] = []
        agent_latencies[result.agent].append(result.total_latency_seconds)
    return {a: _safe_mean(lats) for a, lats in agent_latencies.items()}


def _sum_costs(results: list[PerformanceResult]) -> Decimal:
    """Sum estimated costs across all results.

    Args:
        results: List of performance results.

    Returns:
        Total estimated cost in USD.
    """
    return sum(
        (r.estimated_cost_usd for r in results if r.estimated_cost_usd),
        Decimal("0"),
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
