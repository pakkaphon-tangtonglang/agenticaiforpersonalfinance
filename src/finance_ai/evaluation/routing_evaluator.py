"""Router intent classification evaluation module.

Evaluates router accuracy by comparing predicted intents against
expected intents, producing confusion matrix and per-class metrics.
"""

import time
from decimal import Decimal

from langchain_core.language_models.chat_models import BaseChatModel

from finance_ai.agents.router_agent import classify_query
from finance_ai.evaluation.metrics import (
    compute_confusion_matrix,
    compute_per_class_accuracy,
)
from finance_ai.evaluation.models import (
    RoutingAggregateResult,
    RoutingCase,
    RoutingDataset,
    RoutingResult,
)


def evaluate_single_routing_case(
    model: BaseChatModel,
    case: RoutingCase,
) -> RoutingResult:
    """Evaluate a single routing classification case.

    Args:
        model: Chat model to use for classification.
        case: Routing test case with expected intent.

    Returns:
        RoutingResult with predicted intent and correctness.

    Example:
        >>> result = evaluate_single_routing_case(model, case)
    """
    start = time.perf_counter()
    decision = classify_query(case.query, chat_model=model)
    latency = time.perf_counter() - start

    return RoutingResult(
        case_id=case.case_id,
        query=case.query,
        expected_intent=case.expected_intent,
        predicted_intent=decision.intent,
        predicted_confidence=decision.confidence,
        is_correct=(decision.intent == case.expected_intent),
        latency_seconds=latency,
    )


def evaluate_routing_dataset(
    model: BaseChatModel,
    dataset: RoutingDataset,
) -> RoutingAggregateResult:
    """Evaluate all cases in a routing dataset.

    Args:
        model: Chat model to use for classification.
        dataset: Routing evaluation dataset.

    Returns:
        RoutingAggregateResult with aggregated metrics.

    Example:
        >>> agg = evaluate_routing_dataset(model, dataset)
    """
    results = [evaluate_single_routing_case(model, case) for case in dataset.cases]
    return _aggregate_routing_results(results)


def _aggregate_routing_results(
    results: list[RoutingResult],
) -> RoutingAggregateResult:
    """Aggregate individual routing results into summary metrics.

    Args:
        results: List of individual routing results.

    Returns:
        RoutingAggregateResult with accuracy and confusion matrix.
    """
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    accuracy = Decimal(correct) / Decimal(max(total, 1))

    pairs = [(r.expected_intent, r.predicted_intent) for r in results]
    matrix = compute_confusion_matrix(pairs)
    per_intent = compute_per_class_accuracy(matrix)
    latencies = [r.latency_seconds for r in results]

    return RoutingAggregateResult(
        total_cases=total,
        correct_count=correct,
        accuracy=accuracy.quantize(Decimal("0.0001")),
        per_intent_accuracy=per_intent,
        confusion_matrix=matrix,
        mean_latency_seconds=sum(latencies) / max(len(latencies), 1),
        results=results,
    )
