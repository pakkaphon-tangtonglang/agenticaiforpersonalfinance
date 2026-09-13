"""Evaluation runner that orchestrates all evaluation dimensions."""

import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from sqlalchemy.orm import Session

from finance_ai.agents.router_agent import orchestrate_query
from finance_ai.evaluation.accuracy_evaluator import (
    evaluate_tax_accuracy_dataset,
    evaluate_tax_accuracy_dataset_forced,
)
from finance_ai.evaluation.datasets import (
    load_hallucination_dataset,
    load_quality_dataset,
    load_rag_dataset,
    load_recommendation_safety_dataset,
    load_routing_dataset,
    load_tax_accuracy_dataset,
)
from finance_ai.evaluation.hallucination_evaluator import evaluate_hallucination_dataset
from finance_ai.evaluation.models import (
    AccuracyAggregateResult,
    EvaluationReport,
    HallucinationAggregateResult,
    HallucinationDataset,
    PerformanceAggregateResult,
    QualityAggregateResult,
    RAGAggregateResult,
    RecommendationSafetyAggregateResult,
    RoutingAggregateResult,
)
from finance_ai.evaluation.performance_evaluator import evaluate_performance_dataset
from finance_ai.evaluation.quality_evaluator import evaluate_quality_dataset
from finance_ai.evaluation.rag_evaluator import evaluate_rag_dataset
from finance_ai.evaluation.recommendation_safety_evaluator import evaluate_safety_dataset
from finance_ai.evaluation.routing_evaluator import evaluate_routing_dataset
from finance_ai.evaluation.safety_response_generator import generate_safety_responses
from finance_ai.rag.vector_store import FinanceVectorStore
from finance_ai.core.logging import get_logger

logger = get_logger(__name__)


class EvaluationRunner:
    """Orchestrates evaluation across all dimensions.

    Attributes:
        chat_model: The model being evaluated.
        vector_store: Vector store for RAG evaluation.
        judge_model: Optional LLM for quality/hallucination judging.
        data_dir: Directory containing YAML datasets.

    Example:
        >>> runner = EvaluationRunner(model, store, "google", "gemini")
        >>> report = runner.run_all()
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        chat_model: BaseChatModel,
        vector_store: FinanceVectorStore | None,
        llm_provider: str,
        llm_model: str,
        judge_model: BaseChatModel | None = None,
        db_session_factory: Callable[[], Session] | None = None,
        data_dir: str = "data/evaluation",
    ) -> None:
        """Initialize the evaluation runner.

        Args:
            chat_model: Model being evaluated.
            vector_store: Vector store for RAG.
            llm_provider: Provider name (google, openrouter, etc.).
            llm_model: Model name being evaluated.
            judge_model: Optional judge model for quality eval.
            db_session_factory: Optional DB session factory.
            data_dir: Path to evaluation dataset directory.
        """
        self._model = chat_model
        self._store = vector_store
        self._provider = llm_provider
        self._llm_model = llm_model
        self._judge = judge_model
        self._db_factory = db_session_factory
        self._data_dir = data_dir

    def run_routing(self) -> RoutingAggregateResult:
        """Run routing intent classification evaluation.

        Returns:
            RoutingAggregateResult with accuracy metrics.
        """
        dataset = load_routing_dataset(f"{self._data_dir}/routing_dataset.yaml")
        return evaluate_routing_dataset(self._model, dataset)

    def run_rag(self) -> RAGAggregateResult:
        """Run RAG retrieval evaluation.

        Returns:
            RAGAggregateResult with retrieval metrics.

        Raises:
            ValueError: If vector_store is not configured.
        """
        if self._store is None:
            raise ValueError("vector_store is required for RAG evaluation")
        dataset = load_rag_dataset(f"{self._data_dir}/rag_retrieval_dataset.yaml")
        return evaluate_rag_dataset(self._store, dataset)

    def run_tax_accuracy(self) -> AccuracyAggregateResult:
        """Run tax agent accuracy evaluation.

        Returns:
            AccuracyAggregateResult with accuracy metrics.
        """
        dataset = load_tax_accuracy_dataset(f"{self._data_dir}/tax_accuracy_dataset.yaml")
        return evaluate_tax_accuracy_dataset(self._model, dataset, self._db_factory)

    def run_tax_accuracy_forced(self) -> AccuracyAggregateResult:
        """Run tax accuracy evaluation with forced tool calling.

        Uses eval-only graph that forces calculate_thai_tax tool,
        isolating tool argument accuracy from tool selection accuracy.

        Returns:
            AccuracyAggregateResult with forced-tool accuracy metrics.
        """
        dataset = load_tax_accuracy_dataset(f"{self._data_dir}/tax_accuracy_dataset.yaml")
        return evaluate_tax_accuracy_dataset_forced(self._model, dataset, self._db_factory)

    def run_hallucination(self) -> HallucinationAggregateResult:
        """Run anti-hallucination compliance evaluation.

        Generates agent responses for each case, then checks
        for hallucinations using rule-based and optional LLM judge.

        Returns:
            HallucinationAggregateResult with compliance metrics.
        """
        dataset = load_hallucination_dataset(
            f"{self._data_dir}/hallucination_dataset.yaml",
        )
        responses = _generate_agent_responses(
            self._model,
            dataset,
            self._db_factory,
        )
        use_judge = self._judge is not None
        return evaluate_hallucination_dataset(
            dataset,
            agent_responses=responses,
            use_llm_judge=use_judge,
            judge_model=self._judge,
        )

    def run_quality(self) -> QualityAggregateResult:
        """Run LLM-as-Judge quality evaluation.

        Returns:
            QualityAggregateResult with quality scores.

        Raises:
            ValueError: If no judge model is configured.
        """
        if self._judge is None:
            raise ValueError("Judge model required for quality evaluation")
        dataset = load_quality_dataset(f"{self._data_dir}/quality_dataset.yaml")
        return evaluate_quality_dataset(self._model, self._judge, dataset)

    def run_recommendation_safety(self) -> RecommendationSafetyAggregateResult:
        """Run recommendation safety compliance evaluation.

        Generates Recommendation Agent answers on an isolated database
        (users seeded at each case's risk level), then checks every
        final answer against the deterministic guardrail.

        Returns:
            RecommendationSafetyAggregateResult with the compliance rate.
        """
        dataset = load_recommendation_safety_dataset(
            f"{self._data_dir}/recommendation_safety_dataset.yaml",
        )
        responses, tool_flags = generate_safety_responses(
            self._model,
            dataset,
            self._db_factory,
        )
        return evaluate_safety_dataset(dataset, responses, tool_flags=tool_flags)

    def run_performance(self) -> PerformanceAggregateResult:
        """Run performance (latency/cost) evaluation.

        Returns:
            PerformanceAggregateResult with performance metrics.
        """
        dataset = load_quality_dataset(f"{self._data_dir}/quality_dataset.yaml")
        return evaluate_performance_dataset(
            model=self._model,
            dataset=dataset,
            model_name=self._llm_model,
            db_session_factory=self._db_factory,
        )

    def run_all(
        self,
        skip: set[str] | None = None,
    ) -> EvaluationReport:
        """Run all evaluation dimensions and produce a report.

        Args:
            skip: Set of dimension names to skip.

        Returns:
            EvaluationReport combining all results.
        """
        skip = skip or set()
        report = EvaluationReport(
            report_id=str(uuid.uuid4()),
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            llm_provider=self._provider,
            llm_model=self._llm_model,
        )
        if "routing" not in skip:
            report.routing = _safe_run("routing", self.run_routing)
        if "rag" not in skip:
            report.rag_retrieval = _safe_run("rag", self.run_rag)
        if "accuracy" not in skip:
            report.tax_accuracy = _safe_run("accuracy", self.run_tax_accuracy)
        if "hallucination" not in skip:
            report.hallucination = _safe_run("hallucination", self.run_hallucination)
        if "quality" not in skip and self._judge is not None:
            report.quality = _safe_run("quality", self.run_quality)
        if "safety" not in skip:
            report.recommendation_safety = _safe_run("safety", self.run_recommendation_safety)
        if "performance" not in skip:
            report.performance = _safe_run("performance", self.run_performance)
        return report


def _generate_agent_responses(
    model: BaseChatModel,
    dataset: HallucinationDataset,
    db_session_factory: Callable[[], Session] | None = None,
) -> dict[str, str]:
    """Generate agent responses for each hallucination case.

    Args:
        model: Chat model to generate responses.
        dataset: Hallucination dataset with queries.
        db_session_factory: Optional DB session factory.

    Returns:
        Dict mapping case_id to agent response text.
    """
    responses: dict[str, str] = {}
    for case in dataset.cases:
        try:
            result = orchestrate_query(
                query=case.query,
                chat_model=model,
                db_session_factory=db_session_factory,
            )
            responses[case.case_id] = result.get("response", "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to generate response for %s: %s", case.case_id, exc)
            responses[case.case_id] = ""
    return responses


def _safe_run(name: str, method: Callable[[], Any]) -> Any:
    """Run an evaluation method, returning None on error.

    Args:
        name: Dimension name for logging.
        method: Evaluation method to call.

    Returns:
        Result or None if an error occurred.
    """
    try:
        logger.info("[%s] running...", name)
        result = method()
        logger.info("[%s] done", name)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] SKIPPED — %s", name, exc)
        return None
