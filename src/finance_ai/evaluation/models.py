"""Pydantic models for evaluation datasets and results."""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

# =============================================================================
# Dataset Models (Input)
# =============================================================================


class EvaluationCase(BaseModel):
    """Base model for a single evaluation test case.

    Attributes:
        case_id: Unique identifier for this test case.
        query: User query in Thai.
        description: Human-readable description of what this tests.
        tags: Tags for filtering and grouping.
    """

    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    description: str = Field(default="")
    tags: list[str] = Field(default_factory=list)


class RoutingCase(EvaluationCase):
    """Test case for intent routing evaluation.

    Attributes:
        expected_intent: The correct intent classification.
        difficulty: How hard this case is to classify.
    """

    expected_intent: Literal[
        "tax",
        "expense",
        "asset_monitoring",
        "planning",
        "recommendation",
        "report",
        "general",
        "unknown",
    ]
    difficulty: Literal["easy", "medium", "hard"] = Field(default="medium")


class TaxAccuracyCase(EvaluationCase):
    """Test case with ground truth tax calculation.

    Attributes:
        gross_income: Annual gross income in THB.
        deductions_by_type: Mapping of deduction type to amount.
        withholding_tax_paid: Tax already withheld.
        expected_total_tax: Ground truth total tax.
        expected_effective_rate: Ground truth effective rate.
        tolerance_thb: Acceptable absolute error in THB.
    """

    gross_income: Decimal = Field(ge=Decimal("0"))
    deductions_by_type: dict[str, Decimal] = Field(default_factory=dict)
    withholding_tax_paid: Decimal = Field(default=Decimal("0"))
    expected_total_tax: Decimal = Field(ge=Decimal("0"))
    expected_effective_rate: Decimal = Field(ge=Decimal("0"))
    tolerance_thb: Decimal = Field(default=Decimal("100"))


class RAGCase(EvaluationCase):
    """Test case for RAG retrieval evaluation.

    Attributes:
        domain_filter: Optional domain to filter results.
        expected_source_files: Source files that should appear in results.
        expected_keywords: Keywords that should appear in retrieved content.
    """

    domain_filter: str | None = Field(default=None)
    expected_source_files: list[str] = Field(min_length=1)
    expected_keywords: list[str] = Field(default_factory=list)


class HallucinationCase(EvaluationCase):
    """Test case for anti-hallucination compliance.

    Attributes:
        category: Type of hallucination to check for.
        known_facts: Ground truth facts the response must not contradict.
        forbidden_patterns: Regex patterns that indicate hallucination.
    """

    category: Literal[
        "fabricated_number",
        "fabricated_law",
        "missing_citation",
        "incorrect_calculation",
    ]
    known_facts: list[str] = Field(min_length=1)
    forbidden_patterns: list[str] = Field(default_factory=list)


class QualityCase(EvaluationCase):
    """Test case for LLM-as-Judge quality scoring.

    Attributes:
        expected_agent: Which agent should handle this query.
        quality_criteria: Dimensions to evaluate.
    """

    expected_agent: str = Field(min_length=1)
    quality_criteria: list[str] = Field(
        default_factory=lambda: [
            "relevance",
            "completeness",
            "accuracy",
            "thai_language_quality",
        ]
    )


# =============================================================================
# Dataset Wrappers
# =============================================================================


class RoutingDataset(BaseModel):
    """Complete routing evaluation dataset."""

    name: str = Field(default="routing_evaluation")
    version: str
    cases: list[RoutingCase]


class TaxAccuracyDataset(BaseModel):
    """Complete tax accuracy evaluation dataset."""

    name: str = Field(default="tax_accuracy_evaluation")
    version: str
    cases: list[TaxAccuracyCase]


class RAGDataset(BaseModel):
    """Complete RAG retrieval evaluation dataset."""

    name: str = Field(default="rag_retrieval_evaluation")
    version: str
    cases: list[RAGCase]


class HallucinationDataset(BaseModel):
    """Complete hallucination evaluation dataset."""

    name: str = Field(default="hallucination_evaluation")
    version: str
    cases: list[HallucinationCase]


class QualityDataset(BaseModel):
    """Complete quality evaluation dataset."""

    name: str = Field(default="quality_evaluation")
    version: str
    cases: list[QualityCase]


# =============================================================================
# Result Models (Output)
# =============================================================================


class RoutingResult(BaseModel):
    """Result of a single routing evaluation."""

    case_id: str
    query: str
    expected_intent: str
    predicted_intent: str
    predicted_confidence: Decimal
    is_correct: bool
    latency_seconds: float


class TaxAccuracyResult(BaseModel):
    """Result of a single tax accuracy evaluation."""

    case_id: str
    query: str
    expected_total_tax: Decimal
    extracted_total_tax: Decimal | None
    absolute_error_thb: Decimal | None
    is_within_tolerance: bool
    agent_response: str
    latency_seconds: float


class RAGRetrievalResult(BaseModel):
    """Result of a single RAG retrieval evaluation."""

    case_id: str
    query: str
    retrieved_sources: list[str]
    expected_sources: list[str]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    keyword_hit_rate: float
    latency_seconds: float


class HallucinationResult(BaseModel):
    """Result of a single hallucination check."""

    case_id: str
    query: str
    category: str
    violations_found: list[str]
    is_compliant: bool
    agent_response: str
    latency_seconds: float


class QualityScore(BaseModel):
    """LLM-as-Judge scores for a single response (1-5 each)."""

    relevance: Decimal = Field(ge=Decimal("1"), le=Decimal("5"))
    completeness: Decimal = Field(ge=Decimal("1"), le=Decimal("5"))
    accuracy: Decimal = Field(ge=Decimal("1"), le=Decimal("5"))
    thai_language_quality: Decimal = Field(ge=Decimal("1"), le=Decimal("5"))
    overall: Decimal = Field(ge=Decimal("1"), le=Decimal("5"))
    judge_reasoning: str


class QualityResult(BaseModel):
    """Result of a single quality evaluation."""

    case_id: str
    query: str
    agent_response: str
    scores: QualityScore
    latency_seconds: float


class PerformanceResult(BaseModel):
    """Performance metrics for a single query."""

    case_id: str
    query: str
    agent: str
    total_latency_seconds: float
    rag_latency_seconds: float | None = None
    llm_latency_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: Decimal | None = None


# =============================================================================
# Aggregate Result Models
# =============================================================================


class RoutingAggregateResult(BaseModel):
    """Aggregated routing evaluation metrics."""

    total_cases: int
    correct_count: int
    accuracy: Decimal
    per_intent_accuracy: dict[str, Decimal]
    confusion_matrix: dict[str, dict[str, int]]
    mean_latency_seconds: float
    results: list[RoutingResult]


class RAGAggregateResult(BaseModel):
    """Aggregated RAG retrieval metrics."""

    total_cases: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    mean_keyword_hit_rate: float
    per_domain_precision: dict[str, float]
    mean_latency_seconds: float
    results: list[RAGRetrievalResult]


class AccuracyAggregateResult(BaseModel):
    """Aggregated agent accuracy metrics."""

    agent_type: str
    total_cases: int
    within_tolerance_count: int
    accuracy_rate: Decimal
    mean_absolute_error_thb: Decimal | None
    mean_latency_seconds: float
    results: list[TaxAccuracyResult]


class HallucinationAggregateResult(BaseModel):
    """Aggregated hallucination compliance metrics."""

    total_cases: int
    compliant_count: int
    compliance_rate: Decimal
    violations_by_category: dict[str, int]
    mean_latency_seconds: float
    results: list[HallucinationResult]


class QualityAggregateResult(BaseModel):
    """Aggregated quality scores."""

    total_cases: int
    mean_relevance: Decimal
    mean_completeness: Decimal
    mean_accuracy: Decimal
    mean_thai_language_quality: Decimal
    mean_overall: Decimal
    per_agent_scores: dict[str, dict[str, Decimal]]
    mean_latency_seconds: float
    results: list[QualityResult]


class PerformanceAggregateResult(BaseModel):
    """Aggregated performance metrics."""

    total_queries: int
    mean_total_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    total_input_tokens: int
    total_output_tokens: int
    total_estimated_cost_usd: Decimal
    per_agent_latency: dict[str, float]
    results: list[PerformanceResult]


class EvaluationReport(BaseModel):
    """Complete evaluation report combining all dimensions."""

    report_id: str
    timestamp: str
    llm_provider: str
    llm_model: str
    routing: RoutingAggregateResult | None = None
    rag_retrieval: RAGAggregateResult | None = None
    tax_accuracy: AccuracyAggregateResult | None = None
    hallucination: HallucinationAggregateResult | None = None
    quality: QualityAggregateResult | None = None
    performance: PerformanceAggregateResult | None = None
