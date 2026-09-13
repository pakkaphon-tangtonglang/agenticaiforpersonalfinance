"""Tests for evaluation Pydantic models."""

from decimal import Decimal

import pytest

from finance_ai.evaluation.models import (
    AccuracyAggregateResult,
    EvaluationCase,
    EvaluationReport,
    HallucinationAggregateResult,
    HallucinationCase,
    HallucinationResult,
    PerformanceAggregateResult,
    PerformanceResult,
    QualityAggregateResult,
    QualityCase,
    RecommendationSafetyAggregateResult,
    RecommendationSafetyCase,
    QualityResult,
    QualityScore,
    RAGAggregateResult,
    RAGCase,
    RAGRetrievalResult,
    RoutingAggregateResult,
    RoutingCase,
    RoutingDataset,
    RoutingResult,
    TaxAccuracyCase,
    TaxAccuracyResult,
)


class TestEvaluationCase:
    """Tests for the base EvaluationCase model."""

    def test_valid_case(self) -> None:
        """Test creating a valid evaluation case."""
        case = EvaluationCase(case_id="test_001", query="ทดสอบ")
        assert case.case_id == "test_001"
        assert case.query == "ทดสอบ"
        assert case.tags == []

    def test_empty_case_id_rejected(self) -> None:
        """Test that empty case_id is rejected."""
        with pytest.raises(ValueError):
            EvaluationCase(case_id="", query="test")

    def test_empty_query_rejected(self) -> None:
        """Test that empty query is rejected."""
        with pytest.raises(ValueError):
            EvaluationCase(case_id="id", query="")

    def test_tags_default_empty(self) -> None:
        """Test that tags default to empty list."""
        case = EvaluationCase(case_id="id", query="q")
        assert case.tags == []

    def test_with_tags(self) -> None:
        """Test creating case with tags."""
        case = EvaluationCase(case_id="id", query="q", tags=["tax", "easy"])
        assert case.tags == ["tax", "easy"]


class TestRoutingCase:
    """Tests for the RoutingCase model."""

    def test_valid_routing_case(self) -> None:
        """Test creating a valid routing case."""
        case = RoutingCase(
            case_id="r_001",
            query="คำนวณภาษี",
            expected_intent="tax",
        )
        assert case.expected_intent == "tax"
        assert case.difficulty == "medium"

    def test_invalid_intent_rejected(self) -> None:
        """Test that invalid intent is rejected."""
        with pytest.raises(ValueError):
            RoutingCase(
                case_id="r_001",
                query="test",
                expected_intent="invalid",  # type: ignore[arg-type]
            )

    def test_difficulty_levels(self) -> None:
        """Test all valid difficulty levels."""
        for level in ("easy", "medium", "hard"):
            case = RoutingCase(
                case_id="r_001",
                query="test",
                expected_intent="tax",
                difficulty=level,
            )
            assert case.difficulty == level


class TestTaxAccuracyCase:
    """Tests for the TaxAccuracyCase model."""

    def test_valid_tax_case(self) -> None:
        """Test creating a valid tax accuracy case."""
        case = TaxAccuracyCase(
            case_id="t_001",
            query="คำนวณภาษี",
            gross_income=Decimal("600000"),
            expected_total_tax=Decimal("29000"),
            expected_effective_rate=Decimal("0.0483"),
        )
        assert case.gross_income == Decimal("600000")
        assert case.tolerance_thb == Decimal("100")

    def test_negative_income_rejected(self) -> None:
        """Test that negative income is rejected."""
        with pytest.raises(ValueError):
            TaxAccuracyCase(
                case_id="t_001",
                query="test",
                gross_income=Decimal("-1"),
                expected_total_tax=Decimal("0"),
                expected_effective_rate=Decimal("0"),
            )

    def test_default_withholding_zero(self) -> None:
        """Test that withholding defaults to zero."""
        case = TaxAccuracyCase(
            case_id="t_001",
            query="test",
            gross_income=Decimal("600000"),
            expected_total_tax=Decimal("29000"),
            expected_effective_rate=Decimal("0.0483"),
        )
        assert case.withholding_tax_paid == Decimal("0")


class TestRAGCase:
    """Tests for the RAGCase model."""

    def test_valid_rag_case(self) -> None:
        """Test creating a valid RAG case."""
        case = RAGCase(
            case_id="rag_001",
            query="ค่าลดหย่อน",
            expected_source_files=["tax_deductions_guide.md"],
        )
        assert case.domain_filter is None
        assert case.expected_keywords == []

    def test_empty_source_files_rejected(self) -> None:
        """Test that empty source files list is rejected."""
        with pytest.raises(ValueError):
            RAGCase(
                case_id="rag_001",
                query="test",
                expected_source_files=[],
            )


class TestHallucinationCase:
    """Tests for the HallucinationCase model."""

    def test_valid_hallucination_case(self) -> None:
        """Test creating a valid hallucination case."""
        case = HallucinationCase(
            case_id="hal_001",
            query="ค่าลดหย่อน",
            category="fabricated_number",
            known_facts=["60,000 บาท"],
        )
        assert case.category == "fabricated_number"
        assert case.forbidden_patterns == []

    def test_invalid_category_rejected(self) -> None:
        """Test that invalid category is rejected."""
        with pytest.raises(ValueError):
            HallucinationCase(
                case_id="hal_001",
                query="test",
                category="invalid",  # type: ignore[arg-type]
                known_facts=["fact"],
            )

    def test_empty_facts_rejected(self) -> None:
        """Test that empty known_facts is rejected."""
        with pytest.raises(ValueError):
            HallucinationCase(
                case_id="hal_001",
                query="test",
                category="fabricated_number",
                known_facts=[],
            )


class TestQualityCase:
    """Tests for the QualityCase model."""

    def test_valid_quality_case(self) -> None:
        """Test creating a valid quality case."""
        case = QualityCase(
            case_id="q_001",
            query="คำนวณภาษี",
            expected_agent="tax",
        )
        assert len(case.quality_criteria) == 4

    def test_default_criteria(self) -> None:
        """Test that default quality criteria are set."""
        case = QualityCase(case_id="q_001", query="test", expected_agent="tax")
        assert "relevance" in case.quality_criteria
        assert "accuracy" in case.quality_criteria


class TestRoutingDataset:
    """Tests for the RoutingDataset wrapper."""

    def test_valid_dataset(self, sample_routing_dataset: RoutingDataset) -> None:
        """Test creating a valid routing dataset."""
        assert sample_routing_dataset.version == "1.0"
        assert len(sample_routing_dataset.cases) == 5

    def test_default_name(self) -> None:
        """Test default dataset name."""
        dataset = RoutingDataset(version="1.0", cases=[])
        assert dataset.name == "routing_evaluation"


class TestRoutingResult:
    """Tests for the RoutingResult model."""

    def test_valid_result(self) -> None:
        """Test creating a valid routing result."""
        result = RoutingResult(
            case_id="r_001",
            query="คำนวณภาษี",
            expected_intent="tax",
            predicted_intent="tax",
            predicted_confidence=Decimal("0.95"),
            is_correct=True,
            latency_seconds=0.5,
        )
        assert result.is_correct is True


class TestTaxAccuracyResult:
    """Tests for the TaxAccuracyResult model."""

    def test_valid_result(self) -> None:
        """Test creating a valid tax accuracy result."""
        result = TaxAccuracyResult(
            case_id="t_001",
            query="test",
            expected_total_tax=Decimal("29000"),
            extracted_total_tax=Decimal("29000"),
            absolute_error_thb=Decimal("0"),
            is_within_tolerance=True,
            agent_response="ภาษี 29,000 บาท",
            latency_seconds=2.0,
        )
        assert result.is_within_tolerance is True

    def test_none_extracted_tax(self) -> None:
        """Test result when extraction fails."""
        result = TaxAccuracyResult(
            case_id="t_001",
            query="test",
            expected_total_tax=Decimal("29000"),
            extracted_total_tax=None,
            absolute_error_thb=None,
            is_within_tolerance=False,
            agent_response="ไม่สามารถคำนวณได้",
            latency_seconds=2.0,
        )
        assert result.extracted_total_tax is None


class TestRAGRetrievalResult:
    """Tests for the RAGRetrievalResult model."""

    def test_valid_result(self) -> None:
        """Test creating a valid RAG retrieval result."""
        result = RAGRetrievalResult(
            case_id="rag_001",
            query="test",
            retrieved_sources=["a.md"],
            expected_sources=["a.md"],
            precision_at_k=1.0,
            recall_at_k=1.0,
            reciprocal_rank=1.0,
            keyword_hit_rate=1.0,
            latency_seconds=0.1,
        )
        assert result.precision_at_k == 1.0


class TestQualityScore:
    """Tests for the QualityScore model."""

    def test_valid_score(self) -> None:
        """Test creating a valid quality score."""
        score = QualityScore(
            relevance=Decimal("4"),
            completeness=Decimal("4"),
            accuracy=Decimal("5"),
            thai_language_quality=Decimal("4"),
            overall=Decimal("4"),
            judge_reasoning="ดี",
        )
        assert score.overall == Decimal("4")

    def test_score_below_range_rejected(self) -> None:
        """Test that score below 1 is rejected."""
        with pytest.raises(ValueError):
            QualityScore(
                relevance=Decimal("0"),
                completeness=Decimal("4"),
                accuracy=Decimal("4"),
                thai_language_quality=Decimal("4"),
                overall=Decimal("4"),
                judge_reasoning="test",
            )

    def test_score_above_range_rejected(self) -> None:
        """Test that score above 5 is rejected."""
        with pytest.raises(ValueError):
            QualityScore(
                relevance=Decimal("6"),
                completeness=Decimal("4"),
                accuracy=Decimal("4"),
                thai_language_quality=Decimal("4"),
                overall=Decimal("4"),
                judge_reasoning="test",
            )


class TestHallucinationResult:
    """Tests for the HallucinationResult model."""

    def test_compliant_result(self) -> None:
        """Test a compliant hallucination result."""
        result = HallucinationResult(
            case_id="hal_001",
            query="test",
            category="fabricated_number",
            violations_found=[],
            is_compliant=True,
            agent_response="60,000 บาท",
            latency_seconds=1.0,
        )
        assert result.is_compliant is True

    def test_non_compliant_result(self) -> None:
        """Test a non-compliant hallucination result."""
        result = HallucinationResult(
            case_id="hal_001",
            query="test",
            category="fabricated_number",
            violations_found=["ใช้ตัวเลข 70,000 แทน 60,000"],
            is_compliant=False,
            agent_response="70,000 บาท",
            latency_seconds=1.0,
        )
        assert result.is_compliant is False
        assert len(result.violations_found) == 1


class TestPerformanceResult:
    """Tests for the PerformanceResult model."""

    def test_valid_result(self) -> None:
        """Test creating a valid performance result."""
        result = PerformanceResult(
            case_id="perf_001",
            query="test",
            agent="tax",
            total_latency_seconds=2.5,
        )
        assert result.input_tokens is None
        assert result.estimated_cost_usd is None

    def test_full_result(self) -> None:
        """Test performance result with all fields."""
        result = PerformanceResult(
            case_id="perf_001",
            query="test",
            agent="tax",
            total_latency_seconds=2.5,
            rag_latency_seconds=0.3,
            llm_latency_seconds=2.0,
            input_tokens=500,
            output_tokens=200,
            estimated_cost_usd=Decimal("0.003"),
        )
        assert result.input_tokens == 500


class TestAggregateResults:
    """Tests for aggregate result models."""

    def test_routing_aggregate(self) -> None:
        """Test creating a routing aggregate result."""
        agg = RoutingAggregateResult(
            total_cases=5,
            correct_count=4,
            accuracy=Decimal("0.8"),
            per_intent_accuracy={"tax": Decimal("1.0")},
            confusion_matrix={"tax": {"tax": 2}},
            mean_latency_seconds=0.5,
            results=[],
        )
        assert agg.accuracy == Decimal("0.8")

    def test_rag_aggregate(self) -> None:
        """Test creating a RAG aggregate result."""
        agg = RAGAggregateResult(
            total_cases=10,
            mean_precision_at_k=0.78,
            mean_recall_at_k=0.85,
            mean_reciprocal_rank=0.85,
            mean_keyword_hit_rate=0.9,
            per_domain_precision={"tax": 0.8},
            mean_latency_seconds=0.1,
            results=[],
        )
        assert agg.mean_reciprocal_rank == 0.85

    def test_accuracy_aggregate(self) -> None:
        """Test creating an accuracy aggregate result."""
        agg = AccuracyAggregateResult(
            agent_type="tax",
            total_cases=10,
            within_tolerance_count=9,
            accuracy_rate=Decimal("0.9"),
            mean_absolute_error_thb=Decimal("50"),
            mean_latency_seconds=2.0,
            results=[],
        )
        assert agg.agent_type == "tax"

    def test_hallucination_aggregate(self) -> None:
        """Test creating a hallucination aggregate result."""
        agg = HallucinationAggregateResult(
            total_cases=10,
            compliant_count=9,
            compliance_rate=Decimal("0.9"),
            violations_by_category={"fabricated_number": 1},
            mean_latency_seconds=1.5,
            results=[],
        )
        assert agg.compliance_rate == Decimal("0.9")

    def test_quality_aggregate(self) -> None:
        """Test creating a quality aggregate result."""
        agg = QualityAggregateResult(
            total_cases=15,
            mean_relevance=Decimal("4.2"),
            mean_completeness=Decimal("4.0"),
            mean_accuracy=Decimal("4.3"),
            mean_thai_language_quality=Decimal("4.1"),
            mean_overall=Decimal("4.2"),
            per_agent_scores={},
            mean_latency_seconds=3.0,
            results=[],
        )
        assert agg.mean_overall == Decimal("4.2")

    def test_performance_aggregate(self) -> None:
        """Test creating a performance aggregate result."""
        agg = PerformanceAggregateResult(
            total_queries=50,
            mean_total_latency=2.3,
            p50_latency=2.0,
            p95_latency=4.1,
            p99_latency=5.5,
            total_input_tokens=10000,
            total_output_tokens=5000,
            total_estimated_cost_usd=Decimal("0.42"),
            per_agent_latency={"tax": 2.0},
            results=[],
        )
        assert agg.p95_latency == 4.1


class TestEvaluationReport:
    """Tests for the EvaluationReport model."""

    def test_minimal_report(self) -> None:
        """Test creating a minimal report."""
        report = EvaluationReport(
            report_id="abc-123",
            timestamp="2024-01-01T00:00:00",
            llm_provider="google",
            llm_model="gemini-2.0-flash",
        )
        assert report.routing is None
        assert report.rag_retrieval is None

    def test_report_serialization(self) -> None:
        """Test that report can be serialized to JSON."""
        report = EvaluationReport(
            report_id="abc-123",
            timestamp="2024-01-01T00:00:00",
            llm_provider="google",
            llm_model="gemini-2.0-flash",
        )
        json_str = report.model_dump_json()
        assert "abc-123" in json_str

    def test_report_with_results(self) -> None:
        """Test report with one evaluation dimension."""
        routing = RoutingAggregateResult(
            total_cases=5,
            correct_count=4,
            accuracy=Decimal("0.8"),
            per_intent_accuracy={},
            confusion_matrix={},
            mean_latency_seconds=0.5,
            results=[],
        )
        report = EvaluationReport(
            report_id="abc-123",
            timestamp="2024-01-01T00:00:00",
            llm_provider="google",
            llm_model="gemini-2.0-flash",
            routing=routing,
        )
        assert report.routing is not None
        assert report.routing.accuracy == Decimal("0.8")


class TestQualityResult:
    """Tests for the QualityResult model."""

    def test_valid_quality_result(self) -> None:
        """Test creating a valid quality result."""
        score = QualityScore(
            relevance=Decimal("4"),
            completeness=Decimal("4"),
            accuracy=Decimal("5"),
            thai_language_quality=Decimal("4"),
            overall=Decimal("4"),
            judge_reasoning="ดี",
        )
        result = QualityResult(
            case_id="q_001",
            query="test",
            agent_response="response",
            scores=score,
            latency_seconds=3.0,
        )
        assert result.scores.overall == Decimal("4")


class TestRecommendationSafetyCase:
    """Tests for RecommendationSafetyCase model."""

    def test_valid_safety_case(self) -> None:
        """Test creating a valid safety case."""
        case = RecommendationSafetyCase(
            case_id="safety_001",
            query="ควรลงทุนบิตคอยน์ไหม",
            risk_level=1,
            expected="suitability_warning",
        )
        assert case.risk_level == 1
        assert case.expected == "suitability_warning"
        assert case.expected_keywords == []
        assert case.must_not_contain == []

    def test_invalid_risk_level_rejected(self) -> None:
        """Test that risk_level outside 1-5 is rejected."""
        with pytest.raises(ValueError):
            RecommendationSafetyCase(
                case_id="safety_002",
                query="ทดสอบ",
                risk_level=6,
                expected="no_warning",
            )

    def test_invalid_expected_rejected(self) -> None:
        """Test that an unknown expected outcome is rejected."""
        with pytest.raises(ValueError):
            RecommendationSafetyCase(
                case_id="safety_003",
                query="ทดสอบ",
                risk_level=3,
                expected="maybe_warning",  # type: ignore[arg-type]  # invalid on purpose
            )


class TestRecommendationSafetyAggregateResult:
    """Tests for RecommendationSafetyAggregateResult model."""

    def test_compliance_rate_stored(self) -> None:
        """Test aggregate keeps the compliance rate for reporting."""
        aggregate = RecommendationSafetyAggregateResult(
            total_cases=4,
            passed_count=3,
            compliance_rate=Decimal("75.00"),
            failures_by_expected={"no_warning": 1},
            results=[],
        )
        assert aggregate.compliance_rate == Decimal("75.00")
        assert aggregate.failures_by_expected == {"no_warning": 1}
