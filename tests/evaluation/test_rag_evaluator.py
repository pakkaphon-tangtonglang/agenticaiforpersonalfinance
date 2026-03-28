"""Tests for RAG retrieval evaluator."""

from unittest.mock import MagicMock

from finance_ai.evaluation.models import RAGCase, RAGDataset
from finance_ai.evaluation.rag_evaluator import (
    evaluate_rag_dataset,
    evaluate_single_rag_case,
)
from finance_ai.rag.document_models import RetrievalResult


def _make_mock_store(
    results: list[RetrievalResult],
) -> MagicMock:
    """Create a mock vector store returning given results."""
    store = MagicMock()
    store.search.return_value = results
    return store


class TestEvaluateSingleRAGCase:
    """Tests for evaluate_single_rag_case."""

    def test_perfect_retrieval(self) -> None:
        """Test when retrieval returns exactly the expected source."""
        results = [
            RetrievalResult(
                content="ค่าลดหย่อนส่วนตัว 60,000 บาท",
                source_file="tax_deductions_guide.md",
                domain="tax",
                relevance_score=0.95,
            ),
        ]
        store = _make_mock_store(results)
        case = RAGCase(
            case_id="rag_001",
            query="ค่าลดหย่อนส่วนตัว",
            domain_filter="tax",
            expected_source_files=["tax_deductions_guide.md"],
            expected_keywords=["60,000"],
        )
        result = evaluate_single_rag_case(store, case, top_k=3)
        assert result.recall_at_k == 1.0
        assert result.keyword_hit_rate == 1.0
        assert result.reciprocal_rank == 1.0

    def test_no_relevant_results(self) -> None:
        """Test when retrieval returns no relevant results."""
        results = [
            RetrievalResult(
                content="unrelated content",
                source_file="other.md",
                domain="general",
                relevance_score=0.3,
            ),
        ]
        store = _make_mock_store(results)
        case = RAGCase(
            case_id="rag_002",
            query="test",
            expected_source_files=["tax_deductions_guide.md"],
        )
        result = evaluate_single_rag_case(store, case, top_k=3)
        assert result.recall_at_k == 0.0
        assert result.reciprocal_rank == 0.0

    def test_empty_results(self) -> None:
        """Test when retrieval returns empty results."""
        store = _make_mock_store([])
        case = RAGCase(
            case_id="rag_003",
            query="test",
            expected_source_files=["file.md"],
        )
        result = evaluate_single_rag_case(store, case, top_k=3)
        assert result.precision_at_k == 0.0
        assert result.recall_at_k == 0.0

    def test_latency_recorded(self) -> None:
        """Test that latency is recorded."""
        store = _make_mock_store([])
        case = RAGCase(
            case_id="rag_004",
            query="test",
            expected_source_files=["file.md"],
        )
        result = evaluate_single_rag_case(store, case)
        assert result.latency_seconds >= 0.0


class TestEvaluateRAGDataset:
    """Tests for evaluate_rag_dataset."""

    def test_aggregate_metrics(self) -> None:
        """Test that dataset evaluation aggregates correctly."""
        results = [
            RetrievalResult(
                content="content about tax deductions 60,000",
                source_file="tax_deductions_guide.md",
                domain="tax",
                relevance_score=0.9,
            ),
        ]
        store = _make_mock_store(results)
        dataset = RAGDataset(
            version="1.0",
            cases=[
                RAGCase(
                    case_id="rag_001",
                    query="ค่าลดหย่อน",
                    domain_filter="tax",
                    expected_source_files=["tax_deductions_guide.md"],
                ),
                RAGCase(
                    case_id="rag_002",
                    query="กองทุน RMF",
                    domain_filter="investment",
                    expected_source_files=["investment_mutual_funds.md"],
                ),
            ],
        )
        agg = evaluate_rag_dataset(store, dataset, top_k=3)
        assert agg.total_cases == 2
        assert len(agg.results) == 2
        assert "tax" in agg.per_domain_precision

    def test_empty_dataset(self) -> None:
        """Test evaluation with empty dataset."""
        store = _make_mock_store([])
        dataset = RAGDataset(version="1.0", cases=[])
        agg = evaluate_rag_dataset(store, dataset)
        assert agg.total_cases == 0
        assert agg.mean_precision_at_k == 0.0
