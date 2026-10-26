"""Tests for guru knowledge-gap detection and logging."""

from decimal import Decimal
from unittest.mock import MagicMock

from sqlalchemy.orm import Session

from finance_ai.database.crud.guru_knowledge_gap_crud import GuruKnowledgeGapCRUD
from finance_ai.rag.document_models import RetrievalResult
from finance_ai.tools.guru_gap_service import (
    MAX_GAP_QUERY_LENGTH,
    best_distance,
    is_knowledge_gap,
    list_top_gaps,
    record_knowledge_gap,
    record_gap_if_needed,
)


def _result(distance: float) -> RetrievalResult:
    """A retrieval result with the given Chroma distance (lower = closer)."""
    return RetrievalResult(
        content="x", source_file="a.md", domain="guru", relevance_score=distance, guru_id="buffett"
    )


class TestGapDetection:
    """Tests for best_distance and is_knowledge_gap."""

    def test_best_distance_is_the_smallest(self) -> None:
        """Chroma distances: the closest chunk has the lowest score."""
        assert best_distance([_result(1.4), _result(0.6)]) == Decimal("0.6")

    def test_no_results_is_a_gap(self) -> None:
        """Nothing indexed for the guru is always a gap."""
        assert best_distance([]) is None
        assert is_knowledge_gap([], threshold=1.0)

    def test_far_results_are_a_gap(self) -> None:
        """Even the closest chunk is beyond the threshold."""
        assert is_knowledge_gap([_result(1.3), _result(1.6)], threshold=1.0)

    def test_close_result_is_not_a_gap(self) -> None:
        """One relevant chunk is enough."""
        assert not is_knowledge_gap([_result(0.4), _result(1.6)], threshold=1.0)


class TestRecordAndList:
    """Tests for writing and aggregating gaps."""

    def test_record_truncates_long_queries(self, test_session: Session) -> None:
        """Stored queries are capped so the log stays small."""
        gap = record_knowledge_gap(test_session, "dalio", "ก" * 900, Decimal("1.2345"))

        assert len(gap.query) == MAX_GAP_QUERY_LENGTH
        assert gap.best_distance == Decimal("1.2345")

    def test_list_top_gaps_groups_by_guru_and_query(self, test_session: Session) -> None:
        """Repeated questions rank first with their count."""
        for _ in range(3):
            record_knowledge_gap(test_session, "buffett", "ทองคำ", None)
        record_knowledge_gap(test_session, "dalio", "คริปโต", Decimal("1.5"))

        top = list_top_gaps(test_session, limit=10)

        assert top[0] == ("buffett", "ทองคำ", 3)
        assert ("dalio", "คริปโต", 1) in top

    def test_list_top_gaps_can_filter_by_guru(self, test_session: Session) -> None:
        """Only the requested guru's gaps are returned."""
        record_knowledge_gap(test_session, "buffett", "ทองคำ", None)
        record_knowledge_gap(test_session, "dalio", "คริปโต", None)

        assert list_top_gaps(test_session, limit=10, guru_id="dalio") == [("dalio", "คริปโต", 1)]


class TestRecordGapIfNeeded:
    """Tests for the never-raising hook used by the search tool."""

    def test_records_only_gaps(self, test_session: Session) -> None:
        """A close match writes nothing; a far match writes one row."""
        factory = MagicMock(return_value=test_session)
        test_session.close = MagicMock()  # type: ignore[method-assign]

        assert not record_gap_if_needed(factory, "buffett", "q1", [_result(0.2)], threshold=1.0)
        assert record_gap_if_needed(factory, "buffett", "q2", [_result(1.8)], threshold=1.0)

        assert [gap.query for gap in GuruKnowledgeGapCRUD().get_all(test_session)] == ["q2"]

    def test_missing_session_factory_is_ignored(self) -> None:
        """No database means no logging, and no error."""
        assert not record_gap_if_needed(None, "buffett", "q", [], threshold=1.0)

    def test_database_errors_never_raise(self) -> None:
        """Gap logging must never break a guru answer."""
        broken_factory = MagicMock(side_effect=RuntimeError("db down"))

        assert not record_gap_if_needed(broken_factory, "buffett", "q", [], threshold=1.0)
