"""Knowledge-gap logging for the guru knowledge base.

When a guru search finds nothing close enough, the question is logged so
the most-asked gaps can be filled offline (fetch -> review -> index),
instead of fetching unreviewed web pages at answer time.
"""

from collections.abc import Callable
from decimal import Decimal

from sqlalchemy.orm import Session

from finance_ai.core.logging import get_logger
from finance_ai.database.crud.guru_knowledge_gap_crud import GuruKnowledgeGapCRUD
from finance_ai.database.models.guru_knowledge_gap import GuruKnowledgeGap
from finance_ai.rag.document_models import RetrievalResult

logger = get_logger(__name__)

MAX_GAP_QUERY_LENGTH = 500
_gap_crud = GuruKnowledgeGapCRUD()


def best_distance(results: list[RetrievalResult]) -> Decimal | None:
    """Return the closest chunk's distance (Chroma: lower is closer).

    Args:
        results: Retrieval results for one query.

    Returns:
        Smallest distance as Decimal, or None when there are no results.

    Example:
        >>> best_distance([r_far, r_close])
        Decimal('0.6')
    """
    if not results:
        return None
    return Decimal(str(min(result.relevance_score for result in results)))


def is_knowledge_gap(results: list[RetrievalResult], threshold: float) -> bool:
    """Decide whether a search found nothing relevant enough.

    Args:
        results: Retrieval results for one query.
        threshold: Maximum distance that still counts as relevant.

    Returns:
        True when there are no results or the closest one is too far.

    Example:
        >>> is_knowledge_gap([], threshold=1.0)
        True
    """
    distance = best_distance(results)
    return distance is None or distance > Decimal(str(threshold))


def record_knowledge_gap(
    session: Session,
    guru_id: str,
    query: str,
    distance: Decimal | None,
) -> GuruKnowledgeGap:
    """Store one gap and commit.

    Args:
        session: Database session.
        guru_id: Persona that could not answer from its sources.
        query: The search query (truncated to MAX_GAP_QUERY_LENGTH).
        distance: Best distance found, or None.

    Returns:
        The stored GuruKnowledgeGap.

    Example:
        >>> record_knowledge_gap(session, "buffett", "ทองคำ", Decimal("1.3"))
    """
    gap = _gap_crud.create(
        session, guru_id=guru_id, query=query[:MAX_GAP_QUERY_LENGTH], best_distance=distance
    )
    session.commit()
    return gap


def record_gap_if_needed(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    session_factory: Callable[[], Session] | None,
    guru_id: str,
    query: str,
    results: list[RetrievalResult],
    threshold: float,
) -> bool:
    """Log the query when it is a knowledge gap. Never raises.

    Args:
        session_factory: Session factory from graph state; None skips logging.
        guru_id: Active persona.
        query: The search query.
        results: Retrieval results for the query.
        threshold: Maximum distance that still counts as relevant.

    Returns:
        True when a gap was recorded.

    Example:
        >>> record_gap_if_needed(factory, "buffett", "ทองคำ", results, 1.0)
        True
    """
    if session_factory is None or not is_knowledge_gap(results, threshold):
        return False
    try:
        session = session_factory()
        try:
            record_knowledge_gap(session, guru_id, query, best_distance(results))
        finally:
            session.close()
    except Exception as error:  # pylint: disable=broad-exception-caught
        # Broad by design: logging a gap must never break a guru answer.
        logger.warning("Failed to record guru knowledge gap for %s: %s", guru_id, error)
        return False
    return True


def list_top_gaps(
    session: Session,
    limit: int = 20,
    guru_id: str | None = None,
) -> list[tuple[str, str, int]]:
    """List the most frequent gaps.

    Args:
        session: Database session.
        limit: Maximum rows.
        guru_id: Optional persona filter.

    Returns:
        List of (guru_id, query, count), most frequent first.

    Example:
        >>> list_top_gaps(session, 5)
        [('buffett', 'ทองคำ', 3)]
    """
    return _gap_crud.count_by_query(session, limit, guru_id)
