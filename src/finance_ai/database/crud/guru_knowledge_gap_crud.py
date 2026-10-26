"""CRUD operations for the GuruKnowledgeGap model."""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from finance_ai.database.crud.base_crud import BaseCRUD
from finance_ai.database.models.guru_knowledge_gap import GuruKnowledgeGap


class GuruKnowledgeGapCRUD(BaseCRUD[GuruKnowledgeGap]):
    """CRUD operations for guru knowledge gaps."""

    def __init__(self) -> None:
        """Initialize with the GuruKnowledgeGap model."""
        super().__init__(GuruKnowledgeGap)

    def count_by_query(
        self,
        session: Session,
        limit: int,
        guru_id: str | None = None,
    ) -> list[tuple[str, str, int]]:
        """Count gaps per (guru, question), most frequent first.

        Args:
            session: Database session.
            limit: Maximum rows to return.
            guru_id: Optional persona filter.

        Returns:
            List of (guru_id, query, count) tuples.

        Example:
            >>> GuruKnowledgeGapCRUD().count_by_query(session, 10)
            [('buffett', 'ทองคำ', 3)]
        """
        # pylint false positive: SQLAlchemy func.count is callable at runtime.
        hits = func.count(GuruKnowledgeGap.id).label("hits")  # pylint: disable=not-callable
        statement = select(GuruKnowledgeGap.guru_id, GuruKnowledgeGap.query, hits).group_by(
            GuruKnowledgeGap.guru_id, GuruKnowledgeGap.query
        )
        if guru_id:
            statement = statement.where(GuruKnowledgeGap.guru_id == guru_id)
        rows = session.execute(statement.order_by(hits.desc()).limit(limit)).all()
        return [(row.guru_id, row.query, int(row.hits)) for row in rows]
