"""GuruKnowledgeGap database model: guru questions the knowledge base could not answer."""

from decimal import Decimal
from typing import Optional

from sqlalchemy import Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from finance_ai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class GuruKnowledgeGap(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One guru question whose best retrieved chunk was missing or too far away.

    Used to decide which sources to fetch, review and index next. No user id
    is stored: gaps describe the knowledge base, not the person asking.

    Example:
        >>> gap = GuruKnowledgeGap(guru_id="buffett", query="ทองคำ", best_distance=Decimal("1.3"))
    """

    __tablename__ = "guru_knowledge_gaps"

    guru_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    query: Mapped[str] = mapped_column(String(500), nullable=False)
    best_distance: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
