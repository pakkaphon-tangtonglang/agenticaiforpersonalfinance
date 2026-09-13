"""RiskAssessment database model for SEC suitability questionnaire results."""

from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from finance_ai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from finance_ai.database.models.user import User


class RiskAssessment(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Result of the SEC suitability (risk assessment) questionnaire.

    Stores the raw answers (question id -> choice key), the computed total
    score (10-40), and the derived risk level (1-5) with its Thai category.

    Example:
        >>> assessment = RiskAssessment(
        ...     user_id=uid,
        ...     answers={"1": "ง", "4": ["ก", "ง"], "11": "ข"},
        ...     total_score=40,
        ...     risk_level=5,
        ...     risk_category="เสี่ยงสูงมาก",
        ... )
    """

    __tablename__ = "risk_assessments"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False, index=True
    )
    answers: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    total_score: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_level: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5
    risk_category: Mapped[str] = mapped_column(String(50), nullable=False)  # "เสี่ยงต่ำ"

    user: Mapped["User"] = relationship(back_populates="risk_assessments")
