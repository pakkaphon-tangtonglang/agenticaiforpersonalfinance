"""CRUD operations for the RiskAssessment model."""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.database.crud.base_crud import BaseCRUD
from finance_ai.database.models.risk_assessment import RiskAssessment


class RiskAssessmentCRUD(BaseCRUD[RiskAssessment]):
    """CRUD operations for SEC suitability (risk assessment) results."""

    def __init__(self) -> None:
        """Initialize with the RiskAssessment model."""
        super().__init__(RiskAssessment)

    def create_assessment(self, session: Session, assessment: RiskAssessment) -> RiskAssessment:
        """Persist a new risk assessment. Caller must commit.

        Named create_assessment (not create) to avoid an incompatible
        override of BaseCRUD.create(session, **kwargs).

        Args:
            session: Database session.
            assessment: A fully populated RiskAssessment instance.

        Returns:
            The persisted RiskAssessment (id assigned).

        Example:
            >>> saved = RiskAssessmentCRUD().create_assessment(session, assessment)
            >>> session.commit()
        """
        session.add(assessment)
        session.flush()
        return assessment

    def get_latest_by_user(self, session: Session, user_id: str) -> Optional[RiskAssessment]:
        """Return the user's newest assessment, or None when they have none.

        Orders by created_at (newest first) with id as a deterministic
        tie-breaker for identical timestamps.

        Args:
            session: Database session.
            user_id: UUID string of the user.

        Returns:
            The latest RiskAssessment or None if the user has none.

        Example:
            >>> latest = RiskAssessmentCRUD().get_latest_by_user(session, user_id)
        """
        statement = (
            select(RiskAssessment)
            .where(RiskAssessment.user_id == user_id)
            .order_by(RiskAssessment.created_at.desc(), RiskAssessment.id.desc())
            .limit(1)
        )
        return session.execute(statement).scalar_one_or_none()
