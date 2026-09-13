"""Tests for RiskAssessment model."""

import pytest
from sqlalchemy.orm import Session

from finance_ai.database.models.risk_assessment import RiskAssessment
from finance_ai.database.models.user import User


class TestRiskAssessment:
    """Tests for RiskAssessment model."""

    def test_create_risk_assessment(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Create a risk assessment with required fields."""
        assessment = RiskAssessment(
            user_id=sample_user.id,
            answers={"1": "ง", "4": ["ก", "ง"], "11": "ข"},
            total_score=40,
            risk_level=5,
            risk_category="เสี่ยงสูงมาก",
        )
        test_session.add(assessment)
        test_session.commit()
        test_session.refresh(assessment)

        assert assessment.id is not None
        assert assessment.answers == {"1": "ง", "4": ["ก", "ง"], "11": "ข"}
        assert assessment.total_score == 40
        assert assessment.risk_level == 5
        assert assessment.risk_category == "เสี่ยงสูงมาก"
        assert assessment.created_at is not None
        assert assessment.updated_at is not None

    def test_user_relationship_back_populates(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Assessments appear on the user via the relationship."""
        assessment = RiskAssessment(
            user_id=sample_user.id,
            answers={},
            total_score=10,
            risk_level=1,
            risk_category="เสี่ยงต่ำ",
        )
        test_session.add(assessment)
        test_session.commit()

        assert assessment.user.id == sample_user.id
        assert assessment in sample_user.risk_assessments
