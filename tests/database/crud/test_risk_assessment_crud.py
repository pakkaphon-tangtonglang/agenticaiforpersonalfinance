"""Tests for RiskAssessmentCRUD."""

from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import Session

from finance_ai.database.crud.risk_assessment_crud import RiskAssessmentCRUD
from finance_ai.database.models.risk_assessment import RiskAssessment
from finance_ai.database.models.user import User


@pytest.fixture
def crud() -> RiskAssessmentCRUD:
    """Create a RiskAssessmentCRUD instance."""
    return RiskAssessmentCRUD()


def make_assessment(user_id: str, total_score: int, created_at: datetime) -> RiskAssessment:
    """Build a RiskAssessment with the given score and timestamp."""
    return RiskAssessment(
        user_id=user_id,
        answers={"1": "ง", "4": ["ง"]},
        total_score=total_score,
        risk_level=5,
        risk_category="เสี่ยงสูงมาก",
        created_at=created_at,
    )


class TestRiskAssessmentCRUD:
    """Tests for RiskAssessmentCRUD operations."""

    def test_create_and_get_latest(
        self,
        crud: RiskAssessmentCRUD,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Create an assessment and fetch it back as the user's latest."""
        created_at = datetime(2026, 4, 10, tzinfo=timezone.utc)
        assessment = make_assessment(sample_user.id, 40, created_at)

        saved = crud.create_assessment(test_session, assessment)
        test_session.commit()

        assert saved.id is not None
        fetched = crud.get_latest_by_user(test_session, sample_user.id)
        assert fetched is not None
        assert fetched.id == saved.id
        assert fetched.total_score == 40
        assert fetched.risk_level == 5
        assert fetched.risk_category == "เสี่ยงสูงมาก"
        assert fetched.answers == {"1": "ง", "4": ["ง"]}

    def test_latest_assessment_wins(
        self,
        crud: RiskAssessmentCRUD,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """With two assessments, the newest by created_at is returned."""
        older = make_assessment(sample_user.id, 10, datetime(2026, 4, 1, tzinfo=timezone.utc))
        newer = make_assessment(sample_user.id, 37, datetime(2026, 4, 9, tzinfo=timezone.utc))
        crud.create_assessment(test_session, older)
        crud.create_assessment(test_session, newer)
        test_session.commit()

        latest = crud.get_latest_by_user(test_session, sample_user.id)

        assert latest is not None
        assert latest.total_score == 37

    def test_get_latest_unknown_user_returns_none(
        self,
        crud: RiskAssessmentCRUD,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """A user without assessments yields None."""
        crud.create_assessment(
            test_session,
            make_assessment(sample_user.id, 20, datetime(2026, 4, 5, tzinfo=timezone.utc)),
        )
        test_session.commit()

        assert crud.get_latest_by_user(test_session, "no-such-user") is None
