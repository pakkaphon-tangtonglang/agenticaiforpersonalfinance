"""Tests for the risk-assessment endpoints (submit + latest).

Covers the SEC suitability questionnaire flow over HTTP:
- POST /risk-assessment/submit validates, scores, and persists answers.
- GET /risk-assessment/latest returns the stored result or None.
"""

# pylint: disable=unsubscriptable-object
# ^ false positive: sessionmaker IS subscriptable (SQLAlchemy 2.x generic);
#   astroid cannot infer it, unlike mypy.

# pylint: disable=redefined-outer-name

from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from finance_ai.database.base import Base
from finance_ai.database.models.user import User
from finance_ai.main import app


@pytest.fixture
def client() -> TestClient:
    """Return a TestClient for the app."""
    return TestClient(app)


@pytest.fixture
def temp_db(monkeypatch: pytest.MonkeyPatch) -> sessionmaker[Session]:
    """Create an in-memory DB, patch main._session_factory, and add a user.

    Returns:
        sessionmaker bound to the in-memory engine (with schema + one user).
    """
    engine: Engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine)

    session = factory()
    user = User(
        email="t@example.com",
        hashed_password="x",
        full_name="T",
        tax_id="1234567890123",
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    session.close()

    monkeypatch.setattr("finance_ai.main._session_factory", factory)
    return factory


def _min_score_answers() -> dict[str, Any]:
    """All-ก answers (minimum total score 10)."""
    answers: dict[str, Any] = {str(question_id): "ก" for question_id in range(1, 11)}
    answers["4"] = ["ก"]
    return answers


def _seed_user_id(factory: sessionmaker[Session]) -> str:
    """Return the id of the single user created by the temp_db fixture."""
    session = factory()
    user = session.query(User).first()
    assert user is not None
    user_id = user.id
    session.close()
    return user_id


class TestSubmitRiskAssessment:
    """Tests for POST /risk-assessment/submit."""

    def test_submit_happy_path(self, client: TestClient, temp_db: sessionmaker[Session]) -> None:
        """All-ก answers score 10 -> level 1 เสี่ยงต่ำ with allocation payload."""
        user_id = _seed_user_id(temp_db)

        response = client.post(
            "/risk-assessment/submit",
            json={"user_id": user_id, "answers": _min_score_answers()},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["total_score"] == 10
        assert body["risk_level"] == 1
        assert body["risk_category"] == "เสี่ยงต่ำ"
        allocation = body["allocation"]
        assert len(allocation["columns"]) == 5
        assert len(allocation["row"]) == 5
        assert allocation["footnote"]

    def test_missing_scored_answer_returns_422(
        self, client: TestClient, temp_db: sessionmaker[Session]
    ) -> None:
        """A missing scored question is rejected with 422 naming the question."""
        user_id = _seed_user_id(temp_db)
        answers = _min_score_answers()
        del answers["3"]

        response = client.post(
            "/risk-assessment/submit",
            json={"user_id": user_id, "answers": answers},
        )

        assert response.status_code == 422
        assert "3" in response.json()["detail"]

    def test_invalid_choice_returns_422(
        self, client: TestClient, temp_db: sessionmaker[Session]
    ) -> None:
        """An unknown choice key is rejected with 422."""
        user_id = _seed_user_id(temp_db)
        answers = _min_score_answers()
        answers["1"] = "ฮ"

        response = client.post(
            "/risk-assessment/submit",
            json={"user_id": user_id, "answers": answers},
        )

        assert response.status_code == 422


class TestLatestRiskAssessment:
    """Tests for GET /risk-assessment/latest."""

    def test_latest_after_submit(self, client: TestClient, temp_db: sessionmaker[Session]) -> None:
        """The latest endpoint returns the assessment stored by submit."""
        user_id = _seed_user_id(temp_db)
        client.post(
            "/risk-assessment/submit",
            json={"user_id": user_id, "answers": _min_score_answers()},
        )

        response = client.get("/risk-assessment/latest", params={"user_id": user_id})

        assert response.status_code == 200
        assessment = response.json()["assessment"]
        assert assessment is not None
        assert assessment["total_score"] == 10
        assert assessment["risk_level"] == 1
        assert assessment["risk_category"] == "เสี่ยงต่ำ"
        assert assessment["created_at"]

    @pytest.mark.usefixtures("temp_db")
    def test_latest_unknown_user_returns_none(self, client: TestClient) -> None:
        """A user with no assessment gets assessment: None."""
        response = client.get("/risk-assessment/latest", params={"user_id": "no-such-user"})

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "assessment": None}
