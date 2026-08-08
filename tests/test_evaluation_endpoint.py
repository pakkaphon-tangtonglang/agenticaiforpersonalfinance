"""Tests for the /evaluation/run and /assets/notifications endpoints.

Covers two previously-broken endpoint behaviors:
- POST /evaluation/run must construct EvaluationRunner with the correct
  signature (vector_store, llm_provider, llm_model, db_session_factory) and
  translate the ``dimensions`` query into the runner's ``skip`` set.
- GET /assets/notifications must return the ``content`` column (the
  AssetNotification model has no ``message`` column; the old code crashed).
"""

# pylint: disable=redefined-outer-name

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from finance_ai.database.base import Base
from finance_ai.database.models.asset_notification import AssetNotification
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


def _make_report() -> Any:
    """Build a fake EvaluationReport with a model_dump returning a dict."""
    report = MagicMock()
    report.model_dump.return_value = {"report_id": "r-1"}
    return report


class TestRunEvaluation:
    """Tests for POST /evaluation/run."""

    def test_runs_all_dimensions_when_none(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No dimensions query runs every dimension (skip is empty/None)."""
        report = _make_report()
        runner = MagicMock()
        runner.run_all.return_value = report
        monkeypatch.setattr(
            "finance_ai.evaluation.runner.EvaluationRunner",
            lambda *args, **kwargs: runner,
        )

        response = client.post("/evaluation/run")

        assert response.status_code == 200
        runner.run_all.assert_called_once()
        _, kwargs = runner.run_all.call_args
        assert kwargs.get("skip") is None
        assert response.json() == {"status": "ok", "results": {"report_id": "r-1"}}

    def test_dimensions_become_skip_set(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Requested dimensions are kept; others go into the skip set."""
        report = _make_report()
        runner = MagicMock()
        runner.run_all.return_value = report
        monkeypatch.setattr(
            "finance_ai.evaluation.runner.EvaluationRunner",
            lambda *args, **kwargs: runner,
        )

        response = client.post(
            "/evaluation/run",
            params={"dimensions": ["routing", "accuracy"]},
        )

        assert response.status_code == 200
        _, kwargs = runner.run_all.call_args
        skip = kwargs["skip"]
        assert skip == {"rag", "hallucination", "quality", "performance"}

    def test_constructs_runner_with_required_args(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """EvaluationRunner is built with vector_store, provider, model, factory."""
        captured: dict[str, Any] = {}
        report = _make_report()
        runner = MagicMock()
        runner.run_all.return_value = report

        def _capture(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return runner

        monkeypatch.setattr(
            "finance_ai.evaluation.runner.EvaluationRunner",
            lambda *args, **kwargs: _capture(**kwargs),
        )

        response = client.post("/evaluation/run")

        assert response.status_code == 200
        assert captured["vector_store"] is None
        assert captured["llm_provider"] is not None
        assert captured["llm_model"] is not None
        assert captured["db_session_factory"] is not None


class TestAssetNotificationsField:
    """Tests for GET /assets/notifications (content column bug)."""

    def test_returns_content_not_message(
        self,
        client: TestClient,
        temp_db: sessionmaker[Session],
    ) -> None:
        """The endpoint returns the notification's content text without crashing."""
        session = temp_db()
        user = session.query(User).first()
        assert user is not None
        notification = AssetNotification(
            user_id=user.id,
            symbol="GC=F",
            content="ราคาทอง: 2,350 USD",
            is_read=False,
        )
        session.add(notification)
        session.commit()
        user_id = user.id
        session.close()

        response = client.get("/assets/notifications", params={"user_id": user_id})

        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["symbol"] == "GC=F"
        assert body[0]["message"] == "ราคาทอง: 2,350 USD"
