"""Tests for the /assets/notifications endpoint.

Covers a previously-broken endpoint behavior:
- GET /assets/notifications must return the ``content`` column (the
  AssetNotification model has no ``message`` column; the old code crashed).
"""

# pylint: disable=redefined-outer-name

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
