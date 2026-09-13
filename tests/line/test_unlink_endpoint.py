"""Tests for the POST /line/unlink endpoint (website-side unlink)."""

from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from finance_ai.database.models.line_user_mapping import LineUserMapping
from finance_ai.database.models.user import User
from finance_ai.line.mapping_service import get_or_create_line_mapping
from finance_ai.main import app

WEB_USER_ID = "00000000-de20-4000-8000-000000000001"


class TestLineUnlinkEndpoint:
    """Tests for POST /line/unlink."""

    def _client(self, test_engine: Any, monkeypatch: Any) -> TestClient:
        """TestClient wired to the in-memory engine."""
        monkeypatch.setattr("finance_ai.main._session_factory", sessionmaker(bind=test_engine))
        return TestClient(app)

    def test_unlinks_all_line_chats_of_web_user(self, test_engine: Any, monkeypatch: Any) -> None:
        """Every LINE chat linked to the web user is restored to LINE-only."""
        session_maker = sessionmaker(bind=test_engine)
        with session_maker() as session:
            session.add(
                User(
                    id=WEB_USER_ID,
                    email="web@finance-ai.local",
                    hashed_password="not-a-login",
                    full_name="ผู้ใช้เว็บ",
                )
            )
            session.commit()
            get_or_create_line_mapping(session, "Uline-user-1")
            get_or_create_line_mapping(session, "Uline-user-2")
            from finance_ai.line.link_command import link_line_user

            link_line_user(session, "Uline-user-1", WEB_USER_ID)
            link_line_user(session, "Uline-user-2", WEB_USER_ID)

        client = self._client(test_engine, monkeypatch)
        response = client.post("/line/unlink", json={"user_id": WEB_USER_ID})

        assert response.status_code == 200
        assert sorted(response.json()["unlinked"]) == ["Uline-user-1", "Uline-user-2"]
        with session_maker() as session:
            restored = session.query(LineUserMapping).filter_by(user_id=WEB_USER_ID).count()
            assert restored == 0

    def test_web_user_without_links_returns_empty(self, test_engine: Any, monkeypatch: Any) -> None:
        """Unlinking a user with no LINE chats returns an empty list."""
        client = self._client(test_engine, monkeypatch)
        response = client.post("/line/unlink", json={"user_id": WEB_USER_ID})
        assert response.status_code == 200
        assert response.json()["unlinked"] == []
