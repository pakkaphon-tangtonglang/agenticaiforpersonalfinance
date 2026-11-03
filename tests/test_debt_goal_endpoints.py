"""Tests for the dashboard debt and goal endpoints (/debts, /goals)."""

# pylint: disable=redefined-outer-name

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import finance_ai.main as main_module
from finance_ai.database.base import Base
from finance_ai.database.models.user import User
from finance_ai.main import app


@pytest.fixture
def client_and_user(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, str]]:
    """A client on an in-memory DB with one user."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(main_module, "_session_factory", factory)
    with factory() as session:
        user = User(email="d@x.com", hashed_password="x", full_name="D")
        session.add(user)
        session.commit()
        user_id = user.id
    yield TestClient(app), user_id
    engine.dispose()


def _debt(user_id: str, **changes: Any) -> dict[str, Any]:
    body = {
        "user_id": user_id, "creditor": "เพื่อน", "debt_type": "personal",
        "original_amount": "2000", "remaining_balance": "2000",
    }  # fmt: skip
    return body | changes


def _goal(user_id: str, **changes: Any) -> dict[str, Any]:
    body = {
        "user_id": user_id, "name": "เที่ยวญี่ปุ่น", "goal_type": "travel",
        "target_amount": "60000", "current_amount": "15000",
    }  # fmt: skip
    return body | changes


class TestDebtEndpoints:
    """Add, list, edit and delete debts."""

    def test_full_cycle(self, client_and_user: tuple[TestClient, str]) -> None:
        """POST → GET → PATCH → DELETE."""
        client, user_id = client_and_user
        created = client.post("/debts", json=_debt(user_id))
        assert created.status_code == 201
        debt_id = created.json()["debt_id"]

        edited = client.patch(f"/debts/{debt_id}", json=_debt(user_id, remaining_balance="1500"))
        assert edited.json()["paid_percentage"] == "25.00"
        listing = client.get("/debts", params={"user_id": user_id}).json()
        assert listing["total_remaining"] == "1500.00"

        assert client.delete(f"/debts/{debt_id}", params={"user_id": user_id}).status_code == 204
        assert client.get("/debts", params={"user_id": user_id}).json()["debts"] == []

    def test_invalid_form_is_422(self, client_and_user: tuple[TestClient, str]) -> None:
        """Balance above the original amount is refused."""
        client, user_id = client_and_user
        response = client.post("/debts", json=_debt(user_id, remaining_balance="9000"))
        assert response.status_code == 422

    def test_other_user_is_404(self, client_and_user: tuple[TestClient, str]) -> None:
        """Another user's id cannot edit or delete."""
        client, user_id = client_and_user
        debt_id = client.post("/debts", json=_debt(user_id)).json()["debt_id"]
        edit = client.patch(f"/debts/{debt_id}", json=_debt("someone-else"))
        delete = client.delete(f"/debts/{debt_id}", params={"user_id": "someone-else"})
        assert (edit.status_code, delete.status_code) == (404, 404)


class TestGoalEndpoints:
    """Add, list, edit and delete goals."""

    def test_full_cycle(self, client_and_user: tuple[TestClient, str]) -> None:
        """POST → GET → PATCH → DELETE."""
        client, user_id = client_and_user
        goal_id = client.post("/goals", json=_goal(user_id)).json()["goal_id"]

        edited = client.patch(f"/goals/{goal_id}", json=_goal(user_id, current_amount="30000"))
        assert edited.json()["percentage"] == "50.00"
        [goal] = client.get("/goals", params={"user_id": user_id}).json()
        assert goal["goal_type_label"] == "ท่องเที่ยว"

        assert client.delete(f"/goals/{goal_id}", params={"user_id": user_id}).status_code == 204
        assert client.get("/goals", params={"user_id": user_id}).json() == []

    def test_other_user_is_404(self, client_and_user: tuple[TestClient, str]) -> None:
        """Another user's id cannot edit or delete."""
        client, user_id = client_and_user
        goal_id = client.post("/goals", json=_goal(user_id)).json()["goal_id"]
        edit = client.patch(f"/goals/{goal_id}", json=_goal("someone-else"))
        delete = client.delete(f"/goals/{goal_id}", params={"user_id": "someone-else"})
        assert (edit.status_code, delete.status_code) == (404, 404)
