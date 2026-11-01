"""API tests for the dashboard income/expense history endpoints."""

# pylint: disable=redefined-outer-name

from datetime import date, timedelta
from typing import Any

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from finance_ai.database.base import Base
from finance_ai.main import app

USER = "history-user-0001"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient on an FK-enforcing in-memory DB (behaves like Postgres)."""
    engine: Engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _enable_fk_constraints(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr("finance_ai.main._session_factory", sessionmaker(bind=engine))
    return TestClient(app)


def _body(**overrides: Any) -> dict[str, Any]:
    """A valid create/edit body."""
    values = {
        "user_id": USER,
        "transaction_type": "expense",
        "transaction_date": "2026-10-15",
        "amount": "120.00",
        "category": "food",
        "description": "ข้าวกลางวัน",
    }
    return {**values, **overrides}


def _list(client: TestClient, **params: Any) -> dict[str, Any]:
    """GET /transactions for October 2026."""
    query = {"user_id": USER, "year": 2026, "month": 10, **params}
    response = client.get("/transactions", params=query)
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    return data


class TestTransactionHistoryApi:
    """List, add, edit and delete through the API."""

    def test_add_then_list(self, client: TestClient) -> None:
        """A created entry appears in its month with Thai label and totals."""
        created = client.post("/transactions", json=_body())

        assert created.status_code == 201
        page = _list(client)
        assert [e["description"] for e in page["entries"]] == ["ข้าวกลางวัน"]
        assert page["entries"][0]["category_label"] == "อาหาร"
        assert page["total_expense"] == "120.00"

    def test_edit_entry(self, client: TestClient) -> None:
        """PATCH replaces the entry's values."""
        entry_id = client.post("/transactions", json=_body()).json()["id"]

        edited = client.patch(f"/transactions/{entry_id}", json=_body(amount="99.50"))

        assert edited.status_code == 200
        assert _list(client)["entries"][0]["amount"] == "99.50"

    def test_delete_entry(self, client: TestClient) -> None:
        """DELETE removes the entry."""
        entry_id = client.post("/transactions", json=_body()).json()["id"]

        deleted = client.delete(f"/transactions/{entry_id}", params={"user_id": USER})

        assert deleted.status_code == 204
        assert _list(client)["entries"] == []

    def test_filter_by_type(self, client: TestClient) -> None:
        """type=income shows only income."""
        client.post("/transactions", json=_body())
        client.post("/transactions", json=_body(transaction_type="income", category="salary"))

        page = _list(client, type="income")

        assert [e["transaction_type"] for e in page["entries"]] == ["income"]

    def test_future_date_rejected_with_reason(self, client: TestClient) -> None:
        """Future entries get a 422 that says why."""
        tomorrow = (date.today() + timedelta(days=1)).isoformat()

        response = client.post("/transactions", json=_body(transaction_date=tomorrow))

        assert response.status_code == 422
        assert "future" in str(response.json())

    def test_other_users_entry_is_404(self, client: TestClient) -> None:
        """Another user cannot edit or delete the entry."""
        entry_id = client.post("/transactions", json=_body()).json()["id"]

        assert (
            client.delete(
                f"/transactions/{entry_id}", params={"user_id": "someone-else"}
            ).status_code
            == 404
        )
        assert (
            client.patch(
                f"/transactions/{entry_id}", json=_body(user_id="someone-else")
            ).status_code
            == 404
        )

    def test_invalid_month_rejected(self, client: TestClient) -> None:
        """month must be 1-12."""
        response = client.get("/transactions", params={"user_id": USER, "year": 2026, "month": 13})

        assert response.status_code == 422


@pytest.mark.usefixtures("client")
def test_routes_registered() -> None:
    """All four history routes exist."""
    paths = {
        (method, r.path)
        for r in app.routes
        if isinstance(r, APIRoute)
        for method in r.methods or ()
    }
    assert ("GET", "/transactions") in paths
    assert ("POST", "/transactions") in paths
    assert ("PATCH", "/transactions/{entry_id}") in paths
    assert ("DELETE", "/transactions/{entry_id}") in paths
