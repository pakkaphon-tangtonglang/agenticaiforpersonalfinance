"""The app-wide dependency creates the requesting user before any endpoint runs.

Every request that names a user (``user_id`` in the query string or the JSON
body) gets its user row first, in one place, so no endpoint can forget it.
Uses an FK-enforcing SQLite database so failures match Postgres.
"""

# pylint: disable=redefined-outer-name

from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from finance_ai.database.base import Base
from finance_ai.database.models.user import User
from finance_ai.main import app, provision_request_user


@pytest.fixture
def fk_db(monkeypatch: pytest.MonkeyPatch) -> sessionmaker[Session]:
    """An in-memory DB that enforces foreign keys like Postgres."""
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
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr("finance_ai.main._session_factory", factory)
    return factory


def _users(factory: sessionmaker[Session]) -> set[str]:
    """All user ids in the DB."""
    with factory() as session:
        return {user.id for user in session.query(User).all()}


class TestProvisionRequestUser:
    """One dependency covers every endpoint, current and future."""

    def test_registered_for_the_whole_app(self) -> None:
        """The dependency is attached to the app, not to individual routes."""
        assert any(dep.dependency is provision_request_user for dep in app.router.dependencies)

    @pytest.mark.usefixtures("fk_db")
    def test_user_from_query_string(self, fk_db: sessionmaker[Session]) -> None:
        """GET /dashboard?user_id=... creates the user first."""
        TestClient(app).get("/dashboard", params={"user_id": "q-user-0001"})

        assert "q-user-0001" in _users(fk_db)

    def test_user_from_json_body(
        self, fk_db: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """POST with user_id in the JSON body creates the user first."""
        monkeypatch.setattr("finance_ai.main.resolve_and_validate_symbol", lambda s: "PTT.BK")

        response = TestClient(app).post(
            "/assets/watchlist", json={"user_id": "b-user-0001", "symbol": "ptt"}
        )

        assert response.status_code == 200
        assert "b-user-0001" in _users(fk_db)

    def test_invalid_user_id_rejected_clearly(self, fk_db: sessionmaker[Session]) -> None:
        """An id the users table cannot hold gets a clear 422, not a 500."""
        response = TestClient(app).get("/dashboard", params={"user_id": "x" * 40})

        assert response.status_code == 422
        assert "user_id" in response.json()["detail"]
        assert _users(fk_db) == set()

    def test_requests_without_user_create_nothing(self, fk_db: sessionmaker[Session]) -> None:
        """/health and other user-less requests do not touch the users table."""
        TestClient(app).get("/health")

        assert _users(fk_db) == set()


class TestRecurringPostingOnRequest:
    """Opening the app posts the user's due recurring income and expenses."""

    def test_due_rent_is_posted_on_request(self, fk_db: sessionmaker[Session]) -> None:
        """A request after the due day creates this month's rent income."""
        from datetime import date  # pylint: disable=import-outside-toplevel
        from decimal import Decimal  # pylint: disable=import-outside-toplevel

        from finance_ai.database.models.transaction import (  # pylint: disable=import-outside-toplevel
            Transaction,
        )
        from finance_ai.tools.recurring_service import (  # pylint: disable=import-outside-toplevel
            create_recurring_entry,
        )

        with fk_db() as session:
            session.add(
                User(id="rent-user-01", email="r@x.com", hashed_password="x", full_name="R")
            )
            session.commit()
            create_recurring_entry(
                session, "rent-user-01", "income", "rental", Decimal("8000"), 1, "ค่าเช่า",
                date(2026, 1, 1),
            )  # fmt: skip
        TestClient(app).get("/dashboard", params={"user_id": "rent-user-01"})
        with fk_db() as session:
            assert session.query(Transaction).count() >= 1

    def test_posting_failure_does_not_block_request(
        self, fk_db: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A database error while posting is logged; the endpoint still answers."""
        from sqlalchemy.exc import OperationalError  # pylint: disable=import-outside-toplevel

        def _fail(*_args: Any) -> int:
            raise OperationalError("SELECT", {}, Exception("db down"))

        monkeypatch.setattr("finance_ai.main.post_due_entries", _fail)
        response = TestClient(app).get("/dashboard", params={"user_id": "fail-user-01"})
        assert response.status_code == 200
        assert "fail-user-01" in _users(fk_db)
