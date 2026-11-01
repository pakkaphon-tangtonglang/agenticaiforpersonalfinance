"""Writes from the web UI must work for a user who has never chatted.

Production runs on Postgres, which enforces foreign keys: a transaction or
watchlist row for an unknown ``user_id`` fails unless the user row is
created first. A live check on Render found the CSV upload returning 500
for exactly this reason. These tests use an FK-enforcing SQLite database
so they fail the same way production does.
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
from finance_ai.main import app

NEW_USER = "brand-new-user-0001"


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


def _user_exists(factory: sessionmaker[Session], user_id: str) -> bool:
    """True when the user row exists."""
    with factory() as session:
        return session.get(User, user_id) is not None


class TestNewUserWrites:
    """Each write endpoint creates the user row before inserting."""

    def test_bank_statement_upload(self, fk_db: sessionmaker[Session]) -> None:
        """A CSV upload by a brand-new user imports its rows."""
        csv = "date,description,amount\n2026-10-30,coffee,65.00\n2026-10-31,lunch,120.00\n"

        response = TestClient(app).post(
            "/upload/bank-statement",
            params={"user_id": NEW_USER},
            files={"file": ("statement.csv", csv, "text/csv")},
        )

        assert response.status_code == 200
        assert response.json() == {"inserted": 2, "total": 2}
        assert _user_exists(fk_db, NEW_USER)

    def test_confirm_receipt_transactions(self, fk_db: sessionmaker[Session]) -> None:
        """Confirming scanned receipt items works for a brand-new user."""
        payload = {
            "user_id": NEW_USER,
            "transactions": [
                {
                    "transaction_type": "expense",
                    "amount": "150.00",
                    "transaction_date": "2026-10-31",
                    "description": "cafe",
                    "category": "food",
                }
            ],
        }

        response = TestClient(app).post("/transactions/confirm", json=payload)

        assert response.status_code == 200
        assert response.json()["inserted"] == 1
        assert _user_exists(fk_db, NEW_USER)

    def test_watchlist_add(
        self, fk_db: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Adding to the watchlist works for a brand-new user."""
        monkeypatch.setattr("finance_ai.main.resolve_and_validate_symbol", lambda symbol: "PTT.BK")

        response = TestClient(app).post(
            "/assets/watchlist", json={"user_id": NEW_USER, "symbol": "ptt", "name": "PTT"}
        )

        assert response.status_code == 200
        assert response.json()["symbol"] == "PTT.BK"
        assert _user_exists(fk_db, NEW_USER)
