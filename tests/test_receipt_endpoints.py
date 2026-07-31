"""Tests for FastAPI receipt-scan + confirm endpoints."""

# pylint: disable=redefined-outer-name

from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import asyncio

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from finance_ai.database.base import Base
from finance_ai.database.models.transaction import Transaction
from finance_ai.database.models.user import User
from finance_ai.main import app


def _mock_chat_model(json_content: str) -> Any:
    """Build a mock chat model returning JSON content."""
    mock = MagicMock(spec=BaseChatModel)
    mock.invoke.return_value = AIMessage(content=json_content)
    return mock


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


class TestUploadReceipt:
    """Tests for POST /upload/receipt (read-only draft extraction)."""

    def test_returns_drafts_without_writing(
        self,
        client: TestClient,
        temp_db: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A receipt image returns parsed drafts and writes nothing to the DB."""
        json_text = (
            '[{"transaction_type":"expense","amount":"350.00",'
            '"transaction_date":"2026-03-01","description":"ร้านอาหาร",'
            '"category":"food","confidence":0.9}]'
        )
        monkeypatch.setattr(
            "finance_ai.main.get_ocr_chat_model",
            lambda: _mock_chat_model(json_text),
        )
        response = client.post(
            "/upload/receipt",
            params={"user_id": "any-user"},
            files={"file": ("receipt.png", b"\x89PNGfake", "image/png")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert len(body["drafts"]) == 1
        assert body["drafts"][0]["transaction_type"] == "expense"
        assert body["drafts"][0]["amount"] == "350.00"

        session = temp_db()
        rows = list(session.execute(select(Transaction)).scalars())
        session.close()
        assert rows == []

    def test_empty_drafts_when_model_returns_garbage(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unreadable output returns an empty drafts list with 200."""
        monkeypatch.setattr(
            "finance_ai.main.get_ocr_chat_model",
            lambda: _mock_chat_model("cannot read"),
        )
        response = client.post(
            "/upload/receipt",
            params={"user_id": "any-user"},
            files={"file": ("r.png", b"\x89PNG", "image/png")},
        )
        assert response.status_code == 200
        assert response.json() == {"drafts": [], "total": 0}

    def test_misconfigured_ai_returns_503(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A provider/model error returns 503 with a clear Thai message."""

        def _raise() -> Any:
            raise ValueError("ocr_api_key is required")

        monkeypatch.setattr("finance_ai.main.get_ocr_chat_model", _raise)
        response = client.post(
            "/upload/receipt",
            params={"user_id": "any-user"},
            files={"file": ("receipt.png", b"\x89PNGfake", "image/png")},
        )
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert "AI ไม่สามารถอ่านเอกสารได้" in detail
        assert "ocr_api_key" in detail

    def test_unsupported_file_type_rejected(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-image/pdf upload is rejected with 400."""
        monkeypatch.setattr(
            "finance_ai.main.get_ocr_chat_model",
            lambda: _mock_chat_model("[]"),
        )
        response = client.post(
            "/upload/receipt",
            params={"user_id": "any-user"},
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 400
        assert "Unsupported" in response.json()["detail"]

    def test_unsupported_file_type_rejected_when_ai_broken(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unsupported type must still return 400 even if AI is broken."""

        def _raise() -> Any:
            raise RuntimeError("AI down")

        monkeypatch.setattr("finance_ai.main.get_ocr_chat_model", _raise)
        response = client.post(
            "/upload/receipt",
            params={"user_id": "any-user"},
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 400
        assert "Unsupported" in response.json()["detail"]

    def test_ocr_timeout_returns_504(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A slow OCR call that exceeds the timeout surfaces as a 504."""

        def _slow_extract(_content: bytes, _mime: str) -> Any:
            raise asyncio.TimeoutError()

        monkeypatch.setattr("finance_ai.main._extract_drafts", _slow_extract)
        monkeypatch.setattr(
            "finance_ai.main.get_ocr_chat_model",
            lambda: _mock_chat_model("[]"),
        )
        response = client.post(
            "/upload/receipt",
            params={"user_id": "any-user"},
            files={"file": ("r.png", b"\x89PNG", "image/png")},
        )
        assert response.status_code == 504
        assert "นานเกินไป" in response.json()["detail"]


class TestConfirmTransactions:
    """Tests for POST /transactions/confirm (persist user-confirmed drafts)."""

    def test_persists_confirmed_expense(
        self,
        client: TestClient,
        temp_db: sessionmaker[Session],
    ) -> None:
        """Confirming an expense writes one transaction row."""
        session = temp_db()
        user = session.execute(select(User)).scalar_one()
        session.close()

        response = client.post(
            "/transactions/confirm",
            json={
                "user_id": str(user.id),
                "transactions": [
                    {
                        "transaction_type": "expense",
                        "amount": "350.00",
                        "transaction_date": "2026-03-01",
                        "description": "ร้านอาหาร",
                        "category": "food",
                    }
                ],
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["inserted"] == 1
        assert body["total"] == 1

        session = temp_db()
        rows = list(session.execute(select(Transaction)).scalars())
        session.close()
        assert len(rows) == 1
        assert rows[0].amount == Decimal("350.00")

    def test_rejects_non_positive_amount(
        self,
        client: TestClient,
        temp_db: sessionmaker[Session],
    ) -> None:
        """A non-positive amount is rejected with 422."""
        session = temp_db()
        user = session.execute(select(User)).scalar_one()
        session.close()

        response = client.post(
            "/transactions/confirm",
            json={
                "user_id": str(user.id),
                "transactions": [
                    {
                        "transaction_type": "expense",
                        "amount": "0",
                        "transaction_date": "2026-03-01",
                        "description": "bad",
                        "category": "food",
                    }
                ],
            },
        )
        assert response.status_code == 422

    def test_rejects_invalid_category(
        self,
        client: TestClient,
        temp_db: sessionmaker[Session],
    ) -> None:
        """An out-of-set category is rejected with 422."""
        session = temp_db()
        user = session.execute(select(User)).scalar_one()
        session.close()

        response = client.post(
            "/transactions/confirm",
            json={
                "user_id": str(user.id),
                "transactions": [
                    {
                        "transaction_type": "expense",
                        "amount": "100.00",
                        "transaction_date": "2026-03-01",
                        "description": "x",
                        "category": "banana",
                    }
                ],
            },
        )
        assert response.status_code == 422
