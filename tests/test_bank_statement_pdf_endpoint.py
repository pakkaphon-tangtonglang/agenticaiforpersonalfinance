"""API tests for importing a bank statement PDF (drafts for review, nothing saved)."""

# pylint: disable=redefined-outer-name

from typing import Any

import pytest
from fastapi.testclient import TestClient

from finance_ai.main import app
from finance_ai.tools.bank_statement_pdf import (
    PdfPasswordIncorrectError,
    PdfPasswordRequiredError,
)
from tests.tools.test_bank_statement_pdf import PAGE

ENDPOINT = "/upload/bank-statement/pdf"


def _post(client: TestClient, password: str = "", content: bytes = b"%PDF-fake") -> Any:
    """Upload a (fake) PDF to the statement endpoint."""
    data = {"password": password} if password else {}
    return client.post(
        ENDPOINT,
        params={"user_id": "pdf-user-0001"},
        files={"file": ("statement.pdf", content, "application/pdf")},
        data=data,
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A client whose PDF reader returns one fake statement page."""
    monkeypatch.setattr("finance_ai.main.read_pdf_pages", lambda content, password: [PAGE])
    monkeypatch.setattr("finance_ai.main.get_chat_model", lambda: None)
    return TestClient(app)


class TestStatementPdfEndpoint:
    """Upload → review drafts, with clear codes for password and layout problems."""

    def test_returns_review_drafts_with_own_transfer_flag(self, client: TestClient) -> None:
        """Rows come back as drafts; transfers to the holder are flagged."""
        response = _post(client)

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 4
        assert body["own_transfers"] == 1
        own = [d for d in body["drafts"] if d["own_transfer"]]
        assert own[0]["description"] == "PromptPay x2222 สมชาย ทดสอบดี"
        assert body["drafts"][0]["amount"] == "65.00"

    def test_password_required_code(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A locked PDF answers 422 with code pdf_password_required."""

        def _locked(content: bytes, password: str | None) -> list[str]:
            raise PdfPasswordRequiredError("PDF นี้มีรหัสผ่าน")

        monkeypatch.setattr("finance_ai.main.read_pdf_pages", _locked)

        response = _post(client)

        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "pdf_password_required"

    def test_wrong_password_code(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """A wrong password answers 422 with code pdf_password_incorrect."""

        def _wrong(content: bytes, password: str | None) -> list[str]:
            raise PdfPasswordIncorrectError("รหัสผ่าน PDF ไม่ถูกต้อง")

        monkeypatch.setattr("finance_ai.main.read_pdf_pages", _wrong)

        response = _post(client, password="nope")

        assert response.json()["detail"]["code"] == "pdf_password_incorrect"

    def test_password_is_passed_to_reader(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The password from the form reaches the PDF reader."""
        seen: dict[str, Any] = {}

        def _reader(_content: bytes, password: str | None) -> list[str]:
            seen["password"] = password
            return [PAGE]

        monkeypatch.setattr("finance_ai.main.read_pdf_pages", _reader)

        _post(client, password="01011990")

        assert seen["password"] == "01011990"

    def test_totals_mismatch_code(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rows that disagree with the bank's totals stop the import."""
        broken = PAGE.replace("-120.50 +2,314.50", "-12.50 +2,314.50")
        monkeypatch.setattr("finance_ai.main.read_pdf_pages", lambda content, password: [broken])

        response = _post(client)

        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "statement_totals_mismatch"

    def test_too_large_file_rejected(self, client: TestClient) -> None:
        """Files over the size limit are refused before reading."""
        response = _post(client, content=b"0" * (10 * 1024 * 1024 + 1))

        assert response.status_code == 413

    def test_nothing_is_saved(self, client: TestClient) -> None:
        """The endpoint only returns drafts; the review step saves."""
        _post(client)

        page = client.get(
            "/transactions", params={"user_id": "pdf-user-0001", "year": 2026, "month": 9}
        ).json()
        assert page["entries"] == []
