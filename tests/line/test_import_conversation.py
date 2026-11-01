"""Tests for the LINE document import conversation (confirm / password)."""

from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.database.models.transaction import Transaction
from finance_ai.line.document_import import DocumentReadResult, UnsupportedDocumentError
from finance_ai.line.import_conversation import (
    CANCEL_LABEL,
    CONFIRM_LABEL,
    continue_pending_import,
    start_document_import,
)
from finance_ai.line.pending_imports import PendingImport, PendingImportStore
from finance_ai.tools.bank_statement_pdf import (
    PdfPasswordIncorrectError,
    PdfPasswordRequiredError,
    StatementTotalsMismatchError,
)
from finance_ai.tools.receipt_ocr import ReceiptOcrResult

MODULE = "finance_ai.line.import_conversation"
USER = "Uline-import"
PDF = b"%PDF-1.7"


def _draft() -> ReceiptOcrResult:
    return ReceiptOcrResult(
        transaction_type="expense",
        amount=Decimal("80"),
        transaction_date=date(2026, 10, 30),
        description="กาแฟ",
        category="food",
    )


@pytest.fixture(name="store")
def fixture_store(monkeypatch: Any) -> PendingImportStore:
    """A fresh pending store patched into the module."""
    store = PendingImportStore()
    monkeypatch.setattr(f"{MODULE}.pending_imports", store)
    return store


def _patch_read(monkeypatch: Any, **kwargs: Any) -> MagicMock:
    reader = MagicMock(**kwargs)
    monkeypatch.setattr(f"{MODULE}.read_document", reader)
    return reader


def _start() -> Any:
    return start_document_import(USER, PDF, lambda: "ocr", lambda: "chat")


class TestStartDocumentImport:
    """Reading a new document."""

    def test_drafts_are_kept_and_summarised(
        self, monkeypatch: Any, store: PendingImportStore
    ) -> None:
        """Drafts wait in the store; the reply has the summary and confirm buttons."""
        _patch_read(monkeypatch, return_value=DocumentReadResult(drafts=[_draft()]))
        reply = _start()
        assert "อ่านได้ 1 รายการ" in reply.text
        assert reply.quick_reply_labels == (CONFIRM_LABEL, CANCEL_LABEL)
        pending = store.get(USER)
        assert pending is not None and len(pending.drafts) == 1

    def test_locked_pdf_asks_for_password(
        self, monkeypatch: Any, store: PendingImportStore
    ) -> None:
        """A locked PDF is held in memory and the user is asked for the password."""
        _patch_read(monkeypatch, side_effect=PdfPasswordRequiredError("locked"))
        reply = _start()
        assert "รหัสผ่าน" in reply.text and "ไม่บันทึก" in reply.text
        assert reply.quick_reply_labels == (CANCEL_LABEL,)
        pending = store.get(USER)
        assert pending is not None and pending.locked_pdf == PDF

    def test_nothing_found(self, monkeypatch: Any, store: PendingImportStore) -> None:
        """No drafts: a hint, and nothing waits."""
        _patch_read(monkeypatch, return_value=DocumentReadResult(drafts=[]))
        assert "ไม่พบรายการ" in _start().text
        assert store.get(USER) is None

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (UnsupportedDocumentError("รองรับเฉพาะรูปภาพ"), "รองรับเฉพาะรูปภาพ"),
            (StatementTotalsMismatchError("x"), "ยอดรวม"),
        ],
    )
    def test_read_problems_explained(
        self, monkeypatch: Any, store: PendingImportStore, error: Exception, expected: str
    ) -> None:
        """Known read problems become a Thai explanation."""
        _patch_read(monkeypatch, side_effect=error)
        assert expected in _start().text
        assert store.get(USER) is None


class TestContinuePendingImport:
    """Text messages while an import waits."""

    def _continue(self, text: str, factory: Callable[[], Session]) -> Any:
        return continue_pending_import(USER, text, factory, lambda: "ocr", lambda: "chat")

    def test_no_pending_returns_none(
        self, store: PendingImportStore, db_session_factory: Callable[[], Session]
    ) -> None:
        """Without a pending import the agent handles the text."""
        assert self._continue("บันทึก", db_session_factory) is None

    def test_confirm_saves_drafts(
        self, store: PendingImportStore, db_session_factory: Callable[[], Session]
    ) -> None:
        """'บันทึก' saves every draft for the mapped user and clears the store."""
        store.put(USER, PendingImport(drafts=[_draft()]))
        reply = self._continue(CONFIRM_LABEL, db_session_factory)
        assert reply is not None and "บันทึกแล้ว 1 รายการ" in reply.text
        assert store.get(USER) is None
        with db_session_factory() as session:
            saved = session.scalars(select(Transaction)).all()
        assert [row.amount for row in saved] == [Decimal("80.00")]

    def test_cancel_drops_drafts(
        self, store: PendingImportStore, db_session_factory: Callable[[], Session]
    ) -> None:
        """'ยกเลิก' clears the store and saves nothing."""
        store.put(USER, PendingImport(drafts=[_draft()]))
        reply = self._continue(CANCEL_LABEL, db_session_factory)
        assert reply is not None and "ยกเลิก" in reply.text
        assert store.get(USER) is None

    def test_other_text_leaves_drafts_waiting(
        self, store: PendingImportStore, db_session_factory: Callable[[], Session]
    ) -> None:
        """A normal question goes to the agent; the drafts keep waiting."""
        store.put(USER, PendingImport(drafts=[_draft()]))
        assert self._continue("ภาษีของฉันเท่าไหร่", db_session_factory) is None
        assert store.get(USER) is not None

    def test_password_reads_the_pdf(
        self,
        monkeypatch: Any,
        store: PendingImportStore,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """The next text is used as the password; the drafts then wait for confirmation."""
        store.put(USER, PendingImport(locked_pdf=PDF))
        reader = _patch_read(monkeypatch, return_value=DocumentReadResult(drafts=[_draft()]))
        reply = self._continue("19900101", db_session_factory)
        assert reply is not None and "อ่านได้ 1 รายการ" in reply.text
        reader.assert_called_once()
        assert reader.call_args.args[:2] == (PDF, "19900101")
        pending = store.get(USER)
        assert pending is not None and not pending.is_waiting_for_password()

    def test_wrong_password(
        self,
        monkeypatch: Any,
        store: PendingImportStore,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """A wrong password drops the PDF and asks for it again."""
        store.put(USER, PendingImport(locked_pdf=PDF))
        _patch_read(monkeypatch, side_effect=PdfPasswordIncorrectError("bad"))
        reply = self._continue("wrong", db_session_factory)
        assert reply is not None and "รหัสผ่านไม่ถูกต้อง" in reply.text
        assert store.get(USER) is None

    def test_cancel_password_prompt(
        self,
        monkeypatch: Any,
        store: PendingImportStore,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """'ยกเลิก' at the password prompt drops the PDF without trying it."""
        store.put(USER, PendingImport(locked_pdf=PDF))
        reader = _patch_read(monkeypatch)
        reply = self._continue(CANCEL_LABEL, db_session_factory)
        assert reply is not None and "ยกเลิก" in reply.text
        reader.assert_not_called()
        assert store.get(USER) is None
