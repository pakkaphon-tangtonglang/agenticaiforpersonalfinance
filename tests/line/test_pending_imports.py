"""Tests for the in-memory store of LINE imports waiting on the user."""

from datetime import date
from decimal import Decimal

from finance_ai.line.pending_imports import PendingImport, PendingImportStore
from finance_ai.tools.receipt_ocr import ReceiptOcrResult

USER = "Uline-1"


def _draft() -> ReceiptOcrResult:
    return ReceiptOcrResult(
        transaction_type="expense",
        amount=Decimal("80"),
        transaction_date=date(2026, 11, 1),
        description="กาแฟ",
    )


class FakeClock:
    """Controllable monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TestPendingImportStore:
    """Tests for PendingImportStore."""

    def test_put_drafts_then_get(self) -> None:
        """Stored drafts come back for the same user only."""
        store = PendingImportStore()
        store.put(USER, PendingImport(drafts=[_draft()]))
        pending = store.get(USER)
        assert pending is not None and len(pending.drafts) == 1
        assert store.get("Uother") is None

    def test_put_replaces_previous(self) -> None:
        """A new document replaces whatever was waiting."""
        store = PendingImportStore()
        store.put(USER, PendingImport(drafts=[_draft()]))
        store.put(USER, PendingImport(locked_pdf=b"%PDF"))
        pending = store.get(USER)
        assert pending is not None and pending.is_waiting_for_password()

    def test_pop_removes(self) -> None:
        """pop returns the entry once, then nothing."""
        store = PendingImportStore()
        store.put(USER, PendingImport(drafts=[_draft()]))
        assert store.pop(USER) is not None
        assert store.pop(USER) is None

    def test_entries_expire(self) -> None:
        """Entries older than the time limit are gone."""
        clock = FakeClock()
        store = PendingImportStore(ttl_seconds=60, clock=clock)
        store.put(USER, PendingImport(drafts=[_draft()]))
        clock.now += 61
        assert store.get(USER) is None
        assert store.pop(USER) is None

    def test_drafts_entry_is_not_waiting_for_password(self) -> None:
        """Drafts waiting for confirmation are not a password prompt."""
        assert PendingImport(drafts=[_draft()]).is_waiting_for_password() is False
