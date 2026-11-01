"""In-memory store of LINE document imports waiting on the user.

An entry is either drafts waiting for "บันทึก"/"ยกเลิก", or a locked PDF
waiting for its password. Entries live in memory only (the app runs one
worker) and expire after a short time, so a PDF and its password are never
written to disk or the database.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock

from finance_ai.tools.receipt_ocr import ReceiptOcrResult

PENDING_IMPORT_TTL_SECONDS: float = 15 * 60


@dataclass
class PendingImport:
    """One import waiting on the user."""

    drafts: list[ReceiptOcrResult] = field(default_factory=list)
    locked_pdf: bytes | None = None
    created_at: float = 0.0

    def is_waiting_for_password(self) -> bool:
        """Return True when a locked PDF waits for its password.

        Example:
            >>> PendingImport(locked_pdf=b"%PDF").is_waiting_for_password()
            True
        """
        return self.locked_pdf is not None


class PendingImportStore:
    """Thread-safe per-LINE-user store with expiry."""

    def __init__(
        self,
        ttl_seconds: float = PENDING_IMPORT_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create an empty store.

        Args:
            ttl_seconds: How long an entry stays valid.
            clock: Monotonic clock (injectable for tests).
        """
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: dict[str, PendingImport] = {}
        self._lock = Lock()

    def put(self, line_user_id: str, pending: PendingImport) -> None:
        """Store an entry, replacing any earlier one for this user."""
        pending.created_at = self._clock()
        with self._lock:
            self._entries[line_user_id] = pending

    def get(self, line_user_id: str) -> PendingImport | None:
        """Return the user's live entry without removing it."""
        with self._lock:
            return self._live_entry(line_user_id)

    def pop(self, line_user_id: str) -> PendingImport | None:
        """Remove and return the user's live entry."""
        with self._lock:
            pending = self._live_entry(line_user_id)
            self._entries.pop(line_user_id, None)
            return pending

    def _live_entry(self, line_user_id: str) -> PendingImport | None:
        """Return the entry if not expired, dropping it when it is (lock held)."""
        pending = self._entries.get(line_user_id)
        if pending is None:
            return None
        if self._clock() - pending.created_at > self._ttl_seconds:
            del self._entries[line_user_id]
            return None
        return pending


pending_imports = PendingImportStore()
