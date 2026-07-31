"""Persistence service for confirmed OCR document drafts.

The only write path for the document-scan flow. Converts confirmed
ReceiptOcrResult drafts into ParsedTransaction objects and reuses the
existing bulk_insert_transactions() helper, which handles duplicate
detection and dual-writes income records to the incomes table.

Nothing is persisted until the user confirms — the OCR extraction step
is read-only.
"""

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.orm import Session

from finance_ai.tools.bank_statement_parser import ParsedTransaction
from finance_ai.tools.bank_statement_service import bulk_insert_transactions
from finance_ai.tools.receipt_ocr import ReceiptOcrResult


@dataclass
class ConfirmResult:
    """Outcome of confirming a batch of drafts.

    Attributes:
        inserted: Number of transactions actually inserted.
        total: Total drafts processed.
        skipped: Number of drafts skipped as duplicates.
    """

    inserted: int
    total: int
    skipped: int


def to_parsed_transaction(draft: ReceiptOcrResult) -> ParsedTransaction:
    """Convert a confirmed OCR draft into a ParsedTransaction.

    Args:
        draft: Confirmed ReceiptOcrResult from the user.

    Returns:
        ParsedTransaction ready for bulk_insert_transactions.

    Example:
        >>> to_parsed_transaction(draft).transaction_type
        'expense'
    """
    return ParsedTransaction(
        transaction_date=draft.transaction_date,
        description=draft.description,
        amount=draft.amount,
        category=draft.category,
        transaction_type=draft.transaction_type,
    )


def confirm_receipt_transactions(
    drafts: list[ReceiptOcrResult],
    user_id: str,
    db_session_factory: Callable[[], Session],
) -> ConfirmResult:
    """Persist confirmed drafts to the database.

    Args:
        drafts: User-confirmed OCR drafts.
        user_id: UUID of the user.
        db_session_factory: Session factory callable.

    Returns:
        ConfirmResult with inserted, total, and skipped counts.

    Example:
        >>> result = confirm_receipt_transactions(drafts, "uid", factory)
        >>> result.inserted
        3
    """
    if not drafts:
        return ConfirmResult(inserted=0, total=0, skipped=0)
    parsed = [to_parsed_transaction(d) for d in drafts]
    inserted = bulk_insert_transactions(parsed, user_id, db_session_factory)
    return ConfirmResult(
        inserted=inserted,
        total=len(drafts),
        skipped=len(drafts) - inserted,
    )
