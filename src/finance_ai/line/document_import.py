"""Read a LINE image or PDF into transaction drafts and summarise them.

Images (receipts, transfer slips) go to the OCR vision model. PDFs are read
as bank statements first; a PDF that is not a statement (e.g. a receipt)
falls back to the OCR model. LINE has no review table, so the drafts are
shown as a short text summary the user confirms with a Quick Reply button.
"""

from collections.abc import Callable
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from finance_ai.tools.bank_statement_pdf import (
    StatementDraft,
    UnsupportedStatementError,
    drafts_from_pages,
    read_pdf_pages,
)
from finance_ai.tools.receipt_ocr import ReceiptOcrResult, extract_document_fields

SUMMARY_MAX_ROWS: int = 10
THAI_SHORT_MONTHS: tuple[str, ...] = (
    "ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
    "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค.",
)  # fmt: skip
CONFIRM_HINT: str = (
    'กด "บันทึก" เพื่อบันทึกทั้งหมด หรือ "ยกเลิก"\n' "(ถ้าต้องแก้ทีละรายการ อัปโหลดผ่านเว็บได้)"
)
_MAGIC_PREFIXES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF", "application/pdf"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG", "image/png"),
)


class UnsupportedDocumentError(ValueError):
    """The file is not an image or PDF we can read."""


class DocumentReadResult(BaseModel):
    """Drafts read from one document, plus own-account transfers left out."""

    drafts: list[ReceiptOcrResult]
    own_transfers_skipped: int = 0


def detect_document_mime(content: bytes) -> str:
    """Detect the file type from its first bytes.

    Args:
        content: Raw file bytes.

    Returns:
        "application/pdf", "image/jpeg", "image/png" or "image/webp".

    Raises:
        UnsupportedDocumentError: For any other file.

    Example:
        >>> detect_document_mime(b"%PDF-1.7")
        'application/pdf'
    """
    for prefix, mime_type in _MAGIC_PREFIXES:
        if content.startswith(prefix):
            return mime_type
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    raise UnsupportedDocumentError("รองรับเฉพาะรูปภาพ (JPG, PNG, WEBP) และไฟล์ PDF")


def read_document(
    content: bytes,
    password: str | None,
    ocr_model_provider: Callable[[], Any],
    chat_model_provider: Callable[[], Any],
) -> DocumentReadResult:
    """Read a receipt, slip or statement into drafts (nothing is saved).

    Args:
        content: Raw file bytes.
        password: PDF password, if the user gave one.
        ocr_model_provider: Returns the OCR vision model (called only when needed).
        chat_model_provider: Returns the chat model for the statement AI fallback.

    Returns:
        DocumentReadResult with the drafts.

    Raises:
        UnsupportedDocumentError: Not an image or PDF.
        PdfPasswordRequiredError / PdfPasswordIncorrectError: Locked PDF.
        StatementTotalsMismatchError: Statement rows do not add up.

    Example:
        >>> read_document(jpeg_bytes, None, get_ocr_chat_model, get_chat_model)
    """
    mime_type = detect_document_mime(content)
    if mime_type == "application/pdf":
        try:
            return _read_statement(content, password, chat_model_provider)
        except UnsupportedStatementError:
            pass
    drafts = extract_document_fields(content, mime_type, ocr_model_provider())
    return DocumentReadResult(drafts=drafts)


def _read_statement(
    content: bytes, password: str | None, chat_model_provider: Callable[[], Any]
) -> DocumentReadResult:
    """Read a statement PDF, leaving out transfers between the user's own accounts."""
    rows = drafts_from_pages(read_pdf_pages(content, password), chat_model_provider())
    kept = [_statement_to_draft(row) for row in rows if not row.own_transfer]
    return DocumentReadResult(drafts=kept, own_transfers_skipped=len(rows) - len(kept))


def _statement_to_draft(row: StatementDraft) -> ReceiptOcrResult:
    """Convert a statement row into the draft type the confirm step saves."""
    return ReceiptOcrResult(
        transaction_type=row.transaction_type,
        amount=row.amount,
        transaction_date=row.transaction_date,
        description=row.description,
        category=row.category,
        income_type=row.income_type,
    )


def format_import_summary(result: DocumentReadResult) -> str:
    """Write the LINE summary the user confirms (count, totals, first rows).

    Args:
        result: Drafts read from the document.

    Returns:
        Thai plain text for a LINE message.

    Example:
        >>> print(format_import_summary(result))
        📄 อ่านได้ 2 รายการ ...
    """
    drafts = result.drafts
    lines = [f"📄 อ่านได้ {len(drafts)} รายการ", _totals_line(drafts), ""]
    lines += [
        f"{index}. {_row_text(draft)}" for index, draft in enumerate(drafts[:SUMMARY_MAX_ROWS], 1)
    ]
    if len(drafts) > SUMMARY_MAX_ROWS:
        lines.append(f"...และอีก {len(drafts) - SUMMARY_MAX_ROWS} รายการ")
    if result.own_transfers_skipped:
        lines.append(f"(ไม่รวมรายการโอนเข้าบัญชีตัวเอง {result.own_transfers_skipped} รายการ)")
    return "\n".join([*lines, "", CONFIRM_HINT])


def _totals_line(drafts: list[ReceiptOcrResult]) -> str:
    """Return 'รายจ่าย X บาท · รายรับ Y บาท' (only the types present)."""
    parts = []
    for kind, label in (("expense", "รายจ่าย"), ("income", "รายรับ")):
        amounts = [draft.amount for draft in drafts if draft.transaction_type == kind]
        if amounts:
            parts.append(f"{label} {_baht(sum(amounts, Decimal('0')))} บาท")
    return " · ".join(parts)


def _row_text(draft: ReceiptOcrResult) -> str:
    """Return '1 พ.ย. · กาแฟ · -80 บาท'."""
    when = f"{draft.transaction_date.day} {THAI_SHORT_MONTHS[draft.transaction_date.month - 1]}"
    sign = "+" if draft.transaction_type == "income" else "-"
    return f"{when} · {draft.description} · {sign}{_baht(draft.amount)} บาท"


def _baht(amount: Decimal) -> str:
    """Format money with thousands separators; satang only when present."""
    text = f"{amount:,.2f}"
    return text[:-3] if text.endswith(".00") else text
