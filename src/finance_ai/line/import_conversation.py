"""LINE document import as a short conversation.

1. The user sends a receipt/slip photo or a statement PDF.
2. The bot replies with a summary and "บันทึก" / "ยกเลิก" buttons; nothing
   is saved yet (the same review-before-save rule as the web app).
3. A locked PDF first asks for its password. The password message is used
   once in memory and never saved to chat history or written to logs.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from finance_ai.line.document_import import (
    DocumentReadResult,
    UnsupportedDocumentError,
    format_import_summary,
    read_document,
)
from finance_ai.line.mapping_service import get_or_create_line_mapping
from finance_ai.line.pending_imports import PendingImport, pending_imports
from finance_ai.tools.bank_statement_pdf import (
    PdfPasswordIncorrectError,
    PdfPasswordRequiredError,
    StatementTotalsMismatchError,
)
from finance_ai.tools.receipt_ocr import ReceiptOcrResult
from finance_ai.tools.receipt_ocr_service import confirm_receipt_transactions

CONFIRM_LABEL: str = "บันทึก"
CANCEL_LABEL: str = "ยกเลิก"
PASSWORD_PROMPT: str = (
    "🔒 PDF นี้มีรหัสผ่าน พิมพ์รหัสผ่านส่งมาได้เลย\n"
    "ระบบใช้รหัสนี้เปิดไฟล์ครั้งเดียว ไม่บันทึกไว้ในประวัติแชท\n"
    '(หรือกด "ยกเลิก")'
)
WRONG_PASSWORD: str = "รหัสผ่านไม่ถูกต้อง กรุณาส่ง PDF มาใหม่อีกครั้ง"
NOTHING_FOUND: str = (
    "ไม่พบรายการในเอกสารนี้ ลองถ่ายรูปให้เห็นยอดเงินและวันที่ชัดขึ้น หรืออัปโหลดผ่านเว็บ"
)
TOTALS_MISMATCH: str = (
    "ยอดรวมในใบแจ้งยอดไม่ตรงกับรายการที่อ่านได้ จึงยังไม่บันทึก "
    "กรุณาอัปโหลดผ่านเว็บเพื่อตรวจทีละรายการ"
)
CANCELLED: str = "ยกเลิกแล้ว ไม่ได้บันทึกรายการใด"


@dataclass(frozen=True)
class LineReply:
    """Text to send back, with optional Quick Reply buttons."""

    text: str
    quick_reply_labels: tuple[str, ...] = ()


def start_document_import(
    line_user_id: str,
    content: bytes,
    ocr_model_provider: Callable[[], Any],
    chat_model_provider: Callable[[], Any],
) -> LineReply:
    """Read a new document and ask the user to confirm (or for a password).

    Args:
        line_user_id: LINE userId of the sender.
        content: The image/PDF bytes.
        ocr_model_provider: Returns the OCR vision model.
        chat_model_provider: Returns the chat model (statement AI fallback).

    Returns:
        The reply to send.

    Example:
        >>> start_document_import(uid, jpeg, get_ocr_chat_model, get_chat_model).text
        '📄 อ่านได้ 1 รายการ ...'
    """
    try:
        result = read_document(content, None, ocr_model_provider, chat_model_provider)
    except PdfPasswordRequiredError:
        pending_imports.put(line_user_id, PendingImport(locked_pdf=content))
        return LineReply(PASSWORD_PROMPT, (CANCEL_LABEL,))
    except (UnsupportedDocumentError, StatementTotalsMismatchError) as error:
        return LineReply(_explain(error))
    return _offer_drafts(line_user_id, result)


def continue_pending_import(
    line_user_id: str,
    text: str,
    session_factory: Callable[[], Session],
    ocr_model_provider: Callable[[], Any],
    chat_model_provider: Callable[[], Any],
) -> LineReply | None:
    """Handle a text message while an import waits; None lets the agent answer.

    Args:
        line_user_id: LINE userId of the sender.
        text: The message (may be a PDF password: never log or save it).
        session_factory: Factory for database sessions.
        ocr_model_provider: Returns the OCR vision model.
        chat_model_provider: Returns the chat model.

    Returns:
        The reply, or None when the text is not part of the import.

    Example:
        >>> continue_pending_import(uid, "บันทึก", factory, ocr, chat).text
        '✅ บันทึกแล้ว 3 รายการ'
    """
    pending = pending_imports.get(line_user_id)
    if pending is None:
        return None
    command = text.strip()
    if pending.locked_pdf is not None:
        pending_imports.pop(line_user_id)
        if command == CANCEL_LABEL:
            return LineReply(CANCELLED)
        return _read_with_password(
            line_user_id, pending.locked_pdf, command, ocr_model_provider, chat_model_provider
        )
    return _confirm_or_cancel(line_user_id, command, pending.drafts, session_factory)


def _read_with_password(
    line_user_id: str,
    content: bytes,
    password: str,
    ocr_model_provider: Callable[[], Any],
    chat_model_provider: Callable[[], Any],
) -> LineReply:
    """Try the password once; a wrong one asks for the PDF again."""
    try:
        result = read_document(content, password, ocr_model_provider, chat_model_provider)
    except PdfPasswordIncorrectError:
        return LineReply(WRONG_PASSWORD)
    except (UnsupportedDocumentError, StatementTotalsMismatchError) as error:
        return LineReply(_explain(error))
    return _offer_drafts(line_user_id, result)


def _confirm_or_cancel(
    line_user_id: str,
    command: str,
    drafts: list[ReceiptOcrResult],
    session_factory: Callable[[], Session],
) -> LineReply | None:
    """Save or drop waiting drafts; other text is left to the agent."""
    if command not in (CONFIRM_LABEL, CANCEL_LABEL):
        return None
    pending_imports.pop(line_user_id)
    if command == CANCEL_LABEL:
        return LineReply(CANCELLED)
    inserted, skipped = _save_drafts(line_user_id, drafts, session_factory)
    note = f" (ข้าม {skipped} รายการที่มีอยู่แล้ว)" if skipped else ""
    return LineReply(f"✅ บันทึกแล้ว {inserted} รายการ{note}")


def _save_drafts(
    line_user_id: str, drafts: list[ReceiptOcrResult], session_factory: Callable[[], Session]
) -> tuple[int, int]:
    """Save drafts for the app user linked to this LINE user."""
    with session_factory() as session:
        user_id = get_or_create_line_mapping(session, line_user_id).user_id
    result = confirm_receipt_transactions(drafts, user_id, session_factory)
    return result.inserted, result.skipped


def _offer_drafts(line_user_id: str, result: DocumentReadResult) -> LineReply:
    """Keep the drafts waiting and show the summary with confirm buttons."""
    if not result.drafts:
        return LineReply(NOTHING_FOUND)
    pending_imports.put(line_user_id, PendingImport(drafts=result.drafts))
    return LineReply(format_import_summary(result), (CONFIRM_LABEL, CANCEL_LABEL))


def _explain(error: Exception) -> str:
    """Thai explanation for a document that could not be read."""
    if isinstance(error, StatementTotalsMismatchError):
        return TOTALS_MISMATCH
    return str(error)
