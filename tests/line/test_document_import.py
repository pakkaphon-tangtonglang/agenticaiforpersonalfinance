"""Tests for reading LINE images/PDFs into drafts and summarising them."""

from datetime import date
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

from finance_ai.line.document_import import (
    DocumentReadResult,
    UnsupportedDocumentError,
    detect_document_mime,
    format_import_summary,
    read_document,
)
from finance_ai.tools.bank_statement_pdf import (
    PdfPasswordRequiredError,
    StatementDraft,
    UnsupportedStatementError,
)
from finance_ai.tools.receipt_ocr import ReceiptOcrResult

MODULE = "finance_ai.line.document_import"
JPEG = b"\xff\xd8\xff\xe0rest"
PNG = b"\x89PNG\r\n\x1a\nrest"
PDF = b"%PDF-1.7 rest"


def _receipt(amount: str = "80", kind: str = "expense", text: str = "กาแฟ") -> ReceiptOcrResult:
    return ReceiptOcrResult(
        transaction_type=kind,
        amount=Decimal(amount),
        transaction_date=date(2026, 11, 1),
        description=text,
    )


def _statement_row(amount: str, kind: str, own: bool = False) -> StatementDraft:
    return StatementDraft(
        transaction_type=kind,
        amount=Decimal(amount),
        transaction_date=date(2026, 10, 5),
        description="โอนเงิน",
        own_transfer=own,
    )


class TestDetectDocumentMime:
    """File type comes from the bytes, since LINE gives no reliable name for images."""

    @pytest.mark.parametrize(
        ("content", "expected"),
        [(JPEG, "image/jpeg"), (PNG, "image/png"), (PDF, "application/pdf")],
    )
    def test_known_types(self, content: bytes, expected: str) -> None:
        """JPEG, PNG and PDF are recognised by their magic bytes."""
        assert detect_document_mime(content) == expected

    def test_webp(self) -> None:
        """WEBP (RIFF....WEBP) is recognised."""
        assert detect_document_mime(b"RIFF\x00\x00\x00\x00WEBPrest") == "image/webp"

    def test_unknown_raises(self) -> None:
        """Other files get a clear Thai error."""
        with pytest.raises(UnsupportedDocumentError, match="PDF"):
            detect_document_mime(b"PK\x03\x04 zip")


class TestReadDocument:
    """read_document picks the statement reader or the OCR model."""

    def test_image_uses_ocr(self, monkeypatch: Any) -> None:
        """Images go to the OCR vision model."""
        ocr = MagicMock(return_value=[_receipt()])
        monkeypatch.setattr(f"{MODULE}.extract_document_fields", ocr)
        result = read_document(JPEG, None, lambda: "ocr-model", lambda: None)
        assert result == DocumentReadResult(drafts=[_receipt()], own_transfers_skipped=0)
        ocr.assert_called_once_with(JPEG, "image/jpeg", "ocr-model")

    def test_statement_pdf_skips_own_transfers(self, monkeypatch: Any) -> None:
        """Statement rows become drafts; own-account transfers are left out (as on the web)."""
        monkeypatch.setattr(f"{MODULE}.read_pdf_pages", MagicMock(return_value=["page"]))
        rows = [_statement_row("500", "expense"), _statement_row("1000", "income", own=True)]
        monkeypatch.setattr(f"{MODULE}.drafts_from_pages", MagicMock(return_value=rows))
        result = read_document(PDF, None, lambda: "ocr", lambda: "chat")
        assert [draft.amount for draft in result.drafts] == [Decimal("500")]
        assert result.own_transfers_skipped == 1

    def test_password_is_passed_and_errors_propagate(self, monkeypatch: Any) -> None:
        """A locked PDF raises PdfPasswordRequiredError for the caller to handle."""
        reader = MagicMock(side_effect=PdfPasswordRequiredError("locked"))
        monkeypatch.setattr(f"{MODULE}.read_pdf_pages", reader)
        with pytest.raises(PdfPasswordRequiredError):
            read_document(PDF, "secret", lambda: "ocr", lambda: None)
        reader.assert_called_once_with(PDF, "secret")

    def test_non_statement_pdf_falls_back_to_ocr(self, monkeypatch: Any) -> None:
        """A PDF receipt (not a statement layout) is read by the OCR model."""
        monkeypatch.setattr(f"{MODULE}.read_pdf_pages", MagicMock(return_value=["page"]))
        monkeypatch.setattr(
            f"{MODULE}.drafts_from_pages", MagicMock(side_effect=UnsupportedStatementError("x"))
        )
        ocr = MagicMock(return_value=[_receipt()])
        monkeypatch.setattr(f"{MODULE}.extract_document_fields", ocr)
        result = read_document(PDF, None, lambda: "ocr", lambda: None)
        assert len(result.drafts) == 1
        ocr.assert_called_once_with(PDF, "application/pdf", "ocr")


class TestFormatImportSummary:
    """The LINE summary replaces the web review table."""

    def test_counts_totals_and_rows(self) -> None:
        """Shows the count, totals per type, each row and the buttons hint."""
        result = DocumentReadResult(
            drafts=[_receipt("80"), _receipt("30000", "income", "เงินเดือน")],
            own_transfers_skipped=0,
        )
        summary = format_import_summary(result)
        assert "อ่านได้ 2 รายการ" in summary
        assert "รายจ่าย 80 บาท" in summary
        assert "รายรับ 30,000 บาท" in summary
        assert "1 พ.ย. · กาแฟ · -80 บาท" in summary
        assert "1 พ.ย. · เงินเดือน · +30,000 บาท" in summary
        assert "บันทึก" in summary and "ยกเลิก" in summary

    def test_long_lists_are_cut(self) -> None:
        """Only the first rows are listed, with a count of the rest."""
        drafts = [_receipt(str(index + 1)) for index in range(13)]
        summary = format_import_summary(DocumentReadResult(drafts=drafts, own_transfers_skipped=0))
        assert "10. " in summary and "11. " not in summary
        assert "และอีก 3 รายการ" in summary

    def test_mentions_skipped_own_transfers(self) -> None:
        """Skipped own-account transfers are mentioned."""
        result = DocumentReadResult(drafts=[_receipt()], own_transfers_skipped=2)
        assert "โอนเข้าบัญชีตัวเอง 2 รายการ" in format_import_summary(result)

    def test_decimal_amounts_keep_satang(self) -> None:
        """Amounts with satang keep two decimals."""
        result = DocumentReadResult(drafts=[_receipt("80.5")], own_transfers_skipped=0)
        assert "-80.50 บาท" in format_import_summary(result)
