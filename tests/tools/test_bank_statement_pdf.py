"""Tests for reading bank statements from PDF (fake data in a real statement layout)."""

import io
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from pypdf import PdfWriter

from finance_ai.tools.bank_statement_pdf import (
    MAX_STATEMENT_PAGES,
    PdfPasswordIncorrectError,
    PdfPasswordRequiredError,
    StatementTotalsMismatchError,
    UnsupportedStatementError,
    drafts_from_pages,
    find_account_holder,
    parse_statement_page,
    read_pdf_pages,
)

PAGE = """วันที่ เวลา รายการ ชองทาง หมายเลขเช็ครายละเอียด ถอน/โอนออก ฝาก/โอนเขา
รายการบัญชี
เงินฝากออมทรัพย์ - ไม่มีสมุด
กันยายน 2026
นาย สมชาย ทดสอบดี
XXXX123456
01/09/2026 08:15:00 X1 ENET จ่ายบิล ร้านข้าวมันไก่ตัวอย่าง           -65.00 +1,935.00
02/09/2026 12:00:00 X2 ENET รับโอนจาก KBANK x1111 น.ส. ตัวอย่าง ใจดี           +1,500.00 +3,435.00
03/09/2026 18:30:00 X1 ENET PromptPay x2222 สมชาย ทดสอบดี                 -1,000.00 +2,435.00
04/09/2026 09:00:00 X3 SIPI SIPS TRUE MONEY CO.,LTD.                          -120.50 +2,314.50
รวม -1,185.50 +1,500.00
PAGE 1 of 1"""


def _pdf_bytes(password: str | None = None) -> bytes:
    """A one-page PDF (optionally encrypted)."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    if password:
        writer.encrypt(password)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class TestReadPdfPages:
    """Opening the PDF, including password-protected e-statements."""

    def test_plain_pdf_returns_one_text_per_page(self) -> None:
        """An unencrypted PDF opens without a password."""
        assert len(read_pdf_pages(_pdf_bytes(), None)) == 1

    def test_encrypted_pdf_needs_password(self) -> None:
        """A locked statement asks for its password."""
        with pytest.raises(PdfPasswordRequiredError):
            read_pdf_pages(_pdf_bytes("01011990"), None)

    def test_wrong_password_is_reported(self) -> None:
        """A wrong password is a different, clear error."""
        with pytest.raises(PdfPasswordIncorrectError):
            read_pdf_pages(_pdf_bytes("01011990"), "wrong")

    def test_right_password_opens(self) -> None:
        """The correct password decrypts the statement."""
        assert len(read_pdf_pages(_pdf_bytes("01011990"), "01011990")) == 1

    def test_too_many_pages_rejected(self) -> None:
        """Very long statements are refused with the limit in the message."""
        writer = PdfWriter()
        for _ in range(MAX_STATEMENT_PAGES + 1):
            writer.add_blank_page(width=100, height=100)
        buffer = io.BytesIO()
        writer.write(buffer)

        with pytest.raises(UnsupportedStatementError, match=str(MAX_STATEMENT_PAGES)):
            read_pdf_pages(buffer.getvalue(), None)


class TestParseStatementPage:
    """Rows of the 'รายการบัญชี' layout are read exactly."""

    def test_reads_every_row_with_signed_amounts(self) -> None:
        """Withdrawals are negative, deposits positive, balances ignored."""
        lines = parse_statement_page(PAGE)

        assert [line.amount for line in lines] == [
            Decimal("-65.00"),
            Decimal("1500.00"),
            Decimal("-1000.00"),
            Decimal("-120.50"),
        ]
        assert lines[0].transaction_date == date(2026, 9, 1)
        assert lines[0].description == "จ่ายบิล ร้านข้าวมันไก่ตัวอย่าง"

    def test_totals_mismatch_is_detected(self) -> None:
        """If the parsed rows do not add up to the page's 'รวม' line, stop."""
        broken = PAGE.replace("-120.50 +2,314.50", "-12.50 +2,314.50")

        with pytest.raises(StatementTotalsMismatchError, match="หน้า 1"):
            drafts_from_pages([broken], chat_model=None)

    def test_buddhist_year_converted(self) -> None:
        """Statements printed with พ.ศ. years are converted to ค.ศ."""
        page = PAGE.replace("01/09/2026", "01/09/2569").replace("รวม -1,185.50 +1,500.00\n", "")

        assert parse_statement_page(page)[0].transaction_date == date(2026, 9, 1)


class TestAccountHolderAndOwnTransfers:
    """Money moved between the holder's own accounts is not spending."""

    def test_finds_holder_name_without_title(self) -> None:
        """The holder line 'นาย สมชาย ทดสอบดี' gives 'สมชาย ทดสอบดี'."""
        assert find_account_holder(PAGE) == "สมชาย ทดสอบดี"

    def test_own_transfer_flagged_and_others_not(self) -> None:
        """PromptPay to the holder's own name is flagged; other rows are not."""
        drafts = drafts_from_pages([PAGE], chat_model=None)

        flagged = [d.description for d in drafts if d.own_transfer]
        assert flagged == ["PromptPay x2222 สมชาย ทดสอบดี"]


class TestDraftsFromPages:
    """Pages become review drafts (never saved directly)."""

    def test_types_amounts_and_categories(self) -> None:
        """Signs become income/expense with positive amounts; expenses get a category."""
        drafts = drafts_from_pages([PAGE], chat_model=None)

        assert [(d.transaction_type, d.amount) for d in drafts] == [
            ("expense", Decimal("65.00")),
            ("income", Decimal("1500.00")),
            ("expense", Decimal("1000.00")),
            ("expense", Decimal("120.50")),
        ]
        assert drafts[0].category == "food"

    def test_unknown_layout_uses_ai_fallback_per_page(self) -> None:
        """Pages the exact parser cannot read go to the AI parser, one page each."""
        model = MagicMock()
        model.invoke.return_value.text = (
            '[{"date": "2026-09-05", "description": "ค่าน้ำ", "amount": 300, '
            '"category": "utilities", "type": "expense"}]'
        )

        drafts = drafts_from_pages(["รูปแบบอื่น 05/09 ค่าน้ำ 300", "หน้า 2"], chat_model=model)

        assert model.invoke.call_count == 2
        assert drafts[0].description == "ค่าน้ำ"
        assert drafts[0].category == "utilities"

    def test_unknown_layout_without_model_is_unsupported(self) -> None:
        """Without an AI model, an unknown layout gets a clear message."""
        with pytest.raises(UnsupportedStatementError):
            drafts_from_pages(["ไม่ใช่รายการเดินบัญชี"], chat_model=None)

    def test_scanned_pdf_without_text_is_unsupported(self) -> None:
        """Pages with no text layer (scans) are pointed to the slip scanner."""
        with pytest.raises(UnsupportedStatementError, match="สแกน"):
            drafts_from_pages(["", "  "], chat_model=MagicMock())
