"""Read bank statements from PDF into review drafts.

Thai banks send e-statements as PDFs, usually password-protected. This module
opens them (password used in memory only), reads every page's text layer and
turns rows into drafts the user reviews before anything is saved.

Known layouts are parsed exactly (free, instant) and checked against each
page's printed "รวม" totals; the AI parser is only a per-page fallback for
layouts the exact parser does not recognise. Transfers to the account
holder's own name are flagged so they are not counted as spending.
"""

import io
import re
from datetime import date
from decimal import Decimal
from typing import Any, Optional

from pydantic import BaseModel
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from finance_ai.tools.bank_statement_parser import (
    ParsedTransaction,
    classify_expense_category,
    normalize_thai_date,
    parse_with_llm,
)

MAX_STATEMENT_PAGES = 20
SCANNED_PDF_MESSAGE = "PDF นี้เป็นภาพสแกน ไม่มีข้อความให้อ่าน — ใช้แท็บ 'สแกนสลิป/ใบเสร็จ' แทน"
UNSUPPORTED_LAYOUT_MESSAGE = (
    "ยังอ่านรูปแบบรายการเดินบัญชีนี้ไม่ได้ — ลองดาวน์โหลดเป็น CSV/Excel จากแอปธนาคาร"
)

# "dd/mm/yyyy hh:mm:ss <code> <channel> <description> <±amount> <±balance>"
_ROW = re.compile(
    r"^(\d{2}/\d{2}/\d{4}) \d{2}:\d{2}:\d{2} \S+ \S+ (.+?)\s+"  # date, time, code, channel, text
    r"([+-][\d,]+\.\d{2}) [+-][\d,]+\.\d{2}\s*$"  # signed amount, balance
)
# Page totals line: "รวม -<withdrawals> +<deposits>"
_TOTALS = re.compile(r"^รวม\s+-([\d,]+\.\d{2})\s+\+([\d,]+\.\d{2})")
_HOLDER = re.compile(r"^(?:นาย|นางสาว|นาง|น\.ส\.|Mr\.?|Mrs\.?|Ms\.?|Miss)\s+(.+?)\s*$")


class PdfPasswordRequiredError(ValueError):
    """The PDF is encrypted and no password was given."""


class PdfPasswordIncorrectError(ValueError):
    """The given password does not open the PDF."""


class UnsupportedStatementError(ValueError):
    """The PDF cannot be read as a statement (layout, scan, size)."""


class StatementTotalsMismatchError(ValueError):
    """Parsed rows do not add up to a page's printed totals."""


class StatementLine(BaseModel):
    """One statement row; amount < 0 is a withdrawal, > 0 a deposit."""

    transaction_date: date
    description: str
    amount: Decimal


class StatementDraft(BaseModel):
    """A row prepared for the review table (never saved directly)."""

    transaction_type: str
    amount: Decimal
    transaction_date: date
    description: str
    category: str = "other"
    income_type: str = "other"
    own_transfer: bool = False


def _decimal(text: str) -> Decimal:
    """'1,935.00' or '-65.00' → Decimal."""
    return Decimal(text.replace(",", ""))


def read_pdf_pages(content: bytes, password: Optional[str]) -> list[str]:
    """Open a statement PDF and return each page's text.

    Args:
        content: PDF bytes.
        password: Password for encrypted statements (used in memory only).

    Returns:
        One text string per page ("" for pages without a text layer).

    Raises:
        PdfPasswordRequiredError: Encrypted and no password given.
        PdfPasswordIncorrectError: The password does not open it.
        UnsupportedStatementError: Not a readable PDF, or too many pages.
    """
    try:
        reader = PdfReader(io.BytesIO(content))
    except (PdfReadError, ValueError) as error:
        raise UnsupportedStatementError("ไฟล์นี้ไม่ใช่ PDF ที่อ่านได้") from error
    if reader.is_encrypted:
        _decrypt(reader, password)
    if len(reader.pages) > MAX_STATEMENT_PAGES:
        raise UnsupportedStatementError(
            f"รายการเดินบัญชียาวเกินไป ({len(reader.pages)} หน้า,"
            f" สูงสุด {MAX_STATEMENT_PAGES} หน้า)"
        )
    return [page.extract_text() or "" for page in reader.pages]


def _decrypt(reader: PdfReader, password: Optional[str]) -> None:
    """Decrypt an encrypted statement or raise a password error."""
    if not password:
        raise PdfPasswordRequiredError("PDF นี้มีรหัสผ่าน กรุณาใส่รหัสผ่านของไฟล์")
    if not reader.decrypt(password):
        raise PdfPasswordIncorrectError("รหัสผ่าน PDF ไม่ถูกต้อง")


def parse_statement_page(text: str) -> list[StatementLine]:
    """Read the rows of one page in the 'รายการบัญชี' layout.

    Args:
        text: Page text from the PDF.

    Returns:
        Rows in page order (empty when the layout does not match).

    Example:
        >>> parse_statement_page(page_text)[0].amount
        Decimal('-65.00')
    """
    lines = []
    for raw in text.splitlines():
        match = _ROW.match(raw.strip())
        if match:
            lines.append(
                StatementLine(
                    transaction_date=normalize_thai_date(match.group(1)),
                    description=re.sub(r"\s+", " ", match.group(2)).strip(),
                    amount=_decimal(match.group(3)),
                )
            )
    return lines


def _check_page_totals(lines: list[StatementLine], text: str, page_number: int) -> None:
    """Compare parsed rows with the page's printed 'รวม' line (when present)."""
    totals = next(
        (_TOTALS.match(raw.strip()) for raw in text.splitlines() if _TOTALS.match(raw.strip())),
        None,
    )
    if totals is None:
        return
    withdrawn = -sum((line.amount for line in lines if line.amount < 0), Decimal("0"))
    deposited = sum((line.amount for line in lines if line.amount > 0), Decimal("0"))
    if (withdrawn, deposited) != (_decimal(totals.group(1)), _decimal(totals.group(2))):
        raise StatementTotalsMismatchError(
            f"อ่านหน้า {page_number} ได้ยอดไม่ตรงกับยอดรวมของธนาคาร — ยังไม่บันทึกอะไร"
        )


def find_account_holder(text: str) -> Optional[str]:
    """Return the account holder's name (without title) from the page header.

    Args:
        text: Page text.

    Returns:
        e.g. "สมชาย ทดสอบดี", or None when no holder line is found.
    """
    for raw in text.splitlines():
        if _ROW.match(raw.strip()):
            return None
        match = _HOLDER.match(raw.strip())
        if match:
            return match.group(1)
    return None


def _is_own_transfer(description: str, holder: Optional[str]) -> bool:
    """True when a row names the account holder (money between own accounts)."""
    return bool(holder) and re.sub(r"\s+", "", holder or "") in re.sub(r"\s+", "", description)


def _draft_from_line(line: StatementLine, holder: Optional[str]) -> StatementDraft:
    """Turn a signed statement row into a review draft."""
    is_income = line.amount > 0
    return StatementDraft(
        transaction_type="income" if is_income else "expense",
        amount=abs(line.amount),
        transaction_date=line.transaction_date,
        description=line.description,
        category="other" if is_income else classify_expense_category(line.description),
        own_transfer=_is_own_transfer(line.description, holder),
    )


def _draft_from_parsed(txn: ParsedTransaction) -> StatementDraft:
    """Turn an AI-parsed row into a review draft."""
    return StatementDraft(
        transaction_type=txn.transaction_type,
        amount=abs(txn.amount),
        transaction_date=txn.transaction_date,
        description=txn.description,
        category=txn.category,
    )


def _exact_drafts(pages: list[str]) -> list[StatementDraft]:
    """Parse pages with the exact layout; [] when no page matches it."""
    holder = next((h for h in map(find_account_holder, pages) if h), None)
    drafts: list[StatementDraft] = []
    for number, text in enumerate(pages, start=1):
        lines = parse_statement_page(text)
        _check_page_totals(lines, text, number)
        drafts.extend(_draft_from_line(line, holder) for line in lines)
    return drafts


def drafts_from_pages(pages: list[str], chat_model: Any) -> list[StatementDraft]:
    """Turn statement pages into review drafts.

    Args:
        pages: Text of each page.
        chat_model: Chat model for the per-page AI fallback, or None.

    Returns:
        Drafts in statement order.

    Raises:
        StatementTotalsMismatchError: Exact parsing disagrees with page totals.
        UnsupportedStatementError: Scanned PDF, or unknown layout without a model.
    """
    if not any(text.strip() for text in pages):
        raise UnsupportedStatementError(SCANNED_PDF_MESSAGE)
    drafts = _exact_drafts(pages)
    if drafts:
        return drafts
    if chat_model is None:
        raise UnsupportedStatementError(UNSUPPORTED_LAYOUT_MESSAGE)
    parsed = [txn for text in pages if text.strip() for txn in parse_with_llm(text, chat_model)]
    return [_draft_from_parsed(txn) for txn in parsed]
