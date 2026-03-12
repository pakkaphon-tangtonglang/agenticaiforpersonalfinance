"""Parser for bank statement CSV/Excel files.

Supports generic CSV and Excel formats. Handles Thai Buddhist Era
dates (year > 2400 → subtract 543 for CE).
"""

import csv
import io
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, Field

BUDDHIST_ERA_OFFSET = 543
BUDDHIST_ERA_THRESHOLD = 2400

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "food": ["อาหาร", "ข้าว", "กาแฟ", "ร้านอาหาร", "food", "restaurant", "cafe"],
    "transport": ["แท็กซี่", "bts", "mrt", "grab", "bolt", "น้ำมัน", "ทางด่วน"],
    "health": ["โรงพยาบาล", "ยา", "คลินิก", "หมอ", "pharmacy"],
    "utilities": ["ไฟฟ้า", "น้ำประปา", "โทรศัพท์", "อินเทอร์เน็ต", "true", "ais", "dtac"],
    "shopping": ["shopee", "lazada", "central", "ห้าง", "เสื้อผ้า"],
    "entertainment": ["netflix", "youtube", "spotify", "หนัง", "เกม"],
    "education": ["เรียน", "หนังสือ", "คอร์ส", "course", "udemy"],
}


class ParsedTransaction(BaseModel):
    """A single parsed transaction from a bank statement.

    Attributes:
        transaction_date: Date of the transaction.
        description: Transaction description text.
        amount: Transaction amount (positive = expense).
        category: Classified expense category.
        transaction_type: 'expense' or 'income'.

    Example:
        >>> t = ParsedTransaction(
        ...     transaction_date=date(2026, 3, 1),
        ...     description="ค่าอาหาร",
        ...     amount=Decimal("350"),
        ... )
    """

    transaction_date: date
    description: str
    amount: Decimal
    category: str = Field(default="other")
    transaction_type: str = Field(default="expense")


def parse_bank_statement(
    content: str | bytes,
    file_type: str = "csv",
) -> list[ParsedTransaction]:
    """Parse a bank statement file into transactions.

    Args:
        content: File content (str for CSV, bytes for Excel).
        file_type: 'csv' or 'xlsx'.

    Returns:
        List of parsed transactions.

    Raises:
        ValueError: If file_type is unsupported.

    Example:
        >>> txns = parse_bank_statement("date,desc,amount\\n...", "csv")
    """
    if file_type == "csv":
        text = content if isinstance(content, str) else content.decode("utf-8-sig")
        return parse_generic_csv(text)
    if file_type in ("xlsx", "excel"):
        if not isinstance(content, bytes):
            raise ValueError("Excel content must be bytes")
        return parse_excel_statement(content)
    raise ValueError(f"Unsupported file type: {file_type}")


def parse_generic_csv(content: str) -> list[ParsedTransaction]:
    """Parse a generic CSV with date, description, amount columns.

    Expects headers in first row. Tries to auto-detect columns.

    Args:
        content: CSV text content.

    Returns:
        List of parsed transactions.
    """
    reader = csv.DictReader(io.StringIO(content))
    transactions: list[ParsedTransaction] = []

    field_map = _detect_csv_columns(list(reader.fieldnames or []))
    for row in reader:
        txn = _parse_csv_row(row, field_map)
        if txn is not None:
            transactions.append(txn)
    return transactions


def _detect_csv_columns(
    headers: list[str],
) -> dict[str, str]:
    """Auto-detect column mapping from CSV headers.

    Args:
        headers: List of header strings.

    Returns:
        Dict mapping 'date', 'description', 'amount' to header names.
    """
    mapping: dict[str, str] = {}
    lower_headers = {h.lower().strip(): h for h in headers}

    date_keys = ["date", "วันที่", "transaction_date", "txn_date"]
    desc_keys = ["description", "รายละเอียด", "desc", "memo", "หมายเหตุ"]
    amount_keys = ["amount", "จำนวนเงิน", "debit", "withdrawal", "ยอดเงิน"]

    mapping["date"] = _find_header(lower_headers, date_keys, headers)
    mapping["description"] = _find_header(lower_headers, desc_keys, headers)
    mapping["amount"] = _find_header(lower_headers, amount_keys, headers)
    return mapping


def _find_header(
    lower_map: dict[str, str],
    candidates: list[str],
    fallback_headers: list[str],
) -> str:
    """Find best matching header from candidate list.

    Args:
        lower_map: Lowercase header → original header mapping.
        candidates: Candidate key names to search for.
        fallback_headers: Original headers for index fallback.

    Returns:
        Matched header name.
    """
    for key in candidates:
        if key in lower_map:
            return lower_map[key]
    return fallback_headers[0] if fallback_headers else ""


def _parse_csv_row(
    row: dict[str, Any],
    field_map: dict[str, str],
) -> ParsedTransaction | None:
    """Parse a single CSV row into a ParsedTransaction.

    Args:
        row: CSV row dict.
        field_map: Column mapping.

    Returns:
        ParsedTransaction or None if parsing fails.
    """
    try:
        raw_date = str(row.get(field_map.get("date", ""), "")).strip()
        description = str(row.get(field_map.get("description", ""), "")).strip()
        raw_amount = str(row.get(field_map.get("amount", ""), "0")).strip()

        txn_date = normalize_thai_date(raw_date)
        amount = _parse_amount(raw_amount)
        txn_type = "income" if amount < 0 else "expense"
        category = classify_expense_category(description)

        return ParsedTransaction(
            transaction_date=txn_date,
            description=description,
            amount=abs(amount),
            category=category,
            transaction_type=txn_type,
        )
    except (ValueError, InvalidOperation):
        return None


def _parse_amount(raw: str) -> Decimal:
    """Parse amount string, removing commas.

    Args:
        raw: Amount string like '1,234.56' or '-500'.

    Returns:
        Decimal amount.
    """
    cleaned = raw.replace(",", "").replace(" ", "")
    return Decimal(cleaned)


def normalize_thai_date(date_str: str) -> date:
    """Parse date string, handling Buddhist Era conversion.

    Supports formats: DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD.
    If year > 2400, assumes Buddhist Era and subtracts 543.

    Args:
        date_str: Date string to parse.

    Returns:
        Python date object.

    Raises:
        ValueError: If date cannot be parsed.

    Example:
        >>> normalize_thai_date("12/03/2569")
        datetime.date(2026, 3, 12)
    """
    date_str = date_str.strip()

    for sep in ["/", "-"]:
        parts = date_str.split(sep)
        if len(parts) == 3:
            return _parse_date_parts(parts, sep)

    raise ValueError(f"Cannot parse date: {date_str}")


def _parse_date_parts(parts: list[str], sep: str) -> date:
    """Parse date from 3-part split.

    Args:
        parts: Three string parts of the date.
        sep: The separator used.

    Returns:
        Python date object.

    Raises:
        ValueError: If parts cannot form a valid date.
    """
    nums = [int(p) for p in parts]

    if len(parts[0]) == 4 or (sep == "-" and nums[0] > 31):
        year, month, day = nums[0], nums[1], nums[2]
    else:
        day, month, year = nums[0], nums[1], nums[2]

    if year > BUDDHIST_ERA_THRESHOLD:
        year -= BUDDHIST_ERA_OFFSET

    return date(year, month, day)


def classify_expense_category(description: str) -> str:
    """Classify transaction description into expense category.

    Uses keyword matching on Thai and English terms.

    Args:
        description: Transaction description text.

    Returns:
        Category string like 'food', 'transport', etc.

    Example:
        >>> classify_expense_category("ร้านอาหาร สุกี้")
        'food'
    """
    lower = description.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in lower:
                return category
    return "other"


def parse_excel_statement(content: bytes) -> list[ParsedTransaction]:
    """Parse an Excel bank statement.

    Args:
        content: Excel file content as bytes.

    Returns:
        List of parsed transactions.
    """
    import openpyxl  # noqa: PLC0415

    wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True)
    ws = wb.active
    if ws is None:
        return []

    rows = list(ws.iter_rows(values_only=True))
    if len(rows) < 2:
        return []

    headers = [str(h or "") for h in rows[0]]
    field_map = _detect_csv_columns(headers)
    transactions: list[ParsedTransaction] = []

    for row in rows[1:]:
        row_dict = dict(zip(headers, [str(v or "") for v in row]))
        txn = _parse_csv_row(row_dict, field_map)
        if txn is not None:
            transactions.append(txn)

    wb.close()
    return transactions
