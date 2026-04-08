"""Parser for bank statement CSV/Excel files.

Supports generic CSV and Excel formats. Handles Thai Buddhist Era
dates (year > 2400 → subtract 543 for CE). Falls back to LLM-based
parsing when rule-based detection fails.
"""

import csv
import io
import json
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, Field

BUDDHIST_ERA_OFFSET = 543
BUDDHIST_ERA_THRESHOLD = 2400

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "food": [
        "อาหาร",
        "ข้าว",
        "กาแฟ",
        "ร้านอาหาร",
        "food",
        "restaurant",
        "cafe",
        "dine",
        "sushi",
        "shabu",
        "grocery",
        "supermarket",
        "starbucks",
        "mcdonald",
        "kfc",
        "pizza",
        "7-eleven",
        "lunch",
        "dinner",
        "breakfast",
    ],
    "transport": [
        "แท็กซี่",
        "bts",
        "mrt",
        "grab",
        "bolt",
        "น้ำมัน",
        "ทางด่วน",
        "gasoline",
        "gas station",
        "fuel",
        "taxi",
        "uber",
        "parking",
        "toll",
        "transport",
        "commute",
    ],
    "housing": [
        "ค่าเช่า",
        "บ้าน",
        "คอนโด",
        "หอพัก",
        "rent",
        "mortgage",
        "condo",
        "apartment",
        "housing",
    ],
    "health": [
        "โรงพยาบาล",
        "ยา",
        "คลินิก",
        "หมอ",
        "pharmacy",
        "hospital",
        "doctor",
        "gym",
        "fitness",
        "health",
        "medical",
        "dental",
    ],
    "utilities": [
        "ไฟฟ้า",
        "น้ำประปา",
        "โทรศัพท์",
        "อินเทอร์เน็ต",
        "true",
        "ais",
        "dtac",
        "internet",
        "phone bill",
        "electric",
        "water bill",
        "utility",
    ],
    "shopping": [
        "shopee",
        "lazada",
        "central",
        "ห้าง",
        "เสื้อผ้า",
        "gadget",
        "headphone",
        "amazon",
        "electronics",
        "clothes",
    ],
    "entertainment": [
        "netflix",
        "youtube",
        "spotify",
        "หนัง",
        "เกม",
        "movie",
        "game",
        "subscription",
        "disney",
    ],
    "education": [
        "เรียน",
        "หนังสือ",
        "คอร์ส",
        "course",
        "udemy",
        "school",
        "tuition",
        "book",
        "training",
    ],
    "investment": [
        "ลงทุน",
        "หุ้น",
        "กองทุน",
        "dca",
        "stock",
        "investment",
        "mutual fund",
        "ssf",
        "rmf",
    ],
}

INCOME_KEYWORDS: list[str] = [
    "เงินเดือน",
    "salary",
    "wage",
    "bonus",
    "freelance",
    "รายได้",
    "income",
    "เงินโอนเข้า",
    "deposit",
    "refund",
    "dividend",
    "ปันผล",
    "ดอกเบี้ย",
    "interest",
]


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
        Dict mapping 'date', 'description', 'amount', 'type',
        'category' to matched header names (empty string if not found).
    """
    mapping: dict[str, str] = {}
    lower_headers = {h.lower().strip(): h for h in headers}

    date_keys = ["date", "วันที่", "transaction_date", "txn_date"]
    desc_keys = ["description", "รายละเอียด", "desc", "memo", "หมายเหตุ"]
    amount_keys = ["amount", "จำนวนเงิน", "ยอดเงิน", "debit", "withdrawal"]
    type_keys = ["type", "transaction type", "txn type", "ประเภท"]
    category_keys = ["category", "หมวดหมู่", "หมวด"]

    mapping["date"] = _find_header(lower_headers, date_keys, headers)
    mapping["description"] = _find_header(lower_headers, desc_keys, headers)
    mapping["amount"] = _find_header(lower_headers, amount_keys, headers)
    mapping["type"] = _find_header(lower_headers, type_keys, [])
    mapping["category"] = _find_header(lower_headers, category_keys, [])
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

        raw_type = str(row.get(field_map.get("type", ""), "")).strip().lower()
        raw_category = str(row.get(field_map.get("category", ""), "")).strip()

        txn_date = normalize_thai_date(raw_date)
        amount = _parse_amount(raw_amount)
        txn_type = _resolve_transaction_type(raw_type, description, amount)
        category = _resolve_category(raw_category, description)

        return ParsedTransaction(
            transaction_date=txn_date,
            description=description,
            amount=abs(amount),
            category=category,
            transaction_type=txn_type,
        )
    except (ValueError, InvalidOperation):
        return None


CSV_CATEGORY_MAP: dict[str, str] = {
    "income": "other",
    "housing": "housing",
    "groceries": "food",
    "grocery": "food",
    "dining": "food",
    "food": "food",
    "utilities": "utilities",
    "utility": "utilities",
    "investment": "investment",
    "health": "health",
    "fitness": "health",
    "transport": "transport",
    "transportation": "transport",
    "shopping": "shopping",
    "entertainment": "entertainment",
    "education": "education",
}


def _resolve_transaction_type(
    raw_type: str,
    description: str,
    amount: Decimal,
) -> str:
    """Determine transaction type from explicit type column, then description.

    Priority:
    1. Explicit type column: 'credit' → income, 'debit' → expense
    2. Description keyword match via detect_income()
    3. Default to 'expense'

    Args:
        raw_type: Lowercased value from type column (empty if not present).
        description: Transaction description.
        amount: Parsed amount (sign ignored; use raw_type/description).

    Returns:
        'income' or 'expense'.
    """
    if raw_type in ("credit", "income", "รายรับ", "เครดิต"):
        return "income"
    if raw_type in ("debit", "expense", "รายจ่าย", "เดบิต"):
        return "expense"
    if detect_income(description):
        return "income"
    return "expense"


def _resolve_category(raw_category: str, description: str) -> str:
    """Resolve category from explicit CSV column, then description keywords.

    Args:
        raw_category: Category value from CSV column (empty if not present).
        description: Transaction description for keyword fallback.

    Returns:
        Internal category key (e.g. 'food', 'transport', 'other').
    """
    if raw_category:
        mapped = CSV_CATEGORY_MAP.get(raw_category.lower())
        if mapped:
            return mapped
    return classify_expense_category(description)


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


def detect_income(description: str) -> bool:
    """Check if a description indicates income.

    Args:
        description: Transaction description text.

    Returns:
        True if description matches income keywords.

    Example:
        >>> detect_income("Salary March 2026")
        True
    """
    lower = description.lower()
    return any(kw in lower for kw in INCOME_KEYWORDS)


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


# ─────────────────── LLM Fallback Parser ───────────────────────

_MAX_PREVIEW_CHARS = 3000

_LLM_PARSE_PROMPT: str = (
    "คุณเป็นผู้เชี่ยวชาญแปลงข้อมูลทางการเงินเป็น JSON\n\n"
    "จากข้อมูลด้านล่าง ให้สกัดรายการธุรกรรมทางการเงินออกมาให้ครบทุกรายการ\n"
    "ตอบเป็น JSON array เท่านั้น ห้ามมีข้อความอื่น\n\n"
    "แต่ละรายการต้องมี:\n"
    '- "date": วันที่ในรูปแบบ YYYY-MM-DD\n'
    '- "description": คำอธิบายรายการ\n'
    '- "amount": จำนวนเงิน (ตัวเลข, บวก=รายจ่าย, ลบ=รายรับ)\n'
    '- "category": หมวดหมู่ (food/transport/health/utilities/'
    "shopping/entertainment/education/other)\n"
    '- "type": "expense" หรือ "income"\n\n'
    "กฎ:\n"
    "- ถ้าเป็นรายรับ/เงินเดือน/โอนเข้า → type=income, amount เป็นค่าบวก\n"
    "- ถ้าเป็นรายจ่าย/จ่าย/ซื้อ/โอนออก → type=expense, amount เป็นค่าบวก\n"
    "- เดาหมวดหมู่จากคำอธิบาย\n"
    "- ถ้าวันที่เป็น พ.ศ. ให้แปลงเป็น ค.ศ. (ลบ 543)\n\n"
    "ข้อมูล:\n{content}"
)


def parse_with_llm(
    raw_content: str,
    chat_model: Any,
) -> list[ParsedTransaction]:
    """Parse financial data using LLM when rule-based parsing fails.

    Args:
        raw_content: Raw text content from the uploaded file.
        chat_model: LangChain BaseChatModel instance.

    Returns:
        List of parsed transactions extracted by the LLM.
    """
    from langchain_core.messages import HumanMessage  # noqa: PLC0415

    preview = raw_content[:_MAX_PREVIEW_CHARS]
    prompt = _LLM_PARSE_PROMPT.format(content=preview)
    response = chat_model.invoke([HumanMessage(content=prompt)])
    return _parse_llm_response(str(response.content))


def _parse_llm_response(
    response_text: str,
) -> list[ParsedTransaction]:
    """Parse LLM JSON response into ParsedTransaction list.

    Args:
        response_text: Raw LLM response (should be JSON array).

    Returns:
        List of parsed transactions.
    """
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(items, list):
        return []

    transactions: list[ParsedTransaction] = []
    for item in items:
        txn = _convert_llm_item(item)
        if txn is not None:
            transactions.append(txn)
    return transactions


def _convert_llm_item(
    item: dict[str, Any],
) -> ParsedTransaction | None:
    """Convert a single LLM-extracted item to ParsedTransaction.

    Args:
        item: Dict with date, description, amount, category, type.

    Returns:
        ParsedTransaction or None if conversion fails.
    """
    try:
        raw_date = str(item.get("date", ""))
        txn_date = normalize_thai_date(raw_date)
        description = str(item.get("description", ""))
        amount = Decimal(str(item.get("amount", "0")).replace(",", ""))
        category = str(item.get("category", "other"))
        txn_type = str(item.get("type", "expense"))

        if category not in CATEGORY_KEYWORDS and category != "other":
            category = classify_expense_category(description)
        if txn_type not in ("income", "expense"):
            txn_type = "income" if amount < 0 else "expense"

        return ParsedTransaction(
            transaction_date=txn_date,
            description=description,
            amount=abs(amount),
            category=category,
            transaction_type=txn_type,
        )
    except (ValueError, InvalidOperation, KeyError):
        return None
