"""General financial document OCR via Gemini vision.

Extracts income/expense fields from any financial document image or PDF
(receipts, payslips, tax invoices, transfer slips) using a multimodal
chat model. The model pre-fills (guesses) all fields; the caller is
expected to let the user recheck and confirm before persisting.

This module performs NO database writes — it only returns drafts.

Before sending an image to the vision model, large photos are downscaled
and re-encoded as JPEG: a 31B vision model spends most of its time on
the input image, so shrinking a multi-megabyte phone photo to a
~1568px JPEG typically cuts OCR latency by 50-80%.
"""

import base64
import importlib
import io
import json
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from finance_ai.core.config import get_settings
from finance_ai.tools.bank_statement_parser import (
    classify_expense_category,
    detect_income,
    normalize_thai_date,
)
from finance_ai.tools.expense_constants import VALID_EXPENSE_CATEGORIES

SUPPORTED_MIME_TYPES: tuple[str, ...] = (
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/heic",
    "image/heif",
    "application/pdf",
)

_VALID_TYPES: tuple[str, ...] = ("income", "expense")


class ReceiptOcrResult(BaseModel):
    """A single draft transaction extracted from a document.

    Attributes:
        transaction_type: 'income' or 'expense'.
        amount: Transaction amount in THB (positive).
        transaction_date: Date of the transaction.
        description: Description text from the document.
        category: Expense category key (food, transport, ...).
        income_type: Income type key (salary, freelance, ...) for income.
        withholding_tax: Withholding tax amount (THB) for income.
        employer_name: Employer/source name for income.
        confidence: Model confidence score 0.0-1.0.

    Example:
        >>> ReceiptOcrResult(
        ...     transaction_type="expense",
        ...     amount=Decimal("350"),
        ...     transaction_date=date(2026, 3, 1),
        ...     description="ร้านอาหาร",
        ...     category="food",
        ... )
    """

    transaction_type: str
    amount: Decimal
    transaction_date: date
    description: str
    category: str = Field(default="other")
    income_type: str = Field(default="other")
    withholding_tax: Decimal = Field(default=Decimal("0"))
    employer_name: str | None = Field(default=None)
    confidence: float = Field(default=0.0)


def build_document_ocr_prompt() -> str:
    """Build the Thai prompt instructing the model to extract financial fields.

    Returns:
        Prompt string requesting a strict JSON array of transactions.

    Example:
        >>> prompt = build_document_ocr_prompt()
    """
    categories = ", ".join(VALID_EXPENSE_CATEGORIES)
    return (
        "คุณเป็นผู้เชี่ยวชาญสกัดข้อมูลทางการเงินจากเอกสาร "
        "ไม่ว่าจะเป็น ใบเสร็จ สลิปเงินเดือน ใบกำกับภาษี หรือสลิปโอนเงิน\n\n"
        "สกัดรายการการเงินทั้งหมดจากรูปภาพ แล้วตอบเป็น JSON array เท่านั้น "
        "ห้ามมีข้อความอื่นนอกจาก JSON\n\n"
        "แต่ละรายการต้องมีฟิลด์:\n"
        '- "transaction_type": "income" หรือ "expense"\n'
        "  (รายรับ/เงินเดือน/โอนเข้า = income, รายจ่าย/จ่าย/ซื้อ = expense)\n"
        '- "amount": จำนวนเงินเป็นตัวเลข (บาท, ค่าบวก)\n'
        '- "transaction_date": วันที่ รูปแบบ YYYY-MM-DD '
        "(ถ้าเป็น พ.ศ. ให้ลบ 543 เป็น ค.ศ.)\n"
        '- "description": คำอธิบายรายการ\n'
        f'- "category": เลือกจาก [{categories}] ถ้าไม่ตรงใช้ "other"\n'
        '- "income_type": ประเภทรายได้ (salary, freelance, bonus, '
        "investment, rental, other) เฉพาะ income\n"
        '- "withholding_tax": ภาษีหัก ณ ที่จ่าย (บาท) เฉพาะ income\n'
        '- "employer_name": ชื่อนายจ้าง/แหล่งรายได้ เฉพาะ income\n'
        '- "confidence": ค่าความมั่นใจ 0.0-1.0\n'
    )


def extract_document_fields(
    image_bytes: bytes,
    mime_type: str,
    chat_model: BaseChatModel,
) -> list[ReceiptOcrResult]:
    """Extract transaction drafts from a document image/PDF via vision LLM.

    This is read-only: it returns drafts and performs no database writes.
    The caller must let the user recheck and confirm before persisting.

    Args:
        image_bytes: Raw bytes of the image or PDF.
        mime_type: MIME type (e.g. 'image/png', 'application/pdf').
        chat_model: Multimodal LangChain chat model (e.g. Gemini).

    Returns:
        List of parsed transaction drafts (empty if nothing readable).

    Raises:
        ValueError: If mime_type is not a supported image/PDF type.

    Example:
        >>> drafts = extract_document_fields(b"...", "image/png", model)
    """
    _validate_mime_type(mime_type)
    prepared_bytes, prepared_mime = _preprocess_image(
        image_bytes, mime_type, get_settings().ocr_max_image_edge
    )
    message = _build_vision_message(prepared_bytes, prepared_mime)
    response = _invoke_vision_model(chat_model, [message])
    items = _parse_ocr_json_response(str(response.content))
    return _convert_ocr_items(items)


def _validate_mime_type(mime_type: str) -> None:
    """Raise ValueError if the mime type is not a supported image/PDF.

    Args:
        mime_type: MIME type string to check.

    Raises:
        ValueError: If mime_type is not supported.
    """
    if mime_type.lower() not in SUPPORTED_MIME_TYPES:
        raise ValueError(
            f"Unsupported mime type: '{mime_type}'. "
            f"Supported: {', '.join(SUPPORTED_MIME_TYPES)}"
        )


_IMAGE_MIME_TYPES: tuple[str, ...] = (
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/heic",
    "image/heif",
)


def _preprocess_image(
    image_bytes: bytes,
    mime_type: str,
    max_edge: int,
) -> tuple[bytes, str]:
    """Prepare document bytes for the vision model.

    PDFs are rendered to JPEG first: several OCR providers (notably
    Ollama vision) reject application/pdf inputs outright. HEIC/HEIF
    files go through Pillow but fall back to the original bytes when
    the optional decoder is missing. If Pillow cannot decode the
    bytes, the original is returned so callers still get *some* input
    to the model rather than a hard failure.

    Args:
        image_bytes: Raw uploaded image/PDF bytes.
        mime_type: MIME type of the bytes.
        max_edge: Maximum edge length in pixels; larger images are shrunk.

    Returns:
        Tuple of (processed_bytes, processed_mime_type).
    """
    if mime_type.lower() == "application/pdf":
        image_bytes, mime_type = _render_pdf_first_page(image_bytes, max_edge)
    if mime_type.lower() not in _IMAGE_MIME_TYPES:
        return image_bytes, mime_type
    return _shrink_image_bytes(image_bytes, mime_type, max_edge)


def _render_pdf_first_page(
    pdf_bytes: bytes,
    max_edge: int,
) -> tuple[bytes, str]:
    """Render the first page of a PDF to a downscaled JPEG.

    Falls back to the original bytes on any failure (pypdfium2 not
    installed, corrupt PDF, empty document) — providers that accept
    PDFs natively (e.g. Gemini) still work that way.

    Args:
        pdf_bytes: Raw PDF bytes.
        max_edge: Maximum edge length in pixels for the rendered page.

    Returns:
        Tuple of (jpeg_bytes, "image/jpeg") or the original on failure.
    """
    try:
        # import_module avoids an import-untyped ignore that flips to
        # unused-ignore in mypy envs where pypdfium2 ships stubs.
        pdfium: Any = importlib.import_module("pypdfium2")
    except ImportError:
        return pdf_bytes, "application/pdf"
    try:
        with pdfium.PdfDocument(pdf_bytes) as document:
            bitmap = document[0].render(scale=2.0)
            return _pil_image_to_jpeg_bytes(bitmap.to_pil(), max_edge)
    except (AttributeError, IndexError, OSError, ValueError, pdfium.PdfiumError):
        return pdf_bytes, "application/pdf"


def _pil_image_to_jpeg_bytes(
    image: Any,
    max_edge: int,
) -> tuple[bytes, str]:
    """Convert a PIL image to downscaled JPEG bytes.

    Args:
        image: PIL Image (already loaded).
        max_edge: Maximum edge length in pixels; larger images are shrunk.

    Returns:
        Tuple of (jpeg_bytes, "image/jpeg").
    """
    converted = image.convert("RGB")
    converted.thumbnail((max_edge, max_edge))
    out = io.BytesIO()
    converted.save(out, format="JPEG", quality=85, optimize=True)
    return out.getvalue(), "image/jpeg"


def _shrink_image_bytes(
    image_bytes: bytes,
    mime_type: str,
    max_edge: int,
) -> tuple[bytes, str]:
    """Open with Pillow, downscale, and re-encode as JPEG; fall back on error.

    Args:
        image_bytes: Raw image bytes.
        mime_type: MIME type of the bytes.
        max_edge: Maximum edge length in pixels.

    Returns:
        Tuple of (jpeg_bytes, "image/jpeg") or the original on any failure.
    """
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:
        return image_bytes, mime_type
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            return _pil_image_to_jpeg_bytes(img, max_edge)
    except (UnidentifiedImageError, OSError, ValueError):
        return image_bytes, mime_type


def _invoke_vision_model(
    chat_model: BaseChatModel,
    messages: list[HumanMessage],
) -> Any:
    """Invoke the chat model without forcing a JSON output mode.

    Ollama's format="json" constrains the root to a JSON object, which
    conflicts with the array the prompt requests, and Ollama Cloud
    silently ignores the parameter. The prompt plus the tolerant parser
    are the reliable contract.

    Args:
        chat_model: Multimodal chat model.
        messages: List of messages to send.

    Returns:
        Model response.
    """
    return chat_model.invoke(messages)


def _encode_image_data_url(image_bytes: bytes, mime_type: str) -> str:
    """Encode raw image bytes as a base64 data URL.

    Args:
        image_bytes: Raw image bytes.
        mime_type: MIME type of the bytes.

    Returns:
        Data URL string suitable for image_url content blocks.
    """
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def _build_image_content_block(
    image_bytes: bytes,
    mime_type: str,
) -> dict[str, Any]:
    """Build a base64 data-URL image content block.

    Google Gemini, Ollama, OpenRouter and OpenAI all accept the Chat
    Completions image_url block with a data URL, so a single shape
    serves every provider.

    Args:
        image_bytes: Raw image/PDF bytes.
        mime_type: MIME type of the bytes.

    Returns:
        Image content block dict.
    """
    return {
        "type": "image_url",
        "image_url": {"url": _encode_image_data_url(image_bytes, mime_type)},
    }


def _build_vision_message(
    image_bytes: bytes,
    mime_type: str,
) -> HumanMessage:
    """Build a multimodal HumanMessage with the image and the OCR prompt.

    Args:
        image_bytes: Raw image/PDF bytes.
        mime_type: MIME type of the bytes.

    Returns:
        HumanMessage containing the image part and the text prompt.
    """
    return HumanMessage(
        content=[
            _build_image_content_block(image_bytes, mime_type),
            {"type": "text", "text": build_document_ocr_prompt()},
        ]
    )


def _parse_ocr_json_response(text: str) -> list[dict[str, Any]]:
    """Parse the model's JSON response into a list of item dicts.

    Strips markdown code fences, unwraps dict-wrapped arrays (the shape
    Ollama's JSON mode forces), and tolerates non-JSON output by
    returning an empty list.

    Args:
        text: Raw model response text.

    Returns:
        List of item dicts; empty if parsing fails or no items found.
    """
    cleaned = _strip_code_fences(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    return _extract_item_list(parsed)


def _strip_code_fences(text: str) -> str:
    """Strip a leading markdown code fence from model output.

    Args:
        text: Raw model response text.

    Returns:
        Text with a leading ``` fence removed, or the original text.
    """
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    parts = cleaned.split("\n", 1)
    if len(parts) < 2:
        return ""
    return parts[1].rsplit("```", 1)[0].strip()


def _extract_item_list(parsed: Any) -> list[dict[str, Any]]:
    """Extract a list of item dicts from a parsed JSON value.

    Accepts a top-level list, a dict wrapping the list under a common
    key, a numbered-object shape ({"0": {...}, "1": {...}}), or a single
    item dict.

    Args:
        parsed: Value returned by json.loads.

    Returns:
        List of item dicts; empty if the value holds no items.
    """
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if not isinstance(parsed, dict):
        return []
    return [item for item in _unwrap_dict_items(parsed) if isinstance(item, dict)]


def _unwrap_dict_items(parsed: dict[str, Any]) -> list[Any]:
    """Unwrap a dict into a list of candidate item values.

    Args:
        parsed: Dict from json.loads.

    Returns:
        List of candidate item values (possibly empty).
    """
    for key in ("transactions", "items", "data", "drafts", "results"):
        value = parsed.get(key)
        if isinstance(value, list):
            return value
    for value in parsed.values():
        if isinstance(value, list):
            return value
    dict_values = [value for value in parsed.values() if isinstance(value, dict)]
    if dict_values:
        return dict_values
    if "amount" in parsed:
        return [parsed]
    return []


def _convert_ocr_items(
    items: list[dict[str, Any]],
) -> list[ReceiptOcrResult]:
    """Convert parsed item dicts into validated ReceiptOcrResult drafts.

    Drops any item that fails validation (bad amount, bad date, etc.).

    Args:
        items: List of raw item dicts from the model response.

    Returns:
        List of valid ReceiptOcrResult drafts.
    """
    drafts: list[ReceiptOcrResult] = []
    for item in items:
        draft = _convert_ocr_item(item)
        if draft is not None:
            drafts.append(draft)
    return drafts


def _convert_ocr_item(
    item: dict[str, Any],
) -> ReceiptOcrResult | None:
    """Convert a single item dict into a ReceiptOcrResult, or None on failure.

    Args:
        item: Dict with transaction_type, amount, transaction_date, etc.

    Returns:
        ReceiptOcrResult, or None if required fields are invalid.
    """
    try:
        amount = _parse_amount(item.get("amount"))
        txn_date = normalize_thai_date(str(item.get("transaction_date", "")))
        description = str(item.get("description", "")).strip()
        txn_type = _normalize_type(item.get("transaction_type"), description)
        category = _normalize_category(item.get("category"), description)
        confidence = _parse_confidence(item.get("confidence"))
        income_type, withholding_tax, employer = _income_metadata(item, txn_type)
        return ReceiptOcrResult(
            transaction_type=txn_type,
            amount=amount,
            transaction_date=txn_date,
            description=description,
            category=category,
            income_type=income_type,
            withholding_tax=withholding_tax,
            employer_name=employer,
            confidence=confidence,
        )
    except (ValueError, InvalidOperation, TypeError):
        return None


def _parse_amount(raw: Any) -> Decimal:
    """Parse an amount value into a positive Decimal.

    Args:
        raw: Amount value (str, int, float).

    Returns:
        Decimal amount (absolute value).

    Raises:
        InvalidOperation: If the value cannot be parsed as a number.
    """
    cleaned = str(raw).replace(",", "").strip()
    return abs(Decimal(cleaned))


def _normalize_type(raw: Any, description: str) -> str:
    """Normalize the transaction type, falling back to description keywords.

    Args:
        raw: Raw type value from the model.
        description: Description text for keyword fallback.

    Returns:
        'income' or 'expense'.
    """
    value = str(raw).strip().lower()
    if value in _VALID_TYPES:
        return value
    if detect_income(description):
        return "income"
    return "expense"


def _normalize_category(raw: Any, description: str) -> str:
    """Normalize the category, reclassifying from description if out of set.

    Args:
        raw: Raw category value from the model.
        description: Description text for keyword fallback.

    Returns:
        A valid category key.
    """
    value = str(raw).strip().lower()
    if value in VALID_EXPENSE_CATEGORIES:
        return value
    return classify_expense_category(description)


def _parse_confidence(raw: Any) -> float:
    """Parse a confidence value into a float in [0.0, 1.0].

    Args:
        raw: Raw confidence value from the model.

    Returns:
        Confidence clamped to [0.0, 1.0], or 0.0 if unparseable.
    """
    try:
        value = float(str(raw).strip())
    except (ValueError, TypeError):
        return 0.0
    return max(0.0, min(1.0, value))


def _income_metadata(
    item: dict[str, Any],
    txn_type: str,
) -> tuple[str, Decimal, str | None]:
    """Extract income-specific metadata (only meaningful for income type).

    Args:
        item: Raw item dict.
        txn_type: Resolved transaction type.

    Returns:
        Tuple of (income_type, withholding_tax, employer_name).
    """
    if txn_type != "income":
        return "other", Decimal("0"), None
    income_type = str(item.get("income_type", "other")).strip().lower() or "other"
    try:
        withholding_tax = abs(Decimal(str(item.get("withholding_tax", "0")).replace(",", "")))
    except (InvalidOperation, ValueError):
        withholding_tax = Decimal("0")
    employer = item.get("employer_name")
    employer_name = str(employer).strip() if employer else None
    return income_type, withholding_tax, employer_name or None
