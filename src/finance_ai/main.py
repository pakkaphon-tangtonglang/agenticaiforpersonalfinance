"""FastAPI application for Personal Finance AI.

Provides REST API endpoints for chat, conversations, bank statement
upload, dashboard, and asset monitoring.

Run with: make dev (development, port 8080) or make run (production)
"""

import json
from collections.abc import Generator
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import asyncio

from datetime import date, datetime

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from finance_ai.agents.llm_factory import create_chat_model, create_ocr_chat_model
from finance_ai.agents.router_agent import orchestrate_query
from finance_ai.agents.stream_utils import StreamEvent, orchestrate_query_stream
from finance_ai.core.config import get_settings
from finance_ai.core.logging import get_logger
from finance_ai.database.crud.watched_asset_crud import WatchedAssetCRUD
from finance_ai.database.session import create_database_engine, create_session_factory
from finance_ai.tools.bank_statement_parser import parse_bank_statement
from finance_ai.tools.bank_statement_service import bulk_insert_transactions
from finance_ai.tools.receipt_ocr import (
    SUPPORTED_MIME_TYPES,
    ReceiptOcrResult,
    extract_document_fields,
)
from finance_ai.tools.receipt_ocr_service import confirm_receipt_transactions
from finance_ai.tools.expense_constants import VALID_EXPENSE_CATEGORIES
from finance_ai.tools.conversation_service import (
    create_conversation,
    get_recent_history_as_tuples,
    list_user_conversations,
    load_conversation_messages,
    save_assistant_message,
    save_user_message,
    update_conversation_title,
)
from finance_ai.tools.market_data_models import AssetFetchResult
from finance_ai.tools.report_service import generate_financial_report
from finance_ai.tools.scheduler_service import (
    fetch_asset_data_structured,
    get_unread_notifications,
    mark_all_notifications_read,
)
from finance_ai.tools.symbol_guard import (
    SymbolResolutionError,
    resolve_and_validate_symbol,
)

logger = get_logger(__name__)

app = FastAPI(
    title="Personal Finance AI",
    description="Multi-Agent AI system for personal finance management",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.it.kmitl.ac.th", "http://localhost:8080"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine = create_database_engine()
_session_factory = create_session_factory(_engine)
_chat_model: Any = None
_ocr_chat_model: Any = None


def get_chat_model() -> Any:
    """Get or create the cached chat model instance.

    Returns:
        LangChain BaseChatModel instance.
    """
    global _chat_model  # noqa: PLW0603
    if _chat_model is None:
        _chat_model = create_chat_model()
    return _chat_model


def get_ocr_chat_model() -> Any:
    """Get or create the cached OCR (vision) chat model instance.

    Uses the dedicated OCR_* settings, independent of the main LLM provider.

    Returns:
        LangChain BaseChatModel instance configured for OCR.
    """
    global _ocr_chat_model  # noqa: PLW0603
    if _ocr_chat_model is None:
        _ocr_chat_model = create_ocr_chat_model()
    return _ocr_chat_model


def get_session() -> Generator[Session, None, None]:
    """Yield a database session for dependency injection."""
    session = _session_factory()
    try:
        yield session
    finally:
        session.close()


# ──────────────────── Request / Response Models ────────────────────


class ChatRequest(BaseModel):
    """Chat query request."""

    query: str = Field(..., description="User's natural language query")
    user_id: str = Field(..., description="User UUID")
    conversation_id: str | None = Field(default=None, description="Conversation UUID")


class ChatResponse(BaseModel):
    """Chat query response."""

    intent: str = Field(..., description="Classified intent")
    response: str = Field(..., description="Agent response text")


class ConversationSummary(BaseModel):
    """Conversation list item."""

    id: str
    title: str
    message_count: int = 0


class MessageOut(BaseModel):
    """Single conversation message."""

    role: str
    content: str
    intent: str = ""


class CreateConversationRequest(BaseModel):
    """Create new conversation request."""

    user_id: str = Field(..., description="User UUID")
    title: str = Field(default="แชทใหม่", description="Conversation title")


class AssetFetchRequest(BaseModel):
    """Asset fetch request."""

    user_id: str = Field(..., description="User UUID")
    symbol: str = Field(..., description="Ticker symbol")
    fetch_type: Literal["price", "news", "all"] = Field(
        default="price", description="Which data to fetch"
    )


class AssetFetchResponse(BaseModel):
    """Structured immediate fetch response."""

    status: str = Field(default="ok", description="Request status")
    result: AssetFetchResult = Field(..., description="Fetched asset data")


class WatchlistAddRequest(BaseModel):
    """Watchlist add request."""

    user_id: str = Field(..., description="User UUID")
    symbol: str = Field(..., description="Ticker symbol or free text (e.g. 'ptt')")
    name: str = Field(default="", description="Display name")


class WatchlistItemOut(BaseModel):
    """Watchlist row returned by GET /assets/watchlist."""

    id: str = Field(..., description="Asset UUID used by DELETE /assets/watchlist/{id}")
    symbol: str = Field(..., description="Canonical Yahoo symbol (e.g. 'PTT.BK')")
    name: str = Field(default="", description="Display name")


class RiskAssessmentSubmitRequest(BaseModel):
    """Risk assessment (SEC suitability questionnaire) submission request."""

    user_id: str = Field(..., description="User UUID")
    answers: dict[str, Any] = Field(
        ...,
        description='Answers mapping, e.g. {"1": "ก", "4": ["ก", "ง"], "11": "ก"}',
    )


class ImportResult(BaseModel):
    """Bank statement import result."""

    inserted: int = Field(..., description="Number of transactions inserted")
    total: int = Field(..., description="Total transactions parsed")


class ReceiptDraftOut(BaseModel):
    """A single draft extracted from a document (pre-confirmation)."""

    transaction_type: str = Field(..., description="'income' or 'expense'")
    amount: str = Field(..., description="Amount in THB (string for Decimal)")
    transaction_date: str = Field(..., description="ISO date YYYY-MM-DD")
    description: str = Field(default="", description="Description text")
    category: str = Field(default="other", description="Category key")
    income_type: str = Field(default="other", description="Income type key")
    withholding_tax: str = Field(default="0", description="Withholding tax THB")
    employer_name: str | None = Field(default=None, description="Employer name")
    confidence: float = Field(default=0.0, description="Model confidence 0-1")


class ReceiptScanResponse(BaseModel):
    """Response for receipt scan (drafts only, not yet persisted)."""

    drafts: list[ReceiptDraftOut] = Field(default_factory=list)
    total: int = Field(default=0, description="Number of drafts extracted")


class ConfirmTransactionInput(BaseModel):
    """A single user-confirmed transaction ready to persist."""

    transaction_type: str = Field(..., description="'income' or 'expense'")
    amount: Decimal = Field(..., gt=0, description="Amount in THB (must be > 0)")
    transaction_date: str = Field(..., description="ISO date YYYY-MM-DD")
    description: str = Field(default="", description="Description text")
    category: str = Field(default="other", description="Category key")
    income_type: str = Field(default="other", description="Income type key")
    withholding_tax: Decimal = Field(default=Decimal("0"), description="Withholding tax THB")
    employer_name: str | None = Field(default=None, description="Employer name")

    @field_validator("transaction_type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        """Ensure transaction_type is income or expense."""
        if value not in ("income", "expense"):
            raise ValueError(f"transaction_type must be 'income' or 'expense'. Received: '{value}'")
        return value

    @field_validator("category")
    @classmethod
    def _validate_category(cls, value: str) -> str:
        """Ensure category is a known key."""
        if value not in VALID_EXPENSE_CATEGORIES:
            raise ValueError(
                f"category must be one of {VALID_EXPENSE_CATEGORIES}. Received: '{value}'"
            )
        return value


class ConfirmTransactionRequest(BaseModel):
    """Request body for confirming a batch of drafts."""

    user_id: str = Field(..., description="User UUID")
    transactions: list[ConfirmTransactionInput] = Field(
        default_factory=list, description="Confirmed transactions to persist"
    )


# ──────────────────────────── Chat ────────────────────────────────


@app.post("/chat", response_model=ChatResponse)  # type: ignore[misc]
def chat(req: ChatRequest, session: Session = Depends(get_session)) -> ChatResponse:
    """Process a chat query (non-streaming).

    Args:
        req: Chat request with query and user_id.
        session: Database session.

    Returns:
        ChatResponse with intent and response text.
    """
    history = _load_history(session, req.conversation_id)
    save_user_message(session, req.conversation_id or "", req.query)

    result = orchestrate_query(
        query=req.query,
        chat_model=get_chat_model(),
        user_id=req.user_id,
        db_session_factory=_session_factory,
        chat_history=history,
    )

    save_assistant_message(session, req.conversation_id or "", result["response"], result["intent"])
    return ChatResponse(intent=result["intent"], response=result["response"])


@app.get("/chat/stream")  # type: ignore[misc]
def chat_stream(
    query: str,
    user_id: str,
    conversation_id: str | None = None,
    session: Session = Depends(get_session),
) -> StreamingResponse:
    """Process a chat query with SSE streaming.

    Args:
        query: User's natural language query.
        user_id: User UUID.
        conversation_id: Optional conversation UUID.
        session: Database session.

    Returns:
        StreamingResponse with Server-Sent Events.
    """
    history = _load_history(session, conversation_id)
    save_user_message(session, conversation_id or "", query)

    return StreamingResponse(
        _stream_generator(query, user_id, history, conversation_id, session),
        media_type="text/event-stream",
    )


def _stream_generator(
    query: str,
    user_id: str,
    history: list[tuple[str, str]],
    conversation_id: str | None,
    session: Session,
) -> Generator[str, None, None]:
    """Generate SSE events from the streaming agent.

    Args:
        query: User query.
        user_id: User UUID.
        history: Chat history tuples.
        conversation_id: Conversation UUID.
        session: Database session.

    Yields:
        SSE-formatted event strings.
    """
    full_response = ""
    intent = "unknown"

    for event in orchestrate_query_stream(
        query=query,
        chat_model=get_chat_model(),
        user_id=user_id,
        db_session_factory=_session_factory,
        chat_history=history,
    ):
        sse_data = _format_sse_event(event)
        if sse_data:
            yield sse_data
        if event.event_type == "complete":
            intent = event.intent
        if event.event_type == "token":
            full_response += event.content

    save_assistant_message(session, conversation_id or "", full_response, intent)


def _format_sse_event(event: StreamEvent) -> str | None:
    """Format a StreamEvent as an SSE string.

    Args:
        event: StreamEvent to format.

    Returns:
        SSE-formatted string, or None for status events.
    """
    if event.event_type == "token":
        # JSON-encode so newlines and Thai characters inside content survive
        # the SSE transport. A raw newline in a `data:` field would terminate
        # the event early and drop the rest of the token.
        return f"data: {json.dumps(event.content, ensure_ascii=False)}\n\n"
    if event.event_type == "complete":
        return "event: complete\n" f"data: {json.dumps(event.intent, ensure_ascii=False)}\n\n"
    return None


def _load_history(
    session: Session,
    conversation_id: str | None,
) -> list[tuple[str, str]]:
    """Load chat history for a conversation.

    Args:
        session: Database session.
        conversation_id: Conversation UUID.

    Returns:
        List of (role, content) tuples.
    """
    if not conversation_id:
        return []
    return get_recent_history_as_tuples(session, conversation_id)


# ──────────────────────── Conversations ──────────────────────────


@app.get("/conversations", response_model=list[ConversationSummary])  # type: ignore[misc]
def list_conversations(
    user_id: str, session: Session = Depends(get_session)
) -> list[ConversationSummary]:
    """List conversations for a user.

    Args:
        user_id: User UUID.
        session: Database session.

    Returns:
        List of conversation summaries.
    """
    convs = list_user_conversations(session, user_id, limit=50)
    return [ConversationSummary(id=str(conv.id), title=conv.title or "แชทใหม่") for conv in convs]


@app.post("/conversations", response_model=ConversationSummary)  # type: ignore[misc]
def create_new_conv(
    req: CreateConversationRequest, session: Session = Depends(get_session)
) -> ConversationSummary:
    """Create a new conversation.

    Args:
        req: Create conversation request.
        session: Database session.

    Returns:
        Created conversation summary.
    """
    conv = create_conversation(session, req.user_id, req.title)
    return ConversationSummary(id=str(conv.id), title=conv.title or "แชทใหม่")


@app.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])  # type: ignore[misc]
def get_messages(conversation_id: str, session: Session = Depends(get_session)) -> list[MessageOut]:
    """Get all messages in a conversation.

    Args:
        conversation_id: Conversation UUID.
        session: Database session.

    Returns:
        List of messages.
    """
    messages = load_conversation_messages(session, conversation_id)
    return [
        MessageOut(
            role=msg.get("role", "user"),
            content=msg.get("content", ""),
            intent=msg.get("intent", ""),
        )
        for msg in messages
    ]


# ──────────────────────── Upload ─────────────────────────────────


@app.post("/upload/bank-statement", response_model=ImportResult)  # type: ignore[misc]
async def upload_bank_statement(
    user_id: str,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> ImportResult:
    """Upload and import a bank statement (CSV or XLSX).

    Args:
        user_id: User UUID.
        file: Uploaded file.
        session: Database session.

    Returns:
        ImportResult with inserted and total counts.
    """
    content = await file.read()
    file_type = _detect_file_type(file.filename or "")
    transactions = parse_bank_statement(content, file_type)

    if not transactions:
        return ImportResult(inserted=0, total=0)

    count = bulk_insert_transactions(transactions, user_id, _session_factory)
    return ImportResult(inserted=count, total=len(transactions))


def _detect_file_type(filename: str) -> str:
    """Detect file type from filename.

    Args:
        filename: Name of the uploaded file.

    Returns:
        'csv' or 'xlsx'.
    """
    if filename.lower().endswith(".xlsx"):
        return "xlsx"
    return "csv"


def _detect_receipt_mime(file: UploadFile) -> str:
    """Detect the MIME type of an uploaded receipt image/PDF.

    Prefers the client-provided content_type; falls back to extension.

    Args:
        file: Uploaded file.

    Returns:
        MIME type string.

    Raises:
        ValueError: If the type is not a supported image/PDF.
    """
    mime = (file.content_type or "").lower()
    if not mime:
        mime = _mime_from_filename(file.filename or "")
    if mime not in SUPPORTED_MIME_TYPES:
        raise ValueError(f"Unsupported file type: '{mime}'. Supported: images and PDF.")
    return mime


def _mime_from_filename(filename: str) -> str:
    """Infer a MIME type from a filename extension.

    Args:
        filename: Name of the uploaded file.

    Returns:
        MIME type string (empty string if unknown).
    """
    lower = filename.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith((".heic", ".heif")):
        return "image/heic"
    if lower.endswith(".pdf"):
        return "application/pdf"
    return ""


def _draft_to_out(draft: ReceiptOcrResult) -> ReceiptDraftOut:
    """Convert a ReceiptOcrResult into a JSON-safe ReceiptDraftOut.

    Args:
        draft: Parsed draft from the OCR step.

    Returns:
        ReceiptDraftOut with Decimal/date fields serialized as strings.
    """
    return ReceiptDraftOut(
        transaction_type=draft.transaction_type,
        amount=str(draft.amount),
        transaction_date=draft.transaction_date.isoformat(),
        description=draft.description,
        category=draft.category,
        income_type=draft.income_type,
        withholding_tax=str(draft.withholding_tax),
        employer_name=draft.employer_name,
        confidence=draft.confidence,
    )


@app.post("/upload/receipt", response_model=ReceiptScanResponse)  # type: ignore[misc]
async def upload_receipt(
    user_id: str,
    file: UploadFile = File(...),
) -> ReceiptScanResponse:
    """Scan a financial document image/PDF and return drafts (not persisted).

    The model pre-fills transaction fields; the user must review and
    confirm via /transactions/confirm before anything is saved.

    Args:
        user_id: User UUID.
        file: Uploaded image or PDF.

    Returns:
        ReceiptScanResponse with extracted drafts.

    Raises:
        HTTPException: 400 if the file type is unsupported;
                       503 if the AI provider is misconfigured or fails.
    """
    try:
        mime_type = _detect_receipt_mime(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    content = await file.read()
    try:
        drafts = await _extract_drafts_async(content, mime_type)
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"AI ไม่พร้อมใช้งาน: ติดตั้ง/เปิดใช้ provider ไม่ได้: {exc}",
        ) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="AI อ่านเอกสารนานเกินไป กรุณาลองอีกครั้ง",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"AI ไม่สามารถอ่านเอกสารได้: {exc}",
        ) from exc

    return ReceiptScanResponse(
        drafts=[_draft_to_out(d) for d in drafts],
        total=len(drafts),
    )


def _extract_drafts(
    content: bytes,
    mime_type: str,
) -> list[ReceiptOcrResult]:
    """Extract transaction drafts from the document bytes using the OCR model.

    Uses the dedicated OCR vision model (OCR_* settings), separate from the
    main chat model, so document scanning can run on a vision-capable model.

    Args:
        content: Raw uploaded file bytes.
        mime_type: Detected MIME type.

    Returns:
        List of parsed drafts.
    """
    return extract_document_fields(content, mime_type, get_ocr_chat_model())


async def _extract_drafts_async(
    content: bytes,
    mime_type: str,
) -> list[ReceiptOcrResult]:
    """Run OCR in a worker thread with a server-side timeout.

    The vision-model call is blocking; running it off the event loop keeps
    the API responsive and lets us enforce ``ocr_timeout``. A timeout is
    surfaced to the caller as an ``asyncio.TimeoutError`` (mapped to 504).

    Args:
        content: Raw uploaded file bytes.
        mime_type: Detected MIME type.

    Returns:
        List of parsed drafts.

    Raises:
        asyncio.TimeoutError: If OCR takes longer than ``ocr_timeout``.
    """
    timeout = get_settings().ocr_timeout
    return await asyncio.wait_for(
        asyncio.to_thread(_extract_drafts, content, mime_type),
        timeout=timeout,
    )


def _input_to_draft(
    item: ConfirmTransactionInput,
) -> ReceiptOcrResult:
    """Convert a confirmed input into a ReceiptOcrResult for persistence.

    Args:
        item: User-confirmed transaction input.

    Returns:
        ReceiptOcrResult ready for confirm_receipt_transactions.
    """
    return ReceiptOcrResult(
        transaction_type=item.transaction_type,
        amount=item.amount,
        transaction_date=datetime.strptime(item.transaction_date, "%Y-%m-%d").date(),
        description=item.description,
        category=item.category,
        income_type=item.income_type,
        withholding_tax=item.withholding_tax,
        employer_name=item.employer_name,
    )


@app.post("/transactions/confirm", response_model=ImportResult)  # type: ignore[misc]
def confirm_transactions(req: ConfirmTransactionRequest) -> ImportResult:
    """Persist user-confirmed transaction drafts to the database.

    Args:
        req: Confirm request with user_id and confirmed transactions.

    Returns:
        ImportResult with inserted and total counts.
    """
    drafts = [_input_to_draft(item) for item in req.transactions]
    result = confirm_receipt_transactions(drafts, req.user_id, _session_factory)
    return ImportResult(inserted=result.inserted, total=result.total)


# ──────────────────────── Dashboard ──────────────────────────────


@app.get("/dashboard")  # type: ignore[misc]
def get_dashboard(
    user_id: str,
    year: int | None = None,
    month: int | None = None,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Get financial dashboard data.

    Args:
        user_id: User UUID.
        year: Optional year filter.
        month: Optional month filter.
        session: Database session.

    Returns:
        Financial report dictionary.
    """
    today = date.today()
    report = generate_financial_report(
        session=session,
        user_id=user_id,
        year=year if year is not None else today.year,
        month=month if month is not None else today.month,
    )
    return report.model_dump(mode="json")


# ──────────────────────── Assets ─────────────────────────────────


@app.get("/assets/search")
def search_assets(query: str) -> dict[str, Any]:
    """Search asset symbols by free text (names, tickers, transliterations).

    Args:
        query: Free-text asset name or ticker (e.g. "ปตท", "Apple").

    Returns:
        Dict with candidate results (symbol/name/exchange/type keys).

    Raises:
        HTTPException: 422 when the query is empty or whitespace-only.
    """
    if not query.strip():
        raise HTTPException(
            status_code=422,
            detail="กรุณาระบุคำค้นหา (Query must not be empty)",
        )
    from finance_ai.tools.symbol_search_service import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        search_asset_symbols,
    )

    matches = search_asset_symbols(query.strip())
    results = [
        {
            "symbol": match.symbol,
            "name": match.name,
            "exchange": match.exchange,
            "type": match.quote_type,
        }
        for match in matches
    ]
    return {"status": "ok", "results": results}


@app.post("/assets/fetch", response_model=AssetFetchResponse)  # type: ignore[misc]
def fetch_asset_data(req: AssetFetchRequest) -> AssetFetchResponse:
    """Fetch immediate asset data (price/news) without creating a notification.

    Only scheduled fetches create notifications; this endpoint returns the
    structured result directly.

    Args:
        req: Asset fetch request (symbol and fetch_type).

    Returns:
        AssetFetchResponse wrapping the structured fetch result.
    """
    result = fetch_asset_data_structured(req.symbol, req.fetch_type)
    return AssetFetchResponse(status="ok", result=result)


@app.get("/assets/notifications")  # type: ignore[misc]
def get_asset_notifications(
    user_id: str, session: Session = Depends(get_session)
) -> list[dict[str, Any]]:
    """Get unread asset notifications for a user.

    Args:
        user_id: User UUID.
        session: Database session.

    Returns:
        List of notification dictionaries.
    """
    notifications = get_unread_notifications(session, user_id)
    return [
        {
            "id": str(n.id),
            "symbol": n.symbol,
            "message": n.content,
            "created_at": _format_thai_datetime(n.created_at),
        }
        for n in notifications
    ]


@app.post("/assets/notifications/read")  # type: ignore[misc]
def mark_notifications_read(
    user_id: str, session: Session = Depends(get_session)
) -> dict[str, str]:
    """Mark all notifications as read for a user.

    Args:
        user_id: User UUID.
        session: Database session.

    Returns:
        Status dictionary.
    """
    mark_all_notifications_read(session, user_id)
    return {"status": "ok"}


_THAI_MONTH_ABBREVIATIONS: tuple[str, ...] = (
    "",
    "ม.ค.",
    "ก.พ.",
    "มี.ค.",
    "เม.ย.",
    "พ.ค.",
    "มิ.ย.",
    "ก.ค.",
    "ส.ค.",
    "ก.ย.",
    "ต.ค.",
    "พ.ย.",
    "ธ.ค.",
)


def _format_thai_datetime(value: datetime) -> str:
    """Format a datetime as a compact Thai string ('13 ก.ย. 2026, 17:45').

    Uses the Gregorian year with Thai month abbreviations, matching Thai
    UI conventions for notification timestamps.

    Args:
        value: Datetime to format (naive or timezone-aware).

    Returns:
        Formatted string 'D MMM YYYY, HH:MM'.

    Example:
        >>> _format_thai_datetime(datetime(2026, 9, 13, 17, 45))
        '13 ก.ย. 2026, 17:45'
    """
    month = _THAI_MONTH_ABBREVIATIONS[value.month]
    return f"{value.day} {month} {value.year}, {value.hour:02d}:{value.minute:02d}"


@app.get("/assets/watchlist", response_model=list[WatchlistItemOut])  # type: ignore[misc]
def get_watchlist(user_id: str, session: Session = Depends(get_session)) -> list[WatchlistItemOut]:
    """List the user's watched assets (canonical symbol + display name).

    Args:
        user_id: User UUID.
        session: Database session.

    Returns:
        List of watched assets, oldest first.
    """
    assets = WatchedAssetCRUD().list_for_user(session, user_id)
    return [
        WatchlistItemOut(id=str(asset.id), symbol=asset.symbol, name=asset.name) for asset in assets
    ]


@app.post("/assets/watchlist")  # type: ignore[misc]
def add_watchlist_asset(
    req: WatchlistAddRequest, session: Session = Depends(get_session)
) -> dict[str, str]:
    """Add a symbol to the user's watchlist after validation.

    The input is resolved to a canonical Yahoo symbol via the shared guard;
    unresolvable input is rejected with HTTP 422 and a Thai message.
    Re-adding the same canonical symbol is idempotent.

    Args:
        req: Add request (user_id, symbol, optional name).
        session: Database session.

    Returns:
        {"status": "added"|"already_exists", "symbol", "name"}.

    Raises:
        HTTPException: 422 when the symbol cannot be resolved.
    """
    try:
        canonical_symbol = resolve_and_validate_symbol(req.symbol)
    except SymbolResolutionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    record = WatchedAssetCRUD().add(session, req.user_id, canonical_symbol, req.name)
    session.commit()
    status = "already_exists" if record is None else "added"
    return {"status": status, "symbol": canonical_symbol, "name": req.name}


@app.delete("/assets/watchlist/{asset_id}")  # type: ignore[misc]
def delete_watchlist_asset(
    asset_id: str, user_id: str, session: Session = Depends(get_session)
) -> dict[str, str]:
    """Remove an asset from the user's watchlist.

    The path value may be the canonical symbol (e.g. "PTT.BK") or the
    asset's UUID; both are resolved within the requesting user's list.

    Args:
        asset_id: Canonical symbol or asset UUID.
        user_id: User UUID.
        session: Database session.

    Returns:
        {"status": "ok"}.

    Raises:
        HTTPException: 404 when the asset is not in the user's watchlist.
    """
    if _delete_watchlist_asset(session, WatchedAssetCRUD(), asset_id, user_id):
        session.commit()
        return {"status": "ok"}
    raise HTTPException(
        status_code=404,
        detail=f"ไม่พบ '{asset_id}' ในรายการสินทรัพย์ที่ติดตามของผู้ใช้",
    )


def _delete_watchlist_asset(
    session: Session, crud: WatchedAssetCRUD, asset_id: str, user_id: str
) -> bool:
    """Delete a watched asset matched by symbol first, then by asset id.

    Args:
        session: Database session.
        crud: WatchedAssetCRUD instance.
        asset_id: Canonical symbol or asset UUID from the request path.
        user_id: UUID of the requesting user.

    Returns:
        True when a row owned by the user was deleted.
    """
    asset = crud.get_by_symbol(session, user_id, asset_id)
    if asset is not None:
        return crud.delete(session, asset.id, user_id)
    return crud.delete(session, asset_id, user_id)


# ──────────────────────── Risk Assessment ────────────────────────


@app.post("/risk-assessment/submit")
def submit_risk_assessment(
    req: RiskAssessmentSubmitRequest, session: Session = Depends(get_session)
) -> dict[str, Any]:
    """Validate, score, and store the user's SEC risk assessment.

    Args:
        req: Submission with user_id and the answers mapping.
        session: Database session.

    Returns:
        Dict with total_score, risk_level, risk_category and the example
        asset allocation for the resulting level.

    Raises:
        HTTPException: 422 when any answer is missing or invalid.
    """
    from finance_ai.tools.risk_assessment_service import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        calculate_total_score,
        determine_risk_level,
        validate_answers,
    )

    try:
        validate_answers(req.answers)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    total_score = calculate_total_score(req.answers)
    risk_level, risk_category = determine_risk_level(total_score)
    _save_risk_assessment(session, req, total_score, risk_level, risk_category)
    return {
        "status": "ok",
        "total_score": total_score,
        "risk_level": risk_level,
        "risk_category": risk_category,
        "allocation": _allocation_payload(risk_level),
    }


@app.get("/risk-assessment/latest")
def get_latest_risk_assessment(
    user_id: str, session: Session = Depends(get_session)
) -> dict[str, Any]:
    """Return the user's latest risk assessment, or None when never taken.

    Args:
        user_id: User UUID.
        session: Database session.

    Returns:
        {"status": "ok", "assessment": {...} | None}.
    """
    from finance_ai.database.crud.risk_assessment_crud import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        RiskAssessmentCRUD,
    )

    record = RiskAssessmentCRUD().get_latest_by_user(session, user_id)
    if record is None:
        return {"status": "ok", "assessment": None}
    return {
        "status": "ok",
        "assessment": {
            "total_score": record.total_score,
            "risk_level": record.risk_level,
            "risk_category": record.risk_category,
            "created_at": str(record.created_at),
        },
    }


def _save_risk_assessment(
    session: Session,
    req: RiskAssessmentSubmitRequest,
    total_score: int,
    risk_level: int,
    risk_category: str,
) -> None:
    """Persist the user's assessment result and commit the session.

    Args:
        session: Database session.
        req: The original submit request (user_id + answers).
        total_score: Computed score over the scored questions (10-40).
        risk_level: Resulting risk level (1-5).
        risk_category: Thai investor category name.
    """
    from finance_ai.database.crud.risk_assessment_crud import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        RiskAssessmentCRUD,
    )
    from finance_ai.database.models.risk_assessment import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        RiskAssessment,
    )

    assessment = RiskAssessment(
        user_id=req.user_id,
        answers=req.answers,
        total_score=total_score,
        risk_level=risk_level,
        risk_category=risk_category,
    )
    RiskAssessmentCRUD().create_assessment(session, assessment)
    session.commit()


def _allocation_payload(risk_level: int) -> dict[str, Any]:
    """Build the example asset-allocation payload for a risk level.

    Args:
        risk_level: Risk level 1-5.

    Returns:
        Dict with column labels, the percentage row, and the footnote.
    """
    from finance_ai.tools.risk_assessment_constants import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        ALLOCATION_FOOTNOTE,
    )
    from finance_ai.tools.risk_assessment_service import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        get_allocation,
    )

    columns, row = get_allocation(risk_level)
    return {"columns": list(columns), "row": list(row), "footnote": ALLOCATION_FOOTNOTE}


# ──────────────────────── Health ─────────────────────────────────


@app.get("/health")  # type: ignore[misc]
def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Health status dictionary.
    """
    return {"status": "healthy", "service": "personal-finance-ai"}


_static_dir = Path(__file__).parent / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="web")
