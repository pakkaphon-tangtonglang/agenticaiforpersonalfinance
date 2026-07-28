"""FastAPI application for Personal Finance AI.

Provides REST API endpoints for chat, conversations, bank statement
upload, dashboard, asset monitoring, and evaluation.

Run with: make dev (development) or make run (production)
"""

import json

from collections.abc import Generator
from typing import Any

from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from finance_ai.agents.llm_factory import create_chat_model
from finance_ai.agents.router_agent import orchestrate_query
from finance_ai.agents.stream_utils import StreamEvent, orchestrate_query_stream
from finance_ai.core.config import get_settings
from finance_ai.core.logging import get_logger
from finance_ai.database.session import create_database_engine, create_session_factory
from finance_ai.tools.bank_statement_parser import parse_bank_statement
from finance_ai.tools.bank_statement_service import bulk_insert_transactions
from finance_ai.tools.conversation_service import (
    create_conversation,
    get_recent_history_as_tuples,
    list_user_conversations,
    load_conversation_messages,
    save_assistant_message,
    save_user_message,
    update_conversation_title,
)
from finance_ai.tools.report_service import generate_financial_report
from finance_ai.tools.scheduler_service import (
    execute_immediate_fetch,
    get_unread_notifications,
    mark_all_notifications_read,
)

logger = get_logger(__name__)

app = FastAPI(
    title="Personal Finance AI",
    description="Multi-Agent AI system for personal finance management",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine = create_database_engine()
_session_factory = create_session_factory(_engine)
_chat_model: Any = None


def get_chat_model() -> Any:
    """Get or create the cached chat model instance.

    Returns:
        LangChain BaseChatModel instance.
    """
    global _chat_model  # noqa: PLW0603
    if _chat_model is None:
        _chat_model = create_chat_model()
    return _chat_model


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
    fetch_type: str = Field(default="price", description="'price' or 'news'")


class ImportResult(BaseModel):
    """Bank statement import result."""

    inserted: int = Field(..., description="Number of transactions inserted")
    total: int = Field(..., description="Total transactions parsed")


# ──────────────────────────── Chat ────────────────────────────────


@app.post("/chat", response_model=ChatResponse)
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


@app.get("/chat/stream")
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


@app.get("/conversations", response_model=list[ConversationSummary])
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


@app.post("/conversations", response_model=ConversationSummary)
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


@app.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
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


@app.post("/upload/bank-statement", response_model=ImportResult)
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


# ──────────────────────── Dashboard ──────────────────────────────


@app.get("/dashboard")
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
    report = generate_financial_report(
        session=session,
        user_id=user_id,
        year=year,
        month=month,
    )
    return report.model_dump(mode="json") if hasattr(report, "model_dump") else report


# ──────────────────────── Assets ─────────────────────────────────


@app.post("/assets/fetch")
def fetch_asset_data(
    req: AssetFetchRequest, session: Session = Depends(get_session)
) -> dict[str, Any]:
    """Fetch immediate asset data (price or news).

    Args:
        req: Asset fetch request.
        session: Database session.

    Returns:
        Fetch result summary.
    """
    result = execute_immediate_fetch(
        session=session,
        user_id=req.user_id,
        symbol=req.symbol,
        fetch_type=req.fetch_type,
    )
    return {"status": "ok", "result": result}


@app.get("/assets/notifications")
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
            "message": n.message,
            "created_at": str(n.created_at),
        }
        for n in notifications
    ]


@app.post("/assets/notifications/read")
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


# ──────────────────────── Evaluation ─────────────────────────────


@app.post("/evaluation/run")
def run_evaluation(
    dimensions: list[str] | None = None,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Run the evaluation framework.

    Args:
        dimensions: Optional list of dimensions to evaluate.
        session: Database session.

    Returns:
        Evaluation results dictionary.
    """
    from finance_ai.evaluation.runner import EvaluationRunner  # noqa: PLC0415

    runner = EvaluationRunner(chat_model=get_chat_model(), session_factory=_session_factory)
    results = runner.run_all(dimensions=dimensions)
    return {"status": "ok", "results": results}


# ──────────────────────── Health ─────────────────────────────────


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Health status dictionary.
    """
    return {"status": "healthy", "service": "personal-finance-ai"}
