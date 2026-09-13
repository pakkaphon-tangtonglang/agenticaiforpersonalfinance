"""Service layer for asset schedule management and job execution.

Handles CRUD operations on asset schedules and executes scheduled
data-fetching jobs via the Bright Data API, storing results as
notifications for the user.
"""

from decimal import Decimal
from typing import Any, Optional

from sqlalchemy.orm import Session

from finance_ai.core.logging import get_logger
from finance_ai.database.crud.schedule_crud import (
    AssetNotificationCRUD,
    AssetScheduleCRUD,
)
from finance_ai.database.models.asset_notification import AssetNotification
from finance_ai.database.models.asset_schedule import AssetSchedule
from finance_ai.tools.market_data_models import AssetFetchResult, NewsItem

logger = get_logger(__name__)

# Structured fetch error messages (immediate /assets/fetch endpoint)
_PRICE_UNAVAILABLE_MESSAGE = "ไม่สามารถดึงราคา {symbol} ได้ในขณะนี้"
_NEWS_UNAVAILABLE_MESSAGE = "ไม่พบข่าวสำหรับ {symbol} ในขณะนี้"
_VALID_FETCH_TYPES = ("price", "news", "all")


def create_schedule(
    session: Session,
    user_id: str,
    symbol: str,
    description: str,
    cron_expression: str,
    max_runs: Optional[int] = None,
) -> AssetSchedule:
    """Create a new asset monitoring schedule.

    Args:
        session: SQLAlchemy session.
        user_id: UUID of the user.
        symbol: Ticker symbol to monitor.
        description: Human-readable description.
        cron_expression: Cron expression for timing.
        max_runs: Max executions (None = unlimited).

    Returns:
        Created AssetSchedule instance.

    Example:
        >>> schedule = create_schedule(session, "u1", "GC=F", "ราคาทอง", "0 21 * * *")
    """
    crud = AssetScheduleCRUD()
    return crud.create(session, user_id, symbol, description, cron_expression, max_runs)


def get_user_schedules(
    session: Session,
    user_id: str,
    active_only: bool = False,
) -> list[AssetSchedule]:
    """Get schedules for a user.

    Args:
        session: SQLAlchemy session.
        user_id: UUID of the user.
        active_only: If True, return only active schedules.

    Returns:
        List of AssetSchedule instances.

    Example:
        >>> schedules = get_user_schedules(session, "u1", active_only=True)
    """
    crud = AssetScheduleCRUD()
    if active_only:
        return crud.get_active_by_user(session, user_id)
    return crud.get_by_user(session, user_id)


def delete_schedule(
    session: Session,
    schedule_id: str,
    user_id: str,
) -> bool:
    """Delete a schedule by ID (must belong to user).

    Args:
        session: SQLAlchemy session.
        schedule_id: UUID of the schedule.
        user_id: UUID of the user.

    Returns:
        True if deleted, False if not found.

    Example:
        >>> delete_schedule(session, "sched-1", "u1")
        True
    """
    crud = AssetScheduleCRUD()
    return crud.delete(session, schedule_id, user_id)


def deactivate_schedule(
    session: Session,
    schedule_id: str,
    user_id: str,
) -> bool:
    """Deactivate a schedule by ID.

    Args:
        session: SQLAlchemy session.
        schedule_id: UUID of the schedule.
        user_id: UUID of the user.

    Returns:
        True if deactivated, False if not found.

    Example:
        >>> deactivate_schedule(session, "sched-1", "u1")
        True
    """
    crud = AssetScheduleCRUD()
    return crud.deactivate(session, schedule_id, user_id)


def execute_scheduled_fetch(
    session: Session,
    schedule: AssetSchedule,
) -> AssetNotification:
    """Execute a scheduled data fetch and store the result as a notification.

    Fetches price + news for the schedule's symbol, creates a notification,
    increments run_count, and auto-deactivates if max_runs reached.

    Args:
        session: SQLAlchemy session.
        schedule: The AssetSchedule to execute.

    Returns:
        Created AssetNotification with fetched data.

    Example:
        >>> notification = execute_scheduled_fetch(session, schedule)
    """
    content = _fetch_asset_summary(schedule.symbol)
    notif_crud = AssetNotificationCRUD()
    notification = notif_crud.create(
        session,
        user_id=schedule.user_id,
        symbol=schedule.symbol,
        content=content,
        schedule_id=schedule.id,
    )

    sched_crud = AssetScheduleCRUD()
    new_count = sched_crud.increment_run_count(session, schedule.id)
    if schedule.max_runs and new_count >= schedule.max_runs:
        sched_crud.deactivate(session, schedule.id, schedule.user_id)
        _auto_unregister(schedule.id)

    return notification


def _auto_unregister(schedule_id: str) -> None:
    """Unregister a schedule from APScheduler after max_runs reached.

    Args:
        schedule_id: UUID of the schedule to unregister.
    """
    from finance_ai.tools.background_scheduler import (  # noqa: PLC0415
        unregister_schedule,
    )

    unregister_schedule(schedule_id)
    logger.info("Auto-deactivated schedule %s (max_runs reached)", schedule_id)


def fetch_asset_data_structured(
    symbol: str,
    fetch_type: str = "all",
) -> AssetFetchResult:
    """Fetch structured price/news data without creating a notification.

    Used by the immediate /assets/fetch endpoint. Only scheduled fetches
    create AssetNotification rows.

    Args:
        symbol: Ticker symbol (e.g., "aapl", "PTT.BK"); uppercased.
        fetch_type: "price", "news", or "all".

    Returns:
        AssetFetchResult with the formatted price, currency, parsed news,
        and a combined Thai error message when data is unavailable.

    Raises:
        ValueError: If fetch_type is not one of price/news/all.

    Example:
        >>> result = fetch_asset_data_structured("PTT.BK", "price")
        >>> result.price
        '35.50'
    """
    canonical_symbol = symbol.strip().upper()
    if fetch_type not in _VALID_FETCH_TYPES:
        raise ValueError(
            f"Invalid fetch_type: '{fetch_type}'. Expected one of {_VALID_FETCH_TYPES}."
        )
    price, currency = _fetch_price_fields(canonical_symbol, fetch_type)
    news = _fetch_news_field(canonical_symbol, fetch_type)
    return AssetFetchResult(
        symbol=canonical_symbol,
        price=_format_price(price),
        currency=currency,
        news=news,
        error=_build_fetch_error(canonical_symbol, fetch_type, price, news),
    )


def _fetch_price_fields(
    symbol: str,
    fetch_type: str,
) -> tuple[Optional[Decimal], Optional[str]]:
    """Fetch price and currency for "price"/"all" fetch types.

    Args:
        symbol: Canonical ticker symbol.
        fetch_type: Requested fetch type ("price" or "all").

    Returns:
        (price, currency) tuple; currency is only resolved when a price
        exists. Both None when a news-only fetch was requested.
    """
    if fetch_type not in ("price", "all"):
        return None, None
    from finance_ai.tools.price_client import (  # noqa: PLC0415
        fetch_currency,
        fetch_current_price,
    )

    price = fetch_current_price(symbol)
    currency = fetch_currency(symbol) if price is not None else None
    return price, currency


def _fetch_news_field(symbol: str, fetch_type: str) -> list[NewsItem]:
    """Fetch parsed news items for "news"/"all" fetch types.

    Args:
        symbol: Canonical ticker symbol.
        fetch_type: Requested fetch type ("news" or "all").

    Returns:
        Parsed news items, or [] when news was not requested or none found.
    """
    if fetch_type not in ("news", "all"):
        return []
    from finance_ai.tools.market_data_service import (  # noqa: PLC0415
        fetch_news_items,
    )

    return fetch_news_items(symbol)


def _build_fetch_error(
    symbol: str,
    fetch_type: str,
    price: Optional[Decimal],
    news: list[NewsItem],
) -> Optional[str]:
    """Build the combined Thai error message for unavailable data.

    Args:
        symbol: Canonical ticker symbol.
        fetch_type: Requested fetch type.
        price: Fetched price, or None when unavailable.
        news: Fetched news items, or [] when unavailable.

    Returns:
        Combined Thai message, or None when all requested data was fetched.
    """
    errors: list[str] = []
    if fetch_type in ("price", "all") and price is None:
        errors.append(_PRICE_UNAVAILABLE_MESSAGE.format(symbol=symbol))
    if fetch_type in ("news", "all") and not news:
        errors.append(_NEWS_UNAVAILABLE_MESSAGE.format(symbol=symbol))
    if not errors:
        return None
    return "; ".join(errors)


def _format_price(price: Optional[Decimal]) -> Optional[str]:
    """Format a price Decimal as a fixed 2-decimal string.

    Args:
        price: Price value, or None.

    Returns:
        Formatted string (e.g., "35.50"), or None.

    Example:
        >>> _format_price(Decimal("35.5"))
        '35.50'
    """
    if price is None:
        return None
    return str(price.quantize(Decimal("0.01")))


def _fetch_asset_summary(
    symbol: str,
    chat_model: Any = None,
) -> str:
    """Fetch asset price and news, format a combined summary.

    Args:
        symbol: Ticker symbol to fetch.
        chat_model: Optional ChatModel for LLM summarization.

    Returns:
        Formatted summary text with price and news in Thai.
    """
    from finance_ai.tools.market_data_service import (  # noqa: PLC0415
        fetch_finance_news,
        format_price_with_unit,
    )
    from finance_ai.tools.price_client import (  # noqa: PLC0415
        fetch_current_price,
    )

    price = fetch_current_price(symbol)
    price_text = (
        f"💰 **{symbol}**: ราคาปัจจุบัน {format_price_with_unit(price, symbol)}"
        if price
        else f"ไม่สามารถดึงราคา {symbol} ได้ในขณะนี้"
    )

    news_result = fetch_finance_news(symbol)
    if not news_result.has_news:
        return price_text

    if chat_model is not None:
        news_summary = _summarize_with_llm(
            chat_model,
            symbol,
            news_result.news_content,
        )
        return f"{price_text}\n\n{news_summary}"
    return f"{price_text}\n\n📰 **ข่าวล่าสุด**\n{news_result.news_content}"


_NEWS_SUMMARY_PROMPT: str = (
    "สรุปข่าวเกี่ยวกับ {symbol} จากข้อมูลด้านล่าง เป็นภาษาไทย\n"
    "ตอบเนื้อหาตรงๆ ห้ามขึ้นต้นด้วยคำนำหรือประโยคแนะนำตัว\n\n"
    "รูปแบบ:\n"
    "### [หัวข้อข่าว] (แหล่งที่มา)\n"
    "**สิ่งที่เกิดขึ้น:** รายละเอียดครบถ้วน\n"
    "**ตัวเลขสำคัญ:** % การเปลี่ยนแปลง จำนวนเงิน ฯลฯ\n"
    "**ผลกระทบต่อราคา:** บวก / ลบ / กลาง — เหตุผลสั้นๆ\n\n"
    "ท้ายสุดสรุป:\n"
    "---\n"
    "## วิเคราะห์ภาพรวม\n"
    "- Sentiment รวม พร้อมเหตุผล\n"
    "- ปัจจัยที่ต้องจับตา\n"
    "- ⚠️ เป็นการสรุปข่าว ไม่ใช่คำแนะนำการลงทุน\n\n"
    "ข้อมูลข่าว:\n{raw_news}"
)


def _summarize_with_llm(
    chat_model: Any,
    symbol: str,
    raw_news: str,
) -> str:
    """Use LLM to summarize raw news into structured Thai analysis.

    Args:
        chat_model: LangChain ChatModel instance.
        symbol: Ticker symbol for context.
        raw_news: Raw news content from SERP API.

    Returns:
        LLM-summarized news in Thai markdown format.
    """
    prompt = _NEWS_SUMMARY_PROMPT.format(
        symbol=symbol,
        raw_news=raw_news,
    )
    try:
        response = chat_model.invoke(prompt)
        return str(response.content)
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM summarization failed for %s: %s", symbol, exc)
        return f"📰 **ข่าวล่าสุด {symbol}**\n{raw_news}"


def get_unread_notifications(
    session: Session,
    user_id: str,
) -> list[AssetNotification]:
    """Get unread notifications for a user.

    Args:
        session: SQLAlchemy session.
        user_id: UUID of the user.

    Returns:
        List of unread AssetNotification instances.

    Example:
        >>> notifications = get_unread_notifications(session, "u1")
    """
    crud = AssetNotificationCRUD()
    return crud.get_unread_by_user(session, user_id)


def mark_notification_read(
    session: Session,
    notification_id: str,
    user_id: str,
) -> bool:
    """Mark a single notification as read.

    Args:
        session: SQLAlchemy session.
        notification_id: UUID of the notification.
        user_id: UUID of the user.

    Returns:
        True if marked, False if not found.

    Example:
        >>> mark_notification_read(session, "n1", "u1")
        True
    """
    crud = AssetNotificationCRUD()
    return crud.mark_as_read(session, notification_id, user_id)


def mark_all_notifications_read(
    session: Session,
    user_id: str,
) -> int:
    """Mark all notifications as read for a user.

    Args:
        session: SQLAlchemy session.
        user_id: UUID of the user.

    Returns:
        Number of notifications marked as read.

    Example:
        >>> mark_all_notifications_read(session, "u1")
        3
    """
    crud = AssetNotificationCRUD()
    return crud.mark_all_as_read(session, user_id)
