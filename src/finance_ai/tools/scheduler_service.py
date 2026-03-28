"""Service layer for asset schedule management and job execution.

Handles CRUD operations on asset schedules and executes scheduled
data-fetching jobs via the Bright Data API, storing results as
notifications for the user.
"""

from typing import Any, Optional

from sqlalchemy.orm import Session

from finance_ai.core.logging import get_logger
from finance_ai.database.crud.schedule_crud import (
    AssetNotificationCRUD,
    AssetScheduleCRUD,
)
from finance_ai.database.models.asset_notification import AssetNotification
from finance_ai.database.models.asset_schedule import AssetSchedule

logger = get_logger(__name__)


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
    crud = AssetScheduleCRUD(session)
    return crud.create(user_id, symbol, description, cron_expression, max_runs)


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
    crud = AssetScheduleCRUD(session)
    if active_only:
        return crud.get_active_by_user(user_id)
    return crud.get_by_user(user_id)


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
    crud = AssetScheduleCRUD(session)
    return crud.delete(schedule_id, user_id)


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
    crud = AssetScheduleCRUD(session)
    return crud.deactivate(schedule_id, user_id)


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
    notif_crud = AssetNotificationCRUD(session)
    notification = notif_crud.create(
        user_id=schedule.user_id,
        symbol=schedule.symbol,
        content=content,
        schedule_id=schedule.id,
    )

    sched_crud = AssetScheduleCRUD(session)
    new_count = sched_crud.increment_run_count(schedule.id)
    if schedule.max_runs and new_count >= schedule.max_runs:
        sched_crud.deactivate(schedule.id, schedule.user_id)
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


def execute_immediate_fetch(
    session: Session,
    user_id: str,
    symbol: str,
    fetch_type: str = "all",
    chat_model: Any = None,
) -> AssetNotification:
    """Execute an immediate one-time fetch (no schedule needed).

    Args:
        session: SQLAlchemy session.
        user_id: UUID of the user.
        symbol: Ticker symbol to fetch.
        fetch_type: "news" for news only, "price" for price only,
                    "all" for both.
        chat_model: Optional LangChain ChatModel for LLM summarization.

    Returns:
        Created AssetNotification with fetched data.

    Example:
        >>> notification = execute_immediate_fetch(session, "u1", "GC=F")
    """
    if fetch_type == "news":
        content = _fetch_news_only(symbol, chat_model)
    elif fetch_type == "price":
        content = _fetch_price_only(symbol)
    else:
        content = _fetch_asset_summary(symbol, chat_model)

    crud = AssetNotificationCRUD(session)
    return crud.create(
        user_id=user_id,
        symbol=symbol.strip().upper(),
        content=content,
    )


def _fetch_price_only(symbol: str) -> str:
    """Fetch only price data for a symbol.

    Args:
        symbol: Ticker symbol to fetch.

    Returns:
        Formatted price text in Thai.
    """
    from finance_ai.tools.price_client import (  # noqa: PLC0415
        fetch_current_price,
    )

    price = fetch_current_price(symbol)
    if price is None:
        return f"ไม่สามารถดึงราคา {symbol} ได้ในขณะนี้"
    return f"💰 **{symbol}**: ราคาปัจจุบัน {price:,.2f}"


def _fetch_news_only(
    symbol: str,
    chat_model: Any = None,
) -> str:
    """Fetch news for a symbol, optionally summarize via LLM.

    Args:
        symbol: Ticker symbol to fetch.
        chat_model: Optional ChatModel for LLM summarization.

    Returns:
        Formatted news text in Thai.
    """
    from finance_ai.tools.market_data_service import (  # noqa: PLC0415
        fetch_finance_news,
    )

    news_result = fetch_finance_news(symbol)
    if not news_result.has_news:
        return f"ไม่พบข่าวสำหรับ {symbol} ในขณะนี้"

    if chat_model is not None:
        return _summarize_with_llm(
            chat_model,
            symbol,
            news_result.news_content,
        )
    return f"📰 **ข่าวล่าสุด {symbol}**\n{news_result.news_content}"


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
    )
    from finance_ai.tools.price_client import (  # noqa: PLC0415
        fetch_current_price,
    )

    price = fetch_current_price(symbol)
    price_text = (
        f"💰 **{symbol}**: ราคาปัจจุบัน {price:,.2f}" if price else f"ไม่สามารถดึงราคา {symbol} ได้ในขณะนี้"
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
    crud = AssetNotificationCRUD(session)
    return crud.get_unread_by_user(user_id)


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
    crud = AssetNotificationCRUD(session)
    return crud.mark_as_read(notification_id, user_id)


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
    crud = AssetNotificationCRUD(session)
    return crud.mark_all_as_read(user_id)
