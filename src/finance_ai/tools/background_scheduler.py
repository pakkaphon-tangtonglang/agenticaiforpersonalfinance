"""Background scheduler for periodic asset data fetching.

Uses APScheduler to run cron jobs that fetch asset data and create
notifications for users. Integrates with the scheduler_service layer.
"""

from typing import Any, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

# Singleton scheduler instance
_scheduler: Optional[BackgroundScheduler] = None


def _parse_cron_expression(cron_expr: str) -> dict[str, str]:
    """Parse a cron expression into APScheduler CronTrigger fields.

    Supports standard 5-field cron: minute hour day month day_of_week.

    Args:
        cron_expr: Cron expression (e.g., "42 11 * * *").

    Returns:
        Dict with keys: minute, hour, day, month, day_of_week.

    Raises:
        ValueError: If expression does not have exactly 5 fields.

    Example:
        >>> _parse_cron_expression("42 11 * * *")
        {'minute': '42', 'hour': '11', 'day': '*', 'month': '*', 'day_of_week': '*'}
    """
    parts = cron_expr.strip().split()
    if len(parts) != 5:  # noqa: PLR2004
        raise ValueError(
            f"Invalid cron expression: '{cron_expr}'. "
            f"Expected 5 fields (minute hour day month day_of_week)."
        )
    return {
        "minute": parts[0],
        "hour": parts[1],
        "day": parts[2],
        "month": parts[3],
        "day_of_week": parts[4],
    }


def _execute_job(
    schedule_id: str,
    session_factory: Any,
) -> None:
    """Execute a single scheduled fetch job.

    Called by APScheduler when a cron trigger fires. Opens a new DB
    session, loads the schedule, and calls execute_scheduled_fetch.

    Args:
        schedule_id: UUID of the AssetSchedule to execute.
        session_factory: SQLAlchemy session factory.
    """
    from finance_ai.tools.scheduler_service import (  # noqa: PLC0415
        execute_scheduled_fetch,
    )

    session = session_factory()
    try:
        from finance_ai.database.crud.schedule_crud import (  # noqa: PLC0415
            AssetScheduleCRUD,
        )

        crud = AssetScheduleCRUD(session)
        schedules = crud.get_by_user_and_id(schedule_id)
        if schedules is None:
            logger.warning("Schedule %s not found, skipping", schedule_id)
            return

        notification = execute_scheduled_fetch(session, schedules)
        logger.info(
            "Scheduled fetch completed: %s → %s",
            schedules.symbol,
            notification.content[:50],
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Scheduled job failed for %s: %s", schedule_id, exc)
    finally:
        session.close()


def get_scheduler() -> BackgroundScheduler:
    """Get or create the singleton BackgroundScheduler.

    Returns:
        The BackgroundScheduler instance.
    """
    global _scheduler  # noqa: PLW0603
    if _scheduler is None:
        _scheduler = BackgroundScheduler(
            timezone="Asia/Bangkok",
            job_defaults={"max_instances": 1},
        )
    return _scheduler


def register_schedule(
    schedule_id: str,
    cron_expression: str,
    session_factory: Any,
) -> None:
    """Register a schedule as an APScheduler cron job.

    Args:
        schedule_id: UUID of the AssetSchedule.
        cron_expression: Cron expression (e.g., "42 11 * * *").
        session_factory: SQLAlchemy session factory.

    Example:
        >>> register_schedule("sched-1", "42 11 * * *", factory)
    """
    scheduler = get_scheduler()
    job_id = f"asset_schedule_{schedule_id}"

    # Remove existing job if any (for re-registration)
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)

    cron_fields = _parse_cron_expression(cron_expression)
    trigger = CronTrigger(**cron_fields, timezone="Asia/Bangkok")

    scheduler.add_job(
        _execute_job,
        trigger=trigger,
        id=job_id,
        args=[schedule_id, session_factory],
        replace_existing=True,
    )
    logger.info(
        "Registered schedule job: %s (%s)",
        job_id,
        cron_expression,
    )


def unregister_schedule(schedule_id: str) -> None:
    """Remove a schedule's cron job from APScheduler.

    Args:
        schedule_id: UUID of the AssetSchedule.

    Example:
        >>> unregister_schedule("sched-1")
    """
    scheduler = get_scheduler()
    job_id = f"asset_schedule_{schedule_id}"
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)
        logger.info("Unregistered schedule job: %s", job_id)


def load_all_schedules(session_factory: Any) -> int:
    """Load all active schedules from DB and register them.

    Called at app startup to restore all cron jobs.

    Args:
        session_factory: SQLAlchemy session factory.

    Returns:
        Number of schedules registered.

    Example:
        >>> count = load_all_schedules(factory)
    """
    from finance_ai.database.crud.schedule_crud import (  # noqa: PLC0415
        AssetScheduleCRUD,
    )

    session = session_factory()
    try:
        crud = AssetScheduleCRUD(session)
        all_active = crud.get_all_active()
        for schedule in all_active:
            register_schedule(
                schedule.id,
                schedule.cron_expression,
                session_factory,
            )
        return len(all_active)
    finally:
        session.close()


def start_scheduler(session_factory: Any) -> int:
    """Start the background scheduler and load all active schedules.

    Safe to call multiple times — will not start if already running.

    Args:
        session_factory: SQLAlchemy session factory.

    Returns:
        Number of schedules loaded.

    Example:
        >>> count = start_scheduler(factory)
    """
    scheduler = get_scheduler()
    count = load_all_schedules(session_factory)

    if not scheduler.running:
        scheduler.start()
        logger.info(
            "Background scheduler started with %d active schedules",
            count,
        )

    return count


def stop_scheduler() -> None:
    """Stop the background scheduler gracefully.

    Example:
        >>> stop_scheduler()
    """
    global _scheduler  # noqa: PLW0603
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Background scheduler stopped")
    _scheduler = None
