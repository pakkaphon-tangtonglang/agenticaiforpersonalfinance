"""Per-run database copies and database checks for the architecture study.

A seeded demo database is built once; every run gets its own copy, so a
task that writes data cannot affect the next run. Database checks compare
the run's copy with the untouched template.
"""

import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from finance_ai.database.base import Base
from finance_ai.database.models.financial_goal import FinancialGoal
from finance_ai.database.models.income import Income
from finance_ai.database.models.investment_holding import InvestmentHolding
from finance_ai.database.models.transaction import Transaction
from finance_ai.database.models.watched_asset import WatchedAsset
from finance_ai.database.session import build_engine_options, create_session_factory
from finance_ai.evaluation.architectures.task_models import DbCheck
from finance_ai.tools.demo_seed_service import DEMO_USER_ID, seed_demo_data

TEMPLATE_NAME = "template.db"
EMERGENCY_GOAL_TYPE = "emergency_fund"

SessionFactory = Callable[[], Session]


def _sqlite_engine(path: Path) -> Engine:
    """Create an engine for a SQLite file."""
    url = f"sqlite:///{path.as_posix()}"
    return create_engine(url, **build_engine_options(url))


def build_template_database(directory: Path, as_of: date) -> Path:
    """Create the seeded demo database every run starts from.

    Args:
        directory: Folder for the database files.
        as_of: Reference date for the demo data (fixed for the whole study).

    Returns:
        Path of the template SQLite file.

    Example:
        >>> build_template_database(tmp_path, date(2026, 11, 2)).name  # doctest: +SKIP
        'template.db'
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TEMPLATE_NAME
    path.unlink(missing_ok=True)
    engine = _sqlite_engine(path)
    try:
        Base.metadata.create_all(engine)
        session = create_session_factory(engine)()
        try:
            seed_demo_data(session, as_of=as_of)
        finally:
            session.close()
    finally:
        engine.dispose()
    return path


@contextmanager
def fresh_database(template: Path, run_id: str) -> Iterator[SessionFactory]:
    """Yield a session factory for a private copy of the template.

    Args:
        template: Template database path.
        run_id: Unique id used in the copy's file name.

    Yields:
        Session factory bound to the copy; the copy is deleted afterwards.
    """
    path = template.with_name(f"run_{run_id}.db")
    shutil.copyfile(template, path)
    engine = _sqlite_engine(path)
    try:
        yield create_session_factory(engine)
    finally:
        engine.dispose()
        path.unlink(missing_ok=True)


@contextmanager
def open_template(template: Path) -> Iterator[SessionFactory]:
    """Yield a session factory for the template itself (read-only use).

    Args:
        template: Template database path.

    Yields:
        Session factory bound to the template.
    """
    engine = _sqlite_engine(template)
    try:
        yield create_session_factory(engine)
    finally:
        engine.dispose()


def check_database(
    run_factory: SessionFactory, template_factory: SessionFactory, checks: list[DbCheck]
) -> list[bool]:
    """Evaluate each database check on the run's copy.

    Row checks pass when the copy has more matching rows than the template;
    ``goal_current`` passes when the emergency fund equals the amount.

    Args:
        run_factory: Sessions on the run's copy.
        template_factory: Sessions on the untouched template.
        checks: Checks from the task.

    Returns:
        One bool per check, in order.
    """
    with run_factory() as run_session, template_factory() as template_session:
        return [_passes(check, run_session, template_session) for check in checks]


def _passes(check: DbCheck, run_session: Session, template_session: Session) -> bool:
    """Evaluate one check."""
    if check.kind == "goal_current":
        return _emergency_fund_amount(run_session) == check.amount
    return _count_rows(run_session, check) > _count_rows(template_session, check)


def _count_rows(session: Session, check: DbCheck) -> int:
    """Count the demo user's rows matching a row check."""
    model, condition = _row_filter(check)
    # pylint false positive: SQLAlchemy func.count is callable at runtime.
    count = func.count()  # pylint: disable=not-callable
    statement = select(count).select_from(model).where(model.user_id == DEMO_USER_ID, condition)
    return int(session.scalar(statement) or 0)


def _row_filter(check: DbCheck) -> tuple[Any, Any]:
    """Return (ORM model, filter condition) for a row check."""
    filters: dict[str, Callable[[], tuple[Any, Any]]] = {
        "expense": lambda: (
            Transaction,
            (Transaction.transaction_type == "expense") & (Transaction.amount == check.amount),
        ),
        "income": lambda: (Income, Income.amount == check.amount),
        "goal_target": lambda: (FinancialGoal, FinancialGoal.target_amount == check.amount),
        "watchlist": lambda: (WatchedAsset, WatchedAsset.symbol == check.symbol),
        "holding": lambda: (InvestmentHolding, InvestmentHolding.symbol == check.symbol),
    }
    return filters[check.kind]()


def _emergency_fund_amount(session: Session) -> Decimal | None:
    """Return the demo user's emergency-fund balance."""
    return session.scalar(
        select(FinancialGoal.current_amount).where(
            FinancialGoal.user_id == DEMO_USER_ID,
            FinancialGoal.goal_type == EMERGENCY_GOAL_TYPE,
        )
    )
