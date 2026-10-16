"""Database engine creation and session management."""

from collections.abc import Generator
from typing import Any

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from finance_ai.core.config import get_settings

# Neon closes idle connections (scale-to-zero, ~5 min), so recycle pooled
# connections before that and ping each one on checkout.
POOL_RECYCLE_SECONDS: int = 300


def build_engine_options(db_url: str) -> dict[str, Any]:
    """
    Build create_engine keyword options for the given database URL.

    SQLite needs ``check_same_thread=False``. Server databases get
    ``pool_pre_ping`` and ``pool_recycle`` so a connection dropped by the
    server (e.g. "SSL connection has been closed unexpectedly") is replaced
    instead of failing the request.

    Args:
        db_url: SQLAlchemy database URL.

    Returns:
        dict[str, Any]: Keyword arguments for ``create_engine``.

    Example:
        >>> build_engine_options("sqlite:///./local.db")["connect_args"]
        {'check_same_thread': False}
    """
    if db_url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {
        "connect_args": {},
        "pool_pre_ping": True,
        "pool_recycle": POOL_RECYCLE_SECONDS,
    }


def create_database_engine() -> Engine:
    """
    Create a SQLAlchemy engine from application settings.

    Returns:
        Engine: Configured SQLAlchemy engine.

    Example:
        >>> engine = create_database_engine()
    """
    settings = get_settings()
    return create_engine(
        settings.db_url,
        echo=settings.debug,
        **build_engine_options(settings.db_url),
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """
    Create a session factory bound to the given engine.

    Args:
        engine: SQLAlchemy engine to bind sessions to.

    Returns:
        sessionmaker: Configured session factory.

    Example:
        >>> factory = create_session_factory(engine)
        >>> session = factory()
    """
    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
    )


def get_database_session() -> Generator[Session, None, None]:
    """
    Yield a database session for dependency injection.

    Yields:
        Session: SQLAlchemy session, auto-closed after use.

    Example:
        >>> for session in get_database_session():
        ...     session.query(User).all()
    """
    engine = create_database_engine()
    session_factory = create_session_factory(engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
