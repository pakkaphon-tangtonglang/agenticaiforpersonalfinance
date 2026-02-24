"""Database engine creation and session management."""

from collections.abc import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from finance_ai.core.config import get_settings


def create_database_engine() -> Engine:
    """
    Create a SQLAlchemy engine from application settings.

    Returns:
        Engine: Configured SQLAlchemy engine.

    Example:
        >>> engine = create_database_engine()
    """
    settings = get_settings()
    connect_args: dict[str, bool] = {}
    if settings.db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(
        settings.db_url,
        echo=settings.debug,
        connect_args=connect_args,
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
