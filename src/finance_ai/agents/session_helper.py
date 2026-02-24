"""Session helper for agent tools that need database access."""

from collections.abc import Callable, Generator
from contextlib import contextmanager

from sqlalchemy.orm import Session


@contextmanager
def get_tool_session(
    db_session_factory: Callable[[], Session] | None = None,
) -> Generator[Session, None, None]:
    """Get a database session for a tool call.

    If a factory is provided (e.g. from test injection), uses it.
    Otherwise, falls back to the default database session.

    Args:
        db_session_factory: Optional factory from graph state (for testing).
            If None, uses the default database session.

    Yields:
        Active database session, auto-closed on exit.

    Example:
        >>> with get_tool_session(factory) as session:
        ...     crud.create(session, ...)
    """
    if db_session_factory is not None:
        session = db_session_factory()
        try:
            yield session
        finally:
            session.close()
    else:
        from finance_ai.database.session import get_database_session  # noqa: PLC0415

        gen = get_database_session()
        session = next(gen)
        try:
            yield session
        finally:
            # Advance the generator to trigger its finally block (close session).
            # The generator is exhausted after one yield, so next() raises StopIteration.
            list(gen)  # noqa: W0106
