"""Tests for database engine creation."""

from typing import Any
from unittest.mock import MagicMock, patch

from finance_ai.database.session import (
    POOL_RECYCLE_SECONDS,
    build_engine_options,
    create_database_engine,
)


class TestBuildEngineOptions:
    """Tests for build_engine_options."""

    def test_postgres_enables_pre_ping_and_recycle(self) -> None:
        """Serverless Postgres drops idle SSL connections, so pooled ones must be checked."""
        options = build_engine_options("postgresql://user:pw@host/db")
        assert options["pool_pre_ping"] is True
        assert options["pool_recycle"] == POOL_RECYCLE_SECONDS
        assert not options["connect_args"]

    def test_sqlite_disables_thread_check(self) -> None:
        """SQLite needs check_same_thread=False and no pool recycling."""
        options = build_engine_options("sqlite:///./local.db")
        assert options["connect_args"] == {"check_same_thread": False}
        assert "pool_recycle" not in options


class TestCreateDatabaseEngine:
    """Tests for create_database_engine."""

    def test_passes_options_to_sqlalchemy(self) -> None:
        """The engine is created with the URL, echo flag and pool options."""
        settings: Any = MagicMock(db_url="postgresql://u:p@h/d", debug=False)
        with (
            patch("finance_ai.database.session.get_settings", return_value=settings),
            patch("finance_ai.database.session.create_engine") as mock_create,
        ):
            create_database_engine()
        kwargs = mock_create.call_args.kwargs
        assert mock_create.call_args.args == ("postgresql://u:p@h/d",)
        assert kwargs["echo"] is False
        assert kwargs["pool_pre_ping"] is True
