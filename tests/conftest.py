"""
Shared pytest fixtures for all tests.

Provides mock configurations, clients, and responses for testing.
"""

from collections.abc import Callable, Generator
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from finance_ai.core import api_security
from finance_ai.core.config import Settings
from finance_ai.core.llm.base import LLMResponse
from finance_ai.database.base import Base
from finance_ai.database.models.user import User


@pytest.fixture(autouse=True)
def _isolate_api_security(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep API security middleware out of other tests' way.

    Resets the global rate-limiter state between tests and disables
    rate limiting entirely (test suites send many requests from one
    client, which would otherwise trip the shared 429 counter).
    """
    api_security.reset_rate_limiter()
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")


@pytest.fixture
def mock_settings() -> Settings:
    """
    Create mock settings for testing.

    Returns:
        Settings: Test configuration with safe defaults
    """
    return Settings(
        app_env="development",
        log_level="DEBUG",
        llm_provider="google",
        google_api_key="test-key",
        google_model="gemini-2.5-flash",
        ollama_base_url="http://localhost:11434",
        ollama_model="minimax-m3",
    )


@pytest.fixture
def mock_llm_response() -> LLMResponse:
    """
    Create mock LLM response.

    Returns:
        LLMResponse: Standard response format for testing
    """
    return LLMResponse(
        content="Test response content",
        model="test-model",
        usage={
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        },
        provider="test",
    )


@pytest.fixture
def mock_messages() -> list[dict[str, str]]:
    """
    Create mock message list.

    Returns:
        list: Standard message format for testing
    """
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]


@pytest.fixture
def mock_money() -> Decimal:
    """
    Create mock money amount.

    Returns:
        Decimal: Test money amount
    """
    return Decimal("1000.50")


@pytest.fixture
def test_engine() -> Engine:
    """
    Create an in-memory SQLite engine for testing.

    Uses StaticPool so every session (including those created by agent
    tools via db_session_factory) shares the same in-memory database.
    Without it, each new connection gets its own empty database and
    table creation is invisible to tool sessions.

    Returns:
        Engine: SQLAlchemy engine using in-memory SQLite.
    """
    engine = create_engine(
        "sqlite://",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def test_session(test_engine: Engine) -> Generator[Session, None, None]:
    """
    Create a test session that rolls back after each test.

    Args:
        test_engine: In-memory SQLite engine.

    Yields:
        Session: Clean database session for testing.
    """
    factory = sessionmaker(bind=test_engine)
    session = factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def sample_user(test_session: Session) -> User:
    """
    Create and persist a sample user.

    Args:
        test_session: Database session.

    Returns:
        User: Persisted test user instance.
    """
    user = User(
        email="test@example.com",
        hashed_password="hashed_password_123",
        full_name="Test User",
        tax_id="1234567890123",
        date_of_birth=date(1990, 1, 15),
        marital_status="single",
        number_of_children=0,
        number_of_parents=2,
    )
    test_session.add(user)
    test_session.commit()
    test_session.refresh(user)
    return user


@pytest.fixture
def db_session_factory(test_engine: Engine) -> Callable[[], Session]:
    """
    Create a session factory bound to the test engine.

    Each call creates a new session (matching production behavior).

    Args:
        test_engine: In-memory SQLite engine with tables created.

    Returns:
        Callable that creates new sessions from the test engine.
    """
    return sessionmaker(bind=test_engine)
