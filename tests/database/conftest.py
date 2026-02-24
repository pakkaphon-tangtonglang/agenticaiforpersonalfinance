"""Database test fixtures with in-memory SQLite."""

from collections.abc import Generator
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from finance_ai.database.base import Base
from finance_ai.database.models.user import User


@pytest.fixture
def test_engine() -> Engine:
    """
    Create an in-memory SQLite engine for testing.

    Returns:
        Engine: SQLAlchemy engine using in-memory SQLite.
    """
    engine = create_engine("sqlite:///:memory:", echo=False)
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
    Create and persist a sample user for testing.

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
def sample_user_minimal(test_session: Session) -> User:
    """
    Create a user with only required fields for testing defaults.

    Args:
        test_session: Database session.

    Returns:
        User: Persisted user with default values.
    """
    user = User(
        email="minimal@example.com",
        hashed_password="hashed_password_456",
        full_name="Minimal User",
    )
    test_session.add(user)
    test_session.commit()
    test_session.refresh(user)
    return user


@pytest.fixture
def sample_money_amount() -> Decimal:
    """
    Provide a standard test money amount.

    Returns:
        Decimal: Test amount of 50,000.00 THB.
    """
    return Decimal("50000.00")
