"""Database test fixtures with in-memory SQLite."""

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session

from finance_ai.database.models.user import User


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
