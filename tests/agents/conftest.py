"""Test fixtures for agent tests."""

from collections.abc import Callable, Generator
from datetime import date
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from finance_ai.database.base import Base
from finance_ai.database.models.user import User


@pytest.fixture
def mock_chat_model() -> MagicMock:
    """Create a mock LangChain ChatModel.

    The mock supports bind_tools (returns itself) and invoke.

    Returns:
        MagicMock that behaves like BaseChatModel.
    """
    mock = MagicMock(spec=BaseChatModel)
    mock.bind_tools.return_value = mock
    return mock


@pytest.fixture
def tax_tool_call_message() -> AIMessage:
    """Create an AIMessage with a tool call for calculate_thai_tax.

    Returns:
        AIMessage with tool_calls for a basic tax calculation.
    """
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "name": "calculate_thai_tax",
                "args": {
                    "gross_income": "1200000",
                    "deductions_by_type": {
                        "personal_allowance": "60000",
                        "child_allowance": "60000",
                        "rmf": "100000",
                    },
                    "withholding_tax_paid": "0",
                },
            }
        ],
    )


@pytest.fixture
def tax_formatted_response() -> AIMessage:
    """Create an AIMessage with a formatted tax response.

    Returns:
        AIMessage with Thai-language tax result summary.
    """
    return AIMessage(content="ผลการคำนวณภาษี: รายได้ 1,200,000 บาท ภาษี 86,000 บาท")


@pytest.fixture
def mock_google_settings() -> dict[str, Any]:
    """Settings dict for Google provider testing.

    Returns:
        Dict with Google LLM configuration.
    """
    return {
        "llm_provider": "google",
        "google_api_key": "test-api-key",
        "google_model": "gemini-pro",
        "llm_temperature": 0.7,
        "llm_max_tokens": 4000,
    }


@pytest.fixture
def mock_ollama_settings() -> dict[str, Any]:
    """Settings dict for OLLAMA provider testing.

    Returns:
        Dict with OLLAMA LLM configuration.
    """
    return {
        "llm_provider": "ollama",
        "ollama_base_url": "http://localhost:11434",
        "ollama_model": "THALLE",
        "llm_temperature": 0.7,
        "llm_max_tokens": 4000,
    }


@pytest.fixture
def expense_tool_call_message() -> AIMessage:
    """Create an AIMessage with a tool call for add_expense.

    Returns:
        AIMessage with tool_calls for recording an expense.
    """
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "name": "add_expense",
                "args": {
                    "amount": "80",
                    "category": "food",
                    "description": "กาแฟ",
                    "transaction_date": "2026-02-24",
                },
            }
        ],
    )


@pytest.fixture
def expense_formatted_response() -> AIMessage:
    """Create an AIMessage with a formatted expense response.

    Returns:
        AIMessage with Thai-language expense confirmation.
    """
    return AIMessage(content="บันทึกรายจ่าย: กาแฟ 80 บาท หมวดอาหาร")


@pytest.fixture
def investment_tool_call_message() -> AIMessage:
    """Create an AIMessage with a tool call for view_portfolio.

    Returns:
        AIMessage with tool_calls for viewing portfolio.
    """
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "name": "view_portfolio",
                "args": {},
            }
        ],
    )


@pytest.fixture
def investment_formatted_response() -> AIMessage:
    """Create an AIMessage with a formatted portfolio response.

    Returns:
        AIMessage with Thai-language portfolio summary.
    """
    return AIMessage(content="พอร์ตการลงทุน: มูลค่ารวม 500,000 บาท กำไร 50,000 บาท")


@pytest.fixture
def planning_tool_call_message() -> AIMessage:
    """Create an AIMessage with a tool call for create_financial_goal.

    Returns:
        AIMessage with tool_calls for creating a financial goal.
    """
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "name": "create_financial_goal",
                "args": {
                    "goal_type": "savings",
                    "name": "เงินฉุกเฉิน",
                    "target_amount": "100000",
                },
            }
        ],
    )


@pytest.fixture
def planning_formatted_response() -> AIMessage:
    """Create an AIMessage with a formatted planning response.

    Returns:
        AIMessage with Thai-language planning result summary.
    """
    return AIMessage(content="สร้างเป้าหมาย: ออมเงินฉุกเฉิน 100,000 บาท สำเร็จ")


@pytest.fixture
def recommendation_tool_call_message() -> AIMessage:
    """Create an AIMessage with a tool call for generate_financial_recommendations.

    Returns:
        AIMessage with tool_calls for generating recommendations.
    """
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "name": "generate_financial_recommendations",
                "args": {},
            }
        ],
    )


@pytest.fixture
def recommendation_formatted_response() -> AIMessage:
    """Create an AIMessage with a formatted recommendation response.

    Returns:
        AIMessage with Thai-language recommendation summary.
    """
    return AIMessage(
        content="คำแนะนำการเงิน: พบ 3 ข้อแนะนำ คะแนนสุขภาพ 75/100",
    )


@pytest.fixture
def test_engine() -> Engine:
    """Create an in-memory SQLite engine for agent testing.

    Returns:
        Engine: SQLAlchemy engine using in-memory SQLite.
    """
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def test_session(test_engine: Engine) -> Generator[Session, None, None]:
    """Create a test session that rolls back after each test.

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
    """Create and persist a sample user for agent testing.

    Args:
        test_session: Database session.

    Returns:
        User: Persisted test user instance.
    """
    user = User(
        email="agent-test@example.com",
        hashed_password="hashed_password_123",
        full_name="Agent Test User",
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
def db_session_factory(
    test_engine: Engine,
) -> Callable[[], Session]:
    """Create a session factory bound to the test engine.

    Each call creates a new session (matching production behavior).

    Args:
        test_engine: In-memory SQLite engine with tables created.

    Returns:
        Callable that creates new sessions from the test engine.
    """
    return sessionmaker(bind=test_engine)
