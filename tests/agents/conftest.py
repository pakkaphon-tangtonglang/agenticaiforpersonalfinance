"""Test fixtures for agent tests."""

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from finance_ai.agents.graph_cache import clear_graph_cache


@pytest.fixture(autouse=True)
def _clear_graph_cache() -> Generator[None, None, None]:
    """Clear the graph cache before and after each agent test."""
    clear_graph_cache()
    yield
    clear_graph_cache()


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
        "google_model": "gemini-2.5-flash",
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


