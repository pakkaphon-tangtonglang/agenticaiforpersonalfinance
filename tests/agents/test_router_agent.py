"""Tests for the Router Agent."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from finance_ai.agents.router_agent import (
    build_unsupported_response,
    classify_query,
    execute_expense_agent,
    execute_investment_agent,
    parse_router_response,
    route_query,
)
from finance_ai.agents.schemas import RouterDecision


class TestParseRouterResponse:
    """Tests for parsing LLM responses into RouterDecision."""

    def test_valid_tax_json(self) -> None:
        """Parses valid tax JSON into RouterDecision."""
        content = '{"intent": "tax", "confidence": 0.95}'
        result = parse_router_response(content)
        assert result.intent == "tax"
        assert result.confidence == Decimal("0.95")

    def test_valid_expense_json(self) -> None:
        """Parses valid expense JSON into RouterDecision."""
        content = '{"intent": "expense", "confidence": 0.9}'
        result = parse_router_response(content)
        assert result.intent == "expense"
        assert result.confidence == Decimal("0.9")

    def test_json_with_code_fence(self) -> None:
        """Handles JSON wrapped in markdown code fences."""
        content = '```json\n{"intent": "investment", "confidence": 0.8}\n```'
        result = parse_router_response(content)
        assert result.intent == "investment"
        assert result.confidence == Decimal("0.8")

    def test_invalid_json_returns_default(self) -> None:
        """Returns default unknown decision for invalid JSON."""
        result = parse_router_response("not valid json at all")
        assert result.intent == "unknown"
        assert result.confidence == Decimal("0")

    def test_empty_string_returns_default(self) -> None:
        """Returns default unknown decision for empty string."""
        result = parse_router_response("")
        assert result.intent == "unknown"

    def test_missing_fields_returns_default(self) -> None:
        """Returns default when JSON is missing required fields."""
        result = parse_router_response('{"intent": "tax"}')
        assert result.intent == "unknown"

    def test_non_string_content(self) -> None:
        """Handles non-string content by converting to string."""
        result = parse_router_response(12345)
        assert result.intent == "unknown"


class TestClassifyQuery:
    """Tests for query classification."""

    def test_calls_llm_with_messages(self, mock_chat_model: MagicMock) -> None:
        """Sends system prompt and user query to the LLM."""
        mock_chat_model.invoke.return_value = AIMessage(
            content='{"intent": "tax", "confidence": 0.9}'
        )
        result = classify_query("คำนวณภาษี", chat_model=mock_chat_model)

        mock_chat_model.invoke.assert_called_once()
        call_args = mock_chat_model.invoke.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[1].content == "คำนวณภาษี"
        assert result.intent == "tax"

    def test_handles_llm_garbage_response(self, mock_chat_model: MagicMock) -> None:
        """Returns unknown when LLM returns unparseable response."""
        mock_chat_model.invoke.return_value = AIMessage(content="I don't understand")
        result = classify_query("test", chat_model=mock_chat_model)
        assert result.intent == "unknown"


class TestRouteQuery:
    """Tests for the main routing function."""

    @patch("finance_ai.agents.router_agent.execute_tax_agent")
    @patch("finance_ai.agents.router_agent.classify_query")
    def test_routes_tax_to_tax_agent(
        self,
        mock_classify: MagicMock,
        mock_execute: MagicMock,
    ) -> None:
        """Routes tax intent to the tax agent."""
        mock_classify.return_value = RouterDecision(intent="tax", confidence=Decimal("0.95"))
        mock_execute.return_value = {"intent": "tax", "response": "ผลภาษี"}

        result = route_query("คำนวณภาษี")
        assert result["intent"] == "tax"
        mock_execute.assert_called_once()

    @patch("finance_ai.agents.router_agent.execute_expense_agent")
    @patch("finance_ai.agents.router_agent.classify_query")
    def test_routes_expense_to_expense_agent(
        self,
        mock_classify: MagicMock,
        mock_execute: MagicMock,
    ) -> None:
        """Routes expense intent to the expense agent."""
        mock_classify.return_value = RouterDecision(
            intent="expense",
            confidence=Decimal("0.9"),
        )
        mock_execute.return_value = {"intent": "expense", "response": "บันทึกแล้ว"}

        result = route_query("จ่ายค่ากาแฟ 80 บาท")
        assert result["intent"] == "expense"
        mock_execute.assert_called_once()

    @patch("finance_ai.agents.router_agent.execute_investment_agent")
    @patch("finance_ai.agents.router_agent.classify_query")
    def test_routes_investment_to_investment_agent(
        self,
        mock_classify: MagicMock,
        mock_execute: MagicMock,
    ) -> None:
        """Routes investment intent to the investment agent."""
        mock_classify.return_value = RouterDecision(
            intent="investment",
            confidence=Decimal("0.85"),
        )
        mock_execute.return_value = {"intent": "investment", "response": "พอร์ตของคุณ"}

        result = route_query("ดูพอร์ตหุ้น")
        assert result["intent"] == "investment"
        mock_execute.assert_called_once()

    @patch("finance_ai.agents.router_agent.classify_query")
    def test_unsupported_intent_returns_message(
        self,
        mock_classify: MagicMock,
    ) -> None:
        """Non-supported intents return unsupported message."""
        mock_classify.return_value = RouterDecision(intent="general", confidence=Decimal("0.7"))
        result = route_query("สวัสดี")
        assert result["intent"] == "general"
        assert "ค่าใช้จ่าย" in result["response"]


class TestExecuteInvestmentAgent:
    """Tests for execute_investment_agent."""

    @patch("finance_ai.agents.investment_agent.build_investment_agent_graph")
    def test_returns_investment_response(self, mock_build: MagicMock) -> None:
        """Returns dict with intent='investment' and response."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="มูลค่าพอร์ต 500,000 บาท")],
        }
        mock_build.return_value = mock_graph

        result = execute_investment_agent("ดูพอร์ตของฉัน", user_id="user-1")
        assert result["intent"] == "investment"
        assert "500,000" in result["response"]

    @patch("finance_ai.agents.investment_agent.build_investment_agent_graph")
    def test_passes_user_id_and_session_factory(self, mock_build: MagicMock) -> None:
        """Passes user_id and db_session_factory to the graph invoke."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="ok")],
        }
        mock_build.return_value = mock_graph

        execute_investment_agent("test", user_id="user-123")
        call_args = mock_graph.invoke.call_args[0][0]
        assert call_args["user_id"] == "user-123"
        assert "db_session_factory" in call_args


class TestBuildUnsupportedResponse:
    """Tests for unsupported intent responses."""

    def test_returns_thai_message(self) -> None:
        """Response is in Thai and mentions supported features."""
        decision = RouterDecision(intent="general", confidence=Decimal("0.5"))
        result = build_unsupported_response(decision)
        assert result["intent"] == "general"
        assert "ภาษี" in result["response"]
        assert "ค่าใช้จ่าย" in result["response"]
        assert "การลงทุน" in result["response"]

    def test_unknown_intent(self) -> None:
        """Handles unknown intent gracefully."""
        decision = RouterDecision(intent="unknown", confidence=Decimal("0"))
        result = build_unsupported_response(decision)
        assert result["intent"] == "unknown"
