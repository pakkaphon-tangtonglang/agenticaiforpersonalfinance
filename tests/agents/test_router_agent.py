"""Tests for the Orchestrator Agent."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from finance_ai.agents.router_agent import (
    _build_messages,
    build_unsupported_response,
    classify_query,
    execute_expense_agent,
    execute_asset_monitoring_agent,
    execute_planning_agent,
    execute_report_agent,
    execute_tax_agent,
    parse_orchestrator_response,
    orchestrate_query,
)
from finance_ai.agents.schemas import OrchestratorDecision


class TestParseRouterResponse:
    """Tests for parsing LLM responses into OrchestratorDecision."""

    def test_valid_tax_json(self) -> None:
        """Parses valid tax JSON into OrchestratorDecision."""
        content = '{"intent": "tax", "confidence": 0.95}'
        result = parse_orchestrator_response(content)
        assert result.intent == "tax"
        assert result.confidence == Decimal("0.95")

    def test_valid_expense_json(self) -> None:
        """Parses valid expense JSON into OrchestratorDecision."""
        content = '{"intent": "expense", "confidence": 0.9}'
        result = parse_orchestrator_response(content)
        assert result.intent == "expense"
        assert result.confidence == Decimal("0.9")

    def test_valid_planning_json(self) -> None:
        """Parses valid planning JSON into OrchestratorDecision."""
        content = '{"intent": "planning", "confidence": 0.9}'
        result = parse_orchestrator_response(content)
        assert result.intent == "planning"
        assert result.confidence == Decimal("0.9")

    def test_json_with_code_fence(self) -> None:
        """Handles JSON wrapped in markdown code fences."""
        content = '```json\n{"intent": "asset_monitoring", "confidence": 0.8}\n```'
        result = parse_orchestrator_response(content)
        assert result.intent == "asset_monitoring"
        assert result.confidence == Decimal("0.8")

    def test_invalid_json_returns_default(self) -> None:
        """Returns default unknown decision for invalid JSON."""
        result = parse_orchestrator_response("not valid json at all")
        assert result.intent == "unknown"
        assert result.confidence == Decimal("0")

    def test_empty_string_returns_default(self) -> None:
        """Returns default unknown decision for empty string."""
        result = parse_orchestrator_response("")
        assert result.intent == "unknown"

    def test_missing_fields_returns_default(self) -> None:
        """Returns default when JSON is missing required fields."""
        result = parse_orchestrator_response('{"intent": "tax"}')
        assert result.intent == "unknown"

    def test_non_string_content(self) -> None:
        """Handles non-string content by converting to string."""
        result = parse_orchestrator_response(12345)
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
        mock_classify.return_value = OrchestratorDecision(intent="tax", confidence=Decimal("0.95"))
        mock_execute.return_value = {"intent": "tax", "response": "ผลภาษี"}

        result = orchestrate_query("คำนวณภาษี")
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
        mock_classify.return_value = OrchestratorDecision(
            intent="expense",
            confidence=Decimal("0.9"),
        )
        mock_execute.return_value = {"intent": "expense", "response": "บันทึกแล้ว"}

        result = orchestrate_query("จ่ายค่ากาแฟ 80 บาท")
        assert result["intent"] == "expense"
        mock_execute.assert_called_once()

    @patch("finance_ai.agents.router_agent.execute_asset_monitoring_agent")
    @patch("finance_ai.agents.router_agent.classify_query")
    def test_routes_asset_monitoring_to_asset_monitoring_agent(
        self,
        mock_classify: MagicMock,
        mock_execute: MagicMock,
    ) -> None:
        """Routes asset_monitoring intent to the asset monitoring agent."""
        mock_classify.return_value = OrchestratorDecision(
            intent="asset_monitoring",
            confidence=Decimal("0.85"),
        )
        mock_execute.return_value = {"intent": "asset_monitoring", "response": "พอร์ตของคุณ"}

        result = orchestrate_query("ดูพอร์ตหุ้น")
        assert result["intent"] == "asset_monitoring"
        mock_execute.assert_called_once()

    @patch("finance_ai.agents.router_agent.execute_planning_agent")
    @patch("finance_ai.agents.router_agent.classify_query")
    def test_routes_planning_to_planning_agent(
        self,
        mock_classify: MagicMock,
        mock_execute: MagicMock,
    ) -> None:
        """Routes planning intent to the planning agent."""
        mock_classify.return_value = OrchestratorDecision(
            intent="planning",
            confidence=Decimal("0.9"),
        )
        mock_execute.return_value = {"intent": "planning", "response": "สร้างเป้าหมายแล้ว"}

        result = orchestrate_query("อยากออมเงิน 100,000 บาท")
        assert result["intent"] == "planning"
        mock_execute.assert_called_once()

    @patch("finance_ai.agents.router_agent.execute_planning_agent")
    @patch("finance_ai.agents.router_agent.classify_query")
    def test_general_intent_routes_to_planning(
        self,
        mock_classify: MagicMock,
        mock_execute: MagicMock,
    ) -> None:
        """Routes general intent to the planning agent."""
        mock_classify.return_value = OrchestratorDecision(
            intent="general",
            confidence=Decimal("0.7"),
        )
        mock_execute.return_value = {"intent": "general", "response": "คำแนะนำทั่วไป"}

        result = orchestrate_query("ออมเงินยังไงดี")
        assert result["response"] == "คำแนะนำทั่วไป"
        mock_execute.assert_called_once()

    @patch("finance_ai.agents.router_agent.execute_general_chat")
    @patch("finance_ai.agents.router_agent.classify_query")
    def test_unknown_intent_routes_to_general_chat(
        self,
        mock_classify: MagicMock,
        mock_general_chat: MagicMock,
    ) -> None:
        """Non-finance intents route to general chat."""
        mock_classify.return_value = OrchestratorDecision(
            intent="unknown", confidence=Decimal("0.3")
        )
        mock_general_chat.return_value = {"intent": "general_chat", "response": "สวัสดีค่ะ"}
        result = orchestrate_query("สูตรทำผัดไทย")
        assert result["intent"] == "general_chat"
        mock_general_chat.assert_called_once()


class TestExecuteAssetMonitoringAgent:
    """Tests for execute_asset_monitoring_agent."""

    @patch("finance_ai.agents.graph_cache.get_compiled_graph")
    def test_returns_asset_monitoring_response(self, mock_build: MagicMock) -> None:
        """Returns dict with intent='asset_monitoring' and response."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="มูลค่าพอร์ต 500,000 บาท")],
        }
        mock_build.return_value = mock_graph

        result = execute_asset_monitoring_agent("ดูพอร์ตของฉัน", user_id="user-1")
        assert result["intent"] == "asset_monitoring"
        assert "500,000" in result["response"]

    @patch("finance_ai.agents.graph_cache.get_compiled_graph")
    def test_passes_user_id_and_session_factory(self, mock_build: MagicMock) -> None:
        """Passes user_id and db_session_factory to the graph invoke."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="ok")],
        }
        mock_build.return_value = mock_graph

        execute_asset_monitoring_agent("test", user_id="user-123")
        call_args = mock_graph.invoke.call_args[0][0]
        assert call_args["user_id"] == "user-123"
        assert "db_session_factory" in call_args


class TestExecutePlanningAgent:
    """Tests for execute_planning_agent."""

    @patch("finance_ai.agents.graph_cache.get_compiled_graph")
    def test_returns_planning_response(self, mock_build: MagicMock) -> None:
        """Returns dict with intent='planning' and response."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="สร้างเป้าหมายออมเงิน 100,000 บาท สำเร็จ")],
        }
        mock_build.return_value = mock_graph

        result = execute_planning_agent("อยากออมเงิน", user_id="user-1")
        assert result["intent"] == "planning"
        assert "100,000" in result["response"]

    @patch("finance_ai.agents.graph_cache.get_compiled_graph")
    def test_passes_user_id_and_session_factory(self, mock_build: MagicMock) -> None:
        """Passes user_id and db_session_factory to the graph invoke."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="ok")],
        }
        mock_build.return_value = mock_graph

        execute_planning_agent("test", user_id="user-123")
        call_args = mock_graph.invoke.call_args[0][0]
        assert call_args["user_id"] == "user-123"
        assert "db_session_factory" in call_args


class TestBuildMessages:
    """Tests for _build_messages helper."""

    def test_query_only(self) -> None:
        """Builds messages with just the current query."""
        result = _build_messages("สวัสดี")
        assert result == [("user", "สวัสดี")]

    def test_with_history(self) -> None:
        """Prepends chat history before the current query."""
        history = [("user", "ถามก่อน"), ("assistant", "ตอบก่อน")]
        result = _build_messages("ถามใหม่", chat_history=history)
        assert len(result) == 3
        assert result[0] == ("user", "ถามก่อน")
        assert result[1] == ("assistant", "ตอบก่อน")
        assert result[2] == ("user", "ถามใหม่")

    def test_none_history(self) -> None:
        """None history treated as empty."""
        result = _build_messages("test", chat_history=None)
        assert result == [("user", "test")]

    def test_empty_history(self) -> None:
        """Empty history list works correctly."""
        result = _build_messages("test", chat_history=[])
        assert result == [("user", "test")]

    def test_does_not_mutate_input(self) -> None:
        """Original history list is not modified."""
        history = [("user", "original")]
        _build_messages("new", chat_history=history)
        assert len(history) == 1


class TestChatHistoryPassing:
    """Tests that chat_history is correctly passed through to agents."""

    @patch("finance_ai.agents.graph_cache.get_compiled_graph")
    def test_tax_agent_receives_history(self, mock_build: MagicMock) -> None:
        """Tax agent receives chat history in messages."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="ok")],
        }
        mock_build.return_value = mock_graph
        history = [("user", "prev"), ("assistant", "resp")]

        execute_tax_agent("new query", chat_history=history)
        call_args = mock_graph.invoke.call_args[0][0]
        messages = call_args["messages"]
        assert len(messages) == 3
        assert messages[0] == ("user", "prev")
        assert messages[2] == ("user", "new query")

    @patch("finance_ai.agents.graph_cache.get_compiled_graph")
    def test_tax_agent_no_history(self, mock_build: MagicMock) -> None:
        """Tax agent works without history (backward compatible)."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="ok")],
        }
        mock_build.return_value = mock_graph

        execute_tax_agent("query only")
        call_args = mock_graph.invoke.call_args[0][0]
        messages = call_args["messages"]
        assert len(messages) == 1
        assert messages[0] == ("user", "query only")

    @patch("finance_ai.agents.router_agent.execute_tax_agent")
    @patch("finance_ai.agents.router_agent.classify_query")
    def test_orchestrate_query_passes_history(
        self, mock_classify: MagicMock, mock_execute: MagicMock
    ) -> None:
        """orchestrate_query forwards chat_history to agent."""
        mock_classify.return_value = OrchestratorDecision(intent="tax", confidence=Decimal("0.9"))
        mock_execute.return_value = {"intent": "tax", "response": "ok"}
        history = [("user", "prev")]

        orchestrate_query("query", chat_history=history)
        mock_execute.assert_called_once()
        call_kwargs = mock_execute.call_args
        assert call_kwargs[0][4] == history  # 5th positional arg is chat_history


class TestExecuteReportAgent:
    """Tests for execute_report_agent."""

    @patch("finance_ai.agents.graph_cache.get_compiled_graph")
    def test_returns_report_response(self, mock_build: MagicMock) -> None:
        """Returns dict with intent='report' and response."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="รายงานการเงินประจำเดือน")],
        }
        mock_build.return_value = mock_graph

        result = execute_report_agent("สร้างรายงานการเงิน", user_id="user-1")
        assert result["intent"] == "report"
        assert "รายงาน" in result["response"]

    @patch("finance_ai.agents.graph_cache.get_compiled_graph")
    def test_passes_user_id(self, mock_build: MagicMock) -> None:
        """Passes user_id to the graph invoke."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="ok")],
        }
        mock_build.return_value = mock_graph

        execute_report_agent("test", user_id="user-123")
        call_args = mock_graph.invoke.call_args[0][0]
        assert call_args["user_id"] == "user-123"


class TestRouteQueryReport:
    """Tests for routing to report agent."""

    @patch("finance_ai.agents.router_agent.execute_report_agent")
    @patch("finance_ai.agents.router_agent.classify_query")
    def test_routes_to_report_agent(
        self, mock_classify: MagicMock, mock_execute: MagicMock
    ) -> None:
        """Report intent routes to execute_report_agent."""
        mock_classify.return_value = OrchestratorDecision(
            intent="report", confidence=Decimal("0.9")
        )
        mock_execute.return_value = {"intent": "report", "response": "รายงาน"}

        result = orchestrate_query("สร้างรายงานการเงิน")
        assert result["intent"] == "report"
        mock_execute.assert_called_once()


class TestBuildUnsupportedResponse:
    """Tests for unsupported intent responses."""

    def test_returns_thai_message(self) -> None:
        """Response is in Thai and mentions supported features."""
        decision = OrchestratorDecision(intent="general", confidence=Decimal("0.5"))
        result = build_unsupported_response(decision)
        assert result["intent"] == "general"
        assert "ภาษี" in result["response"]
        assert "ค่าใช้จ่าย" in result["response"]
        assert "การลงทุน" in result["response"]
        assert "วางแผนการเงิน" in result["response"]

    def test_mentions_report(self) -> None:
        """Response mentions report feature."""
        decision = OrchestratorDecision(intent="general", confidence=Decimal("0.5"))
        result = build_unsupported_response(decision)
        assert "รายงาน" in result["response"]

    def test_unknown_intent(self) -> None:
        """Handles unknown intent gracefully."""
        decision = OrchestratorDecision(intent="unknown", confidence=Decimal("0"))
        result = build_unsupported_response(decision)
        assert result["intent"] == "unknown"
