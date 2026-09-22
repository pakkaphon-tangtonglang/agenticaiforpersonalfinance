"""Tests for the Orchestrator Agent."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from finance_ai.agents.router_agent import (
    DEFAULT_ABLATION_CONFIG,
    ROUTER_CONFIDENCE_THRESHOLD,
    RouterAblationConfig,
    _build_clarify_response,
    _build_messages,
    _content_to_text,
    _resolve_asset_hint,
    build_unsupported_response,
    classify_query,
    execute_asset_monitoring_agent,
    execute_expense_agent,
    execute_general_chat,
    execute_planning_agent,
    execute_report_agent,
    execute_tax_agent,
    orchestrate_query,
    parse_orchestrator_response,
)
from finance_ai.agents.schemas import OrchestratorDecision
from finance_ai.tools.market_data_models import AssetSymbolMatch


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

    def test_gemini_content_block_list(self) -> None:
        """Parses JSON inside Gemini 3+ content-block lists."""
        content = [
            {
                "type": "text",
                "text": '{"intent": "tax", "confidence": 0.95}',
                "extras": {"signature": "abc"},
            }
        ]
        result = parse_orchestrator_response(content)
        assert result.intent == "tax"
        assert result.confidence == Decimal("0.95")

    def test_gemini_multiple_blocks_joined(self) -> None:
        """Multiple text blocks are concatenated before parsing."""
        content = [
            {"type": "text", "text": '{"intent": "tax", '},
            {"type": "text", "text": '"confidence": 0.9}'},
        ]
        result = parse_orchestrator_response(content)
        assert result.intent == "tax"
        assert result.confidence == Decimal("0.9")

    def test_content_to_text_string_passthrough(self) -> None:
        """String content is returned unchanged."""
        assert _content_to_text("hello") == "hello"

    def test_content_to_text_non_list_fallback(self) -> None:
        """Non-string, non-list content falls back to str()."""
        assert _content_to_text(42) == "42"


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

    @patch("finance_ai.agents.router_agent.execute_general_chat")
    @patch("finance_ai.agents.router_agent.classify_query")
    def test_general_intent_routes_to_general_chat(
        self,
        mock_classify: MagicMock,
        mock_general_chat: MagicMock,
    ) -> None:
        """Routes general intent to the general chat handler."""
        mock_classify.return_value = OrchestratorDecision(
            intent="general",
            confidence=Decimal("0.7"),
        )
        mock_general_chat.return_value = {"intent": "general_chat", "response": "คำตอบทั่วไป"}

        result = orchestrate_query("ดอกเบี้ยทบต้นคืออะไร")
        assert result["intent"] == "general_chat"
        mock_general_chat.assert_called_once()

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


class TestClassifyQueryWithHistory:
    """Tests for context-aware classification (chat history in the router)."""

    def test_classify_passes_history_to_llm(self, mock_chat_model: MagicMock) -> None:
        """Router LLM receives history messages before the current query."""
        mock_chat_model.invoke.return_value = AIMessage(
            content='{"intent": "asset_monitoring", "confidence": 0.9}'
        )
        history = [("user", "ดูพอร์ต PTT"), ("assistant", "พอร์ตของคุณมี PTT อยู่")]

        classify_query("อันนั้นล่ะ", chat_model=mock_chat_model, chat_history=history)

        call_args = mock_chat_model.invoke.call_args[0][0]
        assert len(call_args) == 4  # system + 2 history + current query
        assert call_args[1].content == "ดูพอร์ต PTT"
        assert isinstance(call_args[2], AIMessage)
        assert call_args[3].content == "อันนั้นล่ะ"

    def test_classify_without_history_has_two_messages(self, mock_chat_model: MagicMock) -> None:
        """No history keeps messages minimal (system + query)."""
        mock_chat_model.invoke.return_value = AIMessage(
            content='{"intent": "tax", "confidence": 0.9}'
        )

        classify_query("คำนวณภาษี", chat_model=mock_chat_model)

        call_args = mock_chat_model.invoke.call_args[0][0]
        assert len(call_args) == 2


class TestOrchestrateForwardsHistoryToClassifier:
    """Tests that orchestrate_query gives chat history to classify_query."""

    @patch("finance_ai.agents.router_agent.execute_tax_agent")
    @patch("finance_ai.agents.router_agent.classify_query")
    def test_classifier_receives_history(
        self, mock_classify: MagicMock, mock_execute: MagicMock
    ) -> None:
        """chat_history is forwarded to the classifier, not just the agent."""
        mock_classify.return_value = OrchestratorDecision(intent="tax", confidence=Decimal("0.9"))
        mock_execute.return_value = {"intent": "tax", "response": "ok"}
        history = [("user", "ก่อนหน้า")]

        orchestrate_query("ภาษี", chat_history=history)
        assert mock_classify.call_args[0][2] == history


class TestConfidenceThreshold:
    """Tests for low-confidence clarify-back behavior."""

    @patch("finance_ai.agents.router_agent.execute_expense_agent")
    @patch("finance_ai.agents.router_agent.classify_query")
    def test_low_confidence_returns_clarify(
        self, mock_classify: MagicMock, mock_execute: MagicMock
    ) -> None:
        """Low-confidence classification asks the user instead of guessing."""
        mock_classify.return_value = OrchestratorDecision(
            intent="expense", confidence=Decimal("0.5")
        )

        result = orchestrate_query("เงิน")
        assert result["intent"] == "clarify"
        mock_execute.assert_not_called()

    def test_clarify_mentions_categories(self) -> None:
        """Clarify response lists example categories in Thai."""
        result = _build_clarify_response()
        assert "รายจ่าย" in result["response"]
        assert "ภาษี" in result["response"]
        assert "หุ้น" in result["response"] or "กองทุน" in result["response"]

    @patch("finance_ai.agents.router_agent.execute_tax_agent")
    @patch("finance_ai.agents.router_agent.classify_query")
    def test_confidence_at_threshold_dispatches(
        self, mock_classify: MagicMock, mock_execute: MagicMock
    ) -> None:
        """Confidence exactly at the threshold dispatches normally."""
        mock_classify.return_value = OrchestratorDecision(
            intent="tax", confidence=ROUTER_CONFIDENCE_THRESHOLD
        )
        mock_execute.return_value = {"intent": "tax", "response": "ok"}

        result = orchestrate_query("คำนวณภาษี")
        assert result["intent"] == "tax"
        mock_execute.assert_called_once()

    @patch("finance_ai.agents.router_agent.execute_general_chat")
    @patch("finance_ai.agents.router_agent.classify_query")
    def test_unknown_intent_skips_clarify(
        self, mock_classify: MagicMock, mock_general_chat: MagicMock
    ) -> None:
        """Unknown intent goes to general chat regardless of confidence."""
        mock_classify.return_value = OrchestratorDecision(
            intent="unknown", confidence=Decimal("0.3")
        )
        mock_general_chat.return_value = {"intent": "general_chat", "response": "สวัสดีค่ะ"}

        result = orchestrate_query("สูตรผัดไทย")
        assert result["intent"] == "general_chat"
        mock_general_chat.assert_called_once()


class TestAssetHintResolution:
    """Tests for pre-route asset symbol search."""

    @patch("finance_ai.agents.router_agent.search_asset_symbols")
    def test_hint_included_when_symbol_found(
        self, mock_search: MagicMock, mock_chat_model: MagicMock
    ) -> None:
        """Router messages include the resolved symbol hint before the query."""
        mock_search.return_value = [
            AssetSymbolMatch(
                symbol="PTT.BK",
                name="PTT Public Company Limited",
                exchange="SET",
                quote_type="EQUITY",
            )
        ]
        mock_chat_model.invoke.return_value = AIMessage(
            content='{"intent": "asset_monitoring", "confidence": 0.9}'
        )

        classify_query("ราคา PTT เท่าไหร่", chat_model=mock_chat_model)

        mock_search.assert_called_once_with("PTT")
        call_args = mock_chat_model.invoke.call_args[0][0]
        hint_messages = [m for m in call_args if "PTT.BK" in str(m.content)]
        assert len(hint_messages) == 1

    @patch("finance_ai.agents.router_agent.search_asset_symbols")
    def test_no_search_for_thai_only_query(self, mock_search: MagicMock) -> None:
        """Thai-only queries skip the symbol search entirely."""
        assert _resolve_asset_hint("จ่ายค่ากาแฟแปดสิบบาท") is None
        mock_search.assert_not_called()

    @patch("finance_ai.agents.router_agent.search_asset_symbols")
    def test_no_hint_when_search_finds_nothing(self, mock_search: MagicMock) -> None:
        """Empty search results produce no hint."""
        mock_search.return_value = []
        assert _resolve_asset_hint("ราคา ABC เท่าไหร่") is None

    @patch("finance_ai.agents.router_agent.search_asset_symbols")
    def test_lowercase_tokens_are_not_searched(self, mock_search: MagicMock) -> None:
        """Lowercase English words (e.g. 'test') do not trigger a search."""
        assert _resolve_asset_hint("test the query") is None
        mock_search.assert_not_called()


def _mock_model_returning(content: str) -> MagicMock:
    """Build a MagicMock chat model whose invoke returns an AIMessage."""
    model = MagicMock()
    model.invoke.return_value = AIMessage(content=content)
    return model


class TestRouterAblationConfigDefaults:
    """Defaults must reproduce current production behavior."""

    def test_defaults_match_production(self) -> None:
        """All helpers on, threshold off (it lives in orchestrate_query)."""
        config = RouterAblationConfig()
        assert config.include_chat_history is True
        assert config.include_few_shot_examples is True
        assert config.resolve_asset_hint is True
        assert config.apply_confidence_threshold is False


class TestClassifyQueryAblationFlags:
    """Each flag must visibly change the messages sent to the LLM."""

    def test_few_shot_disabled_removes_examples_from_prompt(self) -> None:
        """System message has no example block when few-shot is off."""
        config = RouterAblationConfig(include_few_shot_examples=False)
        model = _mock_model_returning('{"intent": "tax", "confidence": 0.9}')
        classify_query("คำนวณภาษี", chat_model=model, config=config)
        sent_messages = model.invoke.call_args[0][0]
        assert "ตัวอย่าง:" not in sent_messages[0].content

    def test_few_shot_enabled_keeps_examples_in_prompt(self) -> None:
        """System message keeps examples when few-shot is on."""
        config = RouterAblationConfig()
        model = _mock_model_returning('{"intent": "tax", "confidence": 0.9}')
        classify_query("คำนวณภาษี", chat_model=model, config=config)
        sent_messages = model.invoke.call_args[0][0]
        assert "จ่ายค่ากาแฟ 80 บาท" in sent_messages[0].content

    def test_history_disabled_omits_history_messages(self) -> None:
        """Only system message + current query are sent when history is off."""
        config = RouterAblationConfig(include_chat_history=False)
        model = _mock_model_returning('{"intent": "tax", "confidence": 0.9}')
        history = [("user", "ก่อนหน้านี้ถามอะไร"), ("assistant", "ตอบอะไรไป")]
        classify_query("คำนวณภาษี", chat_model=model, chat_history=history, config=config)
        sent_messages = model.invoke.call_args[0][0]
        assert len(sent_messages) == 2
        assert isinstance(sent_messages[0], SystemMessage)
        assert isinstance(sent_messages[1], HumanMessage)

    def test_history_enabled_includes_history_messages(self) -> None:
        """History messages are sent when history is on."""
        config = RouterAblationConfig()
        model = _mock_model_returning('{"intent": "tax", "confidence": 0.9}')
        history = [("user", "ก่อนหน้านี้ถามอะไร"), ("assistant", "ตอบอะไรไป")]
        classify_query("คำนวณภาษี", chat_model=model, chat_history=history, config=config)
        sent_messages = model.invoke.call_args[0][0]
        assert len(sent_messages) == 4  # system + 2 history + query

    def test_asset_hint_disabled_skips_symbol_search(self) -> None:
        """Yahoo symbol search is never called when the hint flag is off."""
        config = RouterAblationConfig(resolve_asset_hint=False)
        model = _mock_model_returning('{"intent": "asset_monitoring", "confidence": 0.9}')
        with patch("finance_ai.agents.router_agent.search_asset_symbols") as mock_search:
            classify_query("ดูราคา PTT ล่าสุด", chat_model=model, config=config)
        mock_search.assert_not_called()

    def test_asset_hint_enabled_calls_symbol_search(self) -> None:
        """Yahoo symbol search runs and injects a hint SystemMessage."""
        config = RouterAblationConfig()
        model = _mock_model_returning('{"intent": "asset_monitoring", "confidence": 0.9}')
        match = AssetSymbolMatch(
            symbol="PTT.BK",
            name="PTT Public Company Limited",
            exchange="SET",
            quote_type="EQUITY",
        )
        with patch(
            "finance_ai.agents.router_agent.search_asset_symbols",
            return_value=[match],
        ) as mock_search:
            classify_query("ดูราคา PTT ล่าสุด", chat_model=model, config=config)
        mock_search.assert_called_once()
        sent_messages = model.invoke.call_args[0][0]
        hint_messages = [
            m for m in sent_messages if isinstance(m, SystemMessage) and "หลักทรัพย์" in m.content
        ]
        assert len(hint_messages) == 1

    def test_default_config_behaves_like_no_config(self) -> None:
        """Passing the default config sends identical messages to None."""
        model_a = _mock_model_returning('{"intent": "tax", "confidence": 0.9}')
        model_b = _mock_model_returning('{"intent": "tax", "confidence": 0.9}')
        history = [("user", "สวัสดี")]
        with patch("finance_ai.agents.router_agent.search_asset_symbols"):
            classify_query("คำนวณภาษี", chat_model=model_a, chat_history=history)
            classify_query(
                "คำนวณภาษี",
                chat_model=model_b,
                chat_history=history,
                config=DEFAULT_ABLATION_CONFIG,
            )
        messages_a = [m.content for m in model_a.invoke.call_args[0][0]]
        messages_b = [m.content for m in model_b.invoke.call_args[0][0]]
        assert messages_a == messages_b


class TestClassifyQueryConfidenceThreshold:
    """Threshold flag mirrors orchestrate_query's clarify-back rule."""

    def test_threshold_disabled_returns_low_confidence_decision(self) -> None:
        """Default config does NOT clarify (matches current behavior)."""
        model = _mock_model_returning('{"intent": "tax", "confidence": 0.5}')
        decision = classify_query("คำนวณภาษี", chat_model=model)
        assert decision.intent == "tax"

    def test_threshold_enabled_converts_low_confidence_to_clarify(self) -> None:
        """Below-threshold decisions become intent='clarify'."""
        config = RouterAblationConfig(apply_confidence_threshold=True)
        model = _mock_model_returning('{"intent": "tax", "confidence": 0.5}')
        decision = classify_query("คำนวณภาษี", chat_model=model, config=config)
        assert decision.intent == "clarify"
        assert decision.confidence == Decimal("0.5")

    def test_threshold_enabled_keeps_high_confidence_decision(self) -> None:
        """Above-threshold decisions pass through unchanged."""
        config = RouterAblationConfig(apply_confidence_threshold=True)
        model = _mock_model_returning('{"intent": "tax", "confidence": 0.95}')
        decision = classify_query("คำนวณภาษี", chat_model=model, config=config)
        assert decision.intent == "tax"
        assert decision.confidence == Decimal("0.95")

    def test_threshold_never_clarifies_unknown_intent(self) -> None:
        """Unknown intent goes to general chat even below threshold."""
        config = RouterAblationConfig(apply_confidence_threshold=True)
        model = _mock_model_returning('{"intent": "unknown", "confidence": 0.2}')
        decision = classify_query("สอนทำผัดกระเพรา", chat_model=model, config=config)
        assert decision.intent == "unknown"
