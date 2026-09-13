"""Tests for the Recommendation Agent LangGraph graph."""

from typing import Any
from unittest.mock import MagicMock, patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage

from finance_ai.agents.graph_utils import should_continue
from finance_ai.agents.recommendation_agent import (
    RECOMMENDATION_AGENT_TOOLS,
    build_recommendation_agent_graph,
    create_guardrail_node,
    create_llm_node,
    get_risk_profile_context,
    get_user_risk_level,
)


class TestShouldContinue:
    """Tests for the should_continue routing function."""

    def test_returns_tools_when_tool_calls_present(self) -> None:
        """When last message has tool_calls, should route to 'tools'."""
        message = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "generate_financial_recommendations", "args": {}}],
        )
        state = {
            "messages": [message],
            "user_id": "",
            "db_session_factory": None,
        }
        assert should_continue(state) == "tools"

    def test_returns_end_when_no_tool_calls(self) -> None:
        """When last message has no tool_calls, should route to 'end'."""
        message = AIMessage(content="คำแนะนำการเงิน: ...")
        state = {
            "messages": [message],
            "user_id": "",
            "db_session_factory": None,
        }
        assert should_continue(state) == "end"

    def test_returns_end_when_tool_calls_empty(self) -> None:
        """When tool_calls is empty list, should route to 'end'."""
        message = AIMessage(content="done", tool_calls=[])
        state = {
            "messages": [message],
            "user_id": "",
            "db_session_factory": None,
        }
        assert should_continue(state) == "end"


class TestCreateLlmNode:
    """Tests for the create_llm_node factory function."""

    def test_prepends_system_prompt(self, mock_chat_model: MagicMock) -> None:
        """System prompt should be prepended to messages."""
        mock_chat_model.invoke.return_value = AIMessage(content="response")
        node = create_llm_node(mock_chat_model)
        state: dict[str, Any] = {
            "messages": [("user", "วิเคราะห์การเงิน")],
            "user_id": "",
            "db_session_factory": None,
        }
        node(state)
        call_args = mock_chat_model.invoke.call_args[0][0]
        assert call_args[0].content  # System message is not empty

    def test_returns_messages_dict(self, mock_chat_model: MagicMock) -> None:
        """Node should return dict with 'messages' key."""
        response = AIMessage(content="result")
        mock_chat_model.invoke.return_value = response
        node = create_llm_node(mock_chat_model)
        state: dict[str, Any] = {
            "messages": [("user", "test")],
            "user_id": "",
            "db_session_factory": None,
        }
        result = node(state)
        assert "messages" in result
        assert result["messages"] == [response]

    def test_binds_recommendation_tools(self, mock_chat_model: MagicMock) -> None:
        """Model should have recommendation tools bound."""
        create_llm_node(mock_chat_model)
        mock_chat_model.bind_tools.assert_called_once_with(RECOMMENDATION_AGENT_TOOLS)


RISK_BLOCK_MARKERS = (
    "บริบทผู้ใช้ (แบบประเมินความเหมาะสมในการลงทุน)",
    "เสี่ยงสูงมาก (ระดับ 5, คะแนน 40)",
    "เมื่อให้คำแนะนำการลงทุน ให้เหมาะสมกับระดับความเสี่ยงนี้เสมอ",
)


class TestGetRiskProfileContext:
    """Tests for get_risk_profile_context."""

    def _make_assessment(self) -> MagicMock:
        """Build a mock latest assessment with level-5 attributes."""
        assessment = MagicMock()
        assessment.risk_category = "เสี่ยงสูงมาก"
        assessment.risk_level = 5
        assessment.total_score = 40
        return assessment

    def test_returns_context_block_when_profile_exists(self) -> None:
        """A stored assessment yields the Thai personalization block."""
        with patch("finance_ai.agents.recommendation_agent.RiskAssessmentCRUD") as mock_crud_class:
            mock_crud_class.return_value.get_latest_by_user.return_value = self._make_assessment()
            context = get_risk_profile_context("user-1", MagicMock())

        assert context is not None
        for marker in RISK_BLOCK_MARKERS:
            assert marker in context

    def test_returns_empty_string_when_no_profile(self) -> None:
        """No stored assessment yields an empty string."""
        with patch("finance_ai.agents.recommendation_agent.RiskAssessmentCRUD") as mock_crud_class:
            mock_crud_class.return_value.get_latest_by_user.return_value = None
            assert get_risk_profile_context("user-1", MagicMock()) == ""

    def test_returns_empty_string_on_database_error(self) -> None:
        """A database failure must never raise; it returns an empty string."""
        with patch("finance_ai.agents.recommendation_agent.RiskAssessmentCRUD") as mock_crud_class:
            mock_crud_class.return_value.get_latest_by_user.side_effect = RuntimeError("db down")
            assert get_risk_profile_context("user-1", MagicMock()) == ""


class TestLlmNodeRiskContext:
    """Tests for llm_node system-prompt personalization."""

    def _invoke_and_get_system_content(self, mock_chat_model: MagicMock) -> str:
        """Run llm_node and return the first (system) message content."""
        mock_chat_model.invoke.return_value = AIMessage(content="response")
        node = create_llm_node(mock_chat_model)
        state: dict[str, Any] = {
            "messages": [("user", "วิเคราะห์การเงิน")],
            "user_id": "user-1",
            "db_session_factory": MagicMock(),
        }
        node(state)
        call_args = mock_chat_model.invoke.call_args[0][0]
        return str(call_args[0].content)

    def test_includes_risk_block_when_profile_exists(self, mock_chat_model: MagicMock) -> None:
        """System content contains the risk block when a profile exists."""
        with patch("finance_ai.agents.recommendation_agent.RiskAssessmentCRUD") as mock_crud_class:
            mock_crud_class.return_value.get_latest_by_user.return_value = (
                TestGetRiskProfileContext()._make_assessment()
            )
            content = self._invoke_and_get_system_content(mock_chat_model)

        for marker in RISK_BLOCK_MARKERS:
            assert marker in content

    def test_omits_risk_block_when_no_profile(self, mock_chat_model: MagicMock) -> None:
        """System content has no risk markers when the user has no profile."""
        with patch("finance_ai.agents.recommendation_agent.RiskAssessmentCRUD") as mock_crud_class:
            mock_crud_class.return_value.get_latest_by_user.return_value = None
            content = self._invoke_and_get_system_content(mock_chat_model)

        for marker in RISK_BLOCK_MARKERS:
            assert marker not in content

    def test_survives_database_error(self, mock_chat_model: MagicMock) -> None:
        """A failing risk-profile lookup must not break llm_node."""
        with patch("finance_ai.agents.recommendation_agent.RiskAssessmentCRUD") as mock_crud_class:
            mock_crud_class.return_value.get_latest_by_user.side_effect = RuntimeError("db down")
            content = self._invoke_and_get_system_content(mock_chat_model)

        assert "คำแนะนำ" in content  # base system prompt still present


class TestBuildRecommendationAgentGraph:
    """Tests for build_recommendation_agent_graph function."""

    def test_builds_with_mock_model(self, mock_chat_model: MagicMock) -> None:
        """Should build graph successfully with a mock model."""
        graph = build_recommendation_agent_graph(mock_chat_model)
        assert graph is not None

    def test_has_agent_and_tools_nodes(self, mock_chat_model: MagicMock) -> None:
        """Compiled graph should have 'agent', 'tools', and 'guardrail' nodes."""
        graph = build_recommendation_agent_graph(mock_chat_model)
        node_names = list(graph.get_graph().nodes.keys())
        assert "agent" in node_names
        assert "tools" in node_names
        assert "guardrail" in node_names


class TestGetUserRiskLevel:
    """Tests for get_user_risk_level."""

    def _make_assessment(self, level: int) -> MagicMock:
        """Build a mock latest assessment with the given level."""
        assessment = MagicMock()
        assessment.risk_level = level
        return assessment

    def test_returns_level_when_profile_exists(self) -> None:
        """A stored assessment yields its integer risk level."""
        with patch("finance_ai.agents.recommendation_agent.RiskAssessmentCRUD") as mock_crud_class:
            mock_crud_class.return_value.get_latest_by_user.return_value = self._make_assessment(2)
            assert get_user_risk_level("user-1", MagicMock()) == 2

    def test_returns_none_when_no_profile(self) -> None:
        """No stored assessment yields None."""
        with patch("finance_ai.agents.recommendation_agent.RiskAssessmentCRUD") as mock_crud_class:
            mock_crud_class.return_value.get_latest_by_user.return_value = None
            assert get_user_risk_level("user-1", MagicMock()) is None

    def test_returns_none_on_database_error(self) -> None:
        """A database failure must never raise; it returns None."""
        with patch("finance_ai.agents.recommendation_agent.RiskAssessmentCRUD") as mock_crud_class:
            mock_crud_class.return_value.get_latest_by_user.side_effect = RuntimeError("db down")
            assert get_user_risk_level("user-1", MagicMock()) is None


class TestGuardrailNode:
    """Tests for the post-generation guardrail node."""

    def _state(self, *messages: Any) -> dict[str, Any]:
        """Build a minimal agent state from messages."""
        return {"messages": list(messages), "user_id": "user-1", "db_session_factory": MagicMock()}

    def test_appends_warning_to_final_answer(self) -> None:
        """A low-risk crypto answer gets a warning appended in-place."""
        answer = AIMessage(id="msg-1", content="แนะนำลงทุนในคริปโต")
        with patch("finance_ai.agents.recommendation_agent.get_user_risk_level", return_value=1):
            result = create_guardrail_node()(self._state(answer))
        assert result["messages"][0].id == "msg-1"
        assert "ความเสี่ยง" in result["messages"][0].content

    def test_clean_answer_unchanged(self) -> None:
        """A clean answer produces no new messages."""
        answer = AIMessage(id="msg-1", content="ควรทบทวนงบประมาณรายเดือน")
        with patch("finance_ai.agents.recommendation_agent.get_user_risk_level", return_value=1):
            result = create_guardrail_node()(self._state(answer))
        assert result["messages"] == []

    def test_no_assessment_no_warning(self) -> None:
        """Without a risk level, no mismatch warning is added."""
        answer = AIMessage(id="msg-1", content="แนะนำลงทุนในคริปโต")
        with patch("finance_ai.agents.recommendation_agent.get_user_risk_level", return_value=None):
            result = create_guardrail_node()(self._state(answer))
        assert result["messages"] == []

    def test_tool_call_message_ignored(self) -> None:
        """An intermediate AIMessage with tool_calls is not touched."""
        answer = AIMessage(
            id="msg-1",
            content="",
            tool_calls=[{"id": "c1", "name": "generate_financial_recommendations", "args": {}}],
        )
        with patch("finance_ai.agents.recommendation_agent.get_user_risk_level", return_value=1):
            result = create_guardrail_node()(self._state(answer))
        assert result["messages"] == []

    def test_tool_backed_numbers_not_flagged(self) -> None:
        """When tool results exist, % claims get no disclaimer."""
        tool_result = ToolMessage(content="ราคา PTT 30.25 บาท", tool_call_id="c1")
        answer = AIMessage(id="msg-2", content="หุ้น PTT ให้ผลตอบแทน 10% ต่อปี")
        with patch("finance_ai.agents.recommendation_agent.get_user_risk_level", return_value=3):
            result = create_guardrail_node()(self._state(tool_result, answer))
        assert result["messages"] == []


class TestRecommendationAgentTools:
    """Tests for the RECOMMENDATION_AGENT_TOOLS list."""

    def test_has_five_tools(self) -> None:
        """Should contain exactly 5 tools."""
        assert len(RECOMMENDATION_AGENT_TOOLS) == 5

    def test_contains_expected_tool_names(self) -> None:
        """Should contain recommendation, health, RAG, news, and psychology tools."""
        names = [t.name for t in RECOMMENDATION_AGENT_TOOLS]
        assert "generate_financial_recommendations" in names
        assert "get_financial_health_score" in names
        assert "search_finance_knowledge" in names
        assert "search_finance_news" in names
        assert "detect_psychological_cues" in names
