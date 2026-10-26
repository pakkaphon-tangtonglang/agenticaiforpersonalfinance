"""Tests for the LangGraph Guru Agent."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from finance_ai.agents.graph_cache import get_compiled_graph
from finance_ai.agents.guru_agent import (
    GURU_TOOLS,
    GURU_WRITE_TOOL_NAMES,
    build_guru_agent_graph,
    create_guru_llm_node,
    find_unconfirmed_write_calls,
)
from finance_ai.agents.guru_personas import get_guru_persona
from finance_ai.agents.prompts import build_guru_system_prompt


def _guru_state(guru_id: str, text: str = "ตลาดตกควรทำอย่างไร") -> dict[str, object]:
    """Build a minimal guru graph state."""
    return {
        "messages": [HumanMessage(content=text)],
        "user_id": "test-user",
        "db_session_factory": None,
        "guru_id": guru_id,
    }


def _create_goal_call() -> dict[str, object]:
    """A create_financial_goal tool call as the model would emit it."""
    return {
        "id": "w1",
        "name": "create_financial_goal",
        "args": {"goal_type": "retirement", "name": "เกษียณ", "target_amount": "5000000"},
    }


class TestFindUnconfirmedWriteCalls:
    """Tests for the code-level write confirmation guard."""

    def test_write_without_confirmation_is_flagged(self) -> None:
        """A write call is blocked when the user has not typed ยืนยัน."""
        messages = [
            HumanMessage(content="อยากเก็บเงินเกษียณ"),
            AIMessage(content="", tool_calls=[_create_goal_call()]),
        ]

        assert [call["id"] for call in find_unconfirmed_write_calls(messages)] == ["w1"]

    def test_write_after_confirmation_is_allowed(self) -> None:
        """The latest user message containing ยืนยัน unlocks writes."""
        messages = [
            HumanMessage(content="ยืนยัน สร้างเป้าหมายได้เลย"),
            AIMessage(content="", tool_calls=[_create_goal_call()]),
        ]

        assert not find_unconfirmed_write_calls(messages)

    def test_negated_confirmation_still_blocks(self) -> None:
        """'ไม่ยืนยัน' (do not confirm) must not unlock writes."""
        messages = [
            HumanMessage(content="ไม่ยืนยันนะ"),
            AIMessage(content="", tool_calls=[_create_goal_call()]),
        ]

        assert len(find_unconfirmed_write_calls(messages)) == 1

    def test_read_tools_never_need_confirmation(self) -> None:
        """Read-only calls pass through without confirmation."""
        read_call = {"id": "r1", "name": "get_goals_summary_cross", "args": {}}
        messages = [
            HumanMessage(content="ดูเป้าหมาย"),
            AIMessage(content="", tool_calls=[read_call]),
        ]

        assert not find_unconfirmed_write_calls(messages)


class TestGuruTools:
    """Tests for the guru tool list."""

    def test_includes_guru_search_and_user_data_tools(self) -> None:
        """The guru can read its sources and the user's own finances."""
        tool_names = {tool.name for tool in GURU_TOOLS}

        assert {
            "search_guru_knowledge",
            "get_guru_scorecard",
            "search_finance_knowledge",
            "get_financial_summary",
            "get_expense_summary_cross",
            "get_income_summary_cross",
            "get_goals_summary_cross",
        } <= tool_names

    def test_write_tools_are_only_validated_goal_tools(self) -> None:
        """Only existing CRUD-backed goal tools can write data."""
        tool_names = {tool.name for tool in GURU_TOOLS}

        assert GURU_WRITE_TOOL_NAMES == {"create_financial_goal", "update_savings_goal_cross"}
        assert GURU_WRITE_TOOL_NAMES <= tool_names


class TestCreateGuruLlmNode:
    """Tests for the persona-aware LLM node."""

    @patch("finance_ai.agents.guru_agent.get_risk_profile_context", return_value="")
    def test_uses_persona_prompt_from_state(
        self, _mock_risk: MagicMock, mock_chat_model: MagicMock
    ) -> None:
        """The system prompt is built for the guru named in state."""
        mock_chat_model.invoke.return_value = AIMessage(content="ok")
        node = create_guru_llm_node(mock_chat_model)

        node(_guru_state("dalio"))

        sent_messages = mock_chat_model.invoke.call_args[0][0]
        assert build_guru_system_prompt(get_guru_persona("dalio")) in sent_messages[0].content
        assert sent_messages[1].content == "ตลาดตกควรทำอย่างไร"

    @patch("finance_ai.agents.guru_agent.get_risk_profile_context")
    def test_adds_user_risk_profile(self, mock_risk: MagicMock, mock_chat_model: MagicMock) -> None:
        """The user's questionnaire risk level is part of the context."""
        mock_risk.return_value = "ระดับความเสี่ยงที่รับได้: ปานกลาง\n\n"
        mock_chat_model.invoke.return_value = AIMessage(content="ok")
        node = create_guru_llm_node(mock_chat_model)

        node(_guru_state("buffett"))

        system_prompt = mock_chat_model.invoke.call_args[0][0][0].content
        assert "ระดับความเสี่ยงที่รับได้: ปานกลาง" in system_prompt
        mock_risk.assert_called_once_with("test-user", None)

    @patch("finance_ai.agents.guru_agent.get_risk_profile_context", return_value="")
    def test_one_node_serves_every_persona(
        self, _mock_risk: MagicMock, mock_chat_model: MagicMock
    ) -> None:
        """Switching guru_id switches the persona without rebuilding the node."""
        mock_chat_model.invoke.return_value = AIMessage(content="ok")
        node = create_guru_llm_node(mock_chat_model)

        node(_guru_state("buffett"))
        node(_guru_state("coach_noom"))

        first_prompt = mock_chat_model.invoke.call_args_list[0][0][0][0].content
        second_prompt = mock_chat_model.invoke.call_args_list[1][0][0][0].content
        assert "Warren Buffett" in first_prompt
        assert "โค้ชหนุ่ม" in second_prompt


class TestBuildGuruAgentGraph:
    """Tests for graph construction and execution."""

    @patch("finance_ai.agents.guru_agent.get_risk_profile_context", return_value="")
    def test_graph_returns_model_answer(
        self, _mock_risk: MagicMock, mock_chat_model: MagicMock
    ) -> None:
        """A direct answer (no tool calls) ends the graph."""
        mock_chat_model.invoke.return_value = AIMessage(content="มุมมองแบบ Buffett ...")
        graph = build_guru_agent_graph(mock_chat_model)

        final_state = graph.invoke(_guru_state("buffett"))

        assert final_state["messages"][-1].content == "มุมมองแบบ Buffett ..."
        assert final_state["guru_id"] == "buffett"

    @patch("finance_ai.agents.guru_rag_tool.get_guru_retriever")
    @patch("finance_ai.agents.guru_agent.get_risk_profile_context", return_value="")
    def test_tool_call_searches_only_active_guru(
        self,
        _mock_risk: MagicMock,
        mock_get_retriever: MagicMock,
        mock_chat_model: MagicMock,
    ) -> None:
        """guru_id from state reaches search_guru_knowledge through ToolNode."""
        mock_get_retriever.return_value.retrieve_by_metadata.return_value = []
        mock_chat_model.invoke.side_effect = [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "search_guru_knowledge", "args": {"query": "ตลาดตก"}}
                ],
            ),
            AIMessage(content="สรุปคำตอบ"),
        ]
        graph = build_guru_agent_graph(mock_chat_model)

        graph.invoke(_guru_state("dalio"))

        mock_get_retriever.return_value.retrieve_by_metadata.assert_called_once_with(
            "ตลาดตก", {"guru_id": "dalio"}
        )

    @patch("finance_ai.agents.guru_agent.get_risk_profile_context", return_value="")
    def test_unconfirmed_write_is_blocked_in_graph(
        self, _mock_risk: MagicMock, mock_chat_model: MagicMock
    ) -> None:
        """A goal write without the user's 'ยืนยัน' never reaches the database tool."""
        mock_chat_model.invoke.side_effect = [
            AIMessage(content="", tool_calls=[_create_goal_call()]),
            AIMessage(content="ขอให้ยืนยันก่อนนะ"),
        ]
        graph = build_guru_agent_graph(mock_chat_model)

        with patch("finance_ai.agents.planning_tools.get_tool_session") as mock_session:
            final_state = graph.invoke(_guru_state("coach_noom", "อยากเก็บเงินเกษียณ"))

        mock_session.assert_not_called()
        tool_message = final_state["messages"][2]
        assert "ยืนยัน" in tool_message.content

    def test_graph_cache_builds_guru_graph(self, mock_chat_model: MagicMock) -> None:
        """The guru graph is registered in the shared graph cache."""
        assert get_compiled_graph("guru", mock_chat_model) is not None
