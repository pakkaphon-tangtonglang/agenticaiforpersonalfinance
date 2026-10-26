"""LangGraph Guru Agent: answers through a public guru's money philosophy.

One ReAct graph serves every persona. The persona is read from
state["guru_id"] on each LLM call, so a single cached graph handles
Buffett, Dalio, ดร.นิเวศน์ and โค้ชหนุ่ม.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from finance_ai.agents.cross_agent_tools import (
    get_expense_summary_cross,
    get_goals_summary_cross,
    get_income_summary_cross,
    update_savings_goal_cross,
)
from finance_ai.agents.graph_utils import should_continue
from finance_ai.agents.guru_personas import get_guru_persona
from finance_ai.agents.guru_rag_tool import search_guru_knowledge
from finance_ai.agents.guru_scorecard_tool import get_guru_scorecard
from finance_ai.agents.planning_tools import calculate_saving_plan, create_financial_goal
from finance_ai.agents.prompts import build_guru_system_prompt, get_date_context
from finance_ai.agents.rag_tool import search_finance_knowledge
from finance_ai.agents.recommendation_agent import get_risk_profile_context
from finance_ai.agents.report_tools import get_financial_summary
from finance_ai.agents.schemas import GuruAgentState
from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

GURU_WRITE_TOOL_NAMES = {"create_financial_goal", "update_savings_goal_cross"}
CONFIRMATION_WORD = "ยืนยัน"
NEGATED_CONFIRMATION = "ไม่ยืนยัน"

GURU_TOOLS = [
    search_guru_knowledge,
    get_guru_scorecard,
    search_finance_knowledge,
    get_financial_summary,
    get_expense_summary_cross,
    get_income_summary_cross,
    get_goals_summary_cross,
    calculate_saving_plan,
    create_financial_goal,
    update_savings_goal_cross,
]


def _latest_user_text(messages: list[Any]) -> str:
    """Return the text of the most recent human message.

    Args:
        messages: Conversation messages, oldest first.

    Returns:
        The latest human message content, or "" if none.
    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def find_unconfirmed_write_calls(messages: list[Any]) -> list[dict[str, Any]]:
    """Find write tool calls the user has not confirmed with 'ยืนยัน'.

    Enforced in code because a prompt rule alone cannot stop the model
    from writing to the user's data.

    Args:
        messages: Conversation messages; the last one holds the tool calls.

    Returns:
        The write tool calls to block (empty when all are allowed).

    Example:
        >>> find_unconfirmed_write_calls([HumanMessage("ยืนยัน"), ai_with_goal_call])
        []
    """
    tool_calls = getattr(messages[-1], "tool_calls", None) or []
    user_text = _latest_user_text(messages)
    if CONFIRMATION_WORD in user_text and NEGATED_CONFIRMATION not in user_text:
        return []
    return [call for call in tool_calls if call["name"] in GURU_WRITE_TOOL_NAMES]


def create_guarded_tools_node() -> Any:
    """Create a tools node that refuses unconfirmed write tool calls.

    When any write call is unconfirmed, every call in that turn gets an
    error ToolMessage so the model asks the user to confirm first.

    Returns:
        A callable node function for the LangGraph graph.

    Example:
        >>> node = create_guarded_tools_node()
    """
    tool_node = ToolNode(GURU_TOOLS)

    def tools_node(state: GuruAgentState) -> Any:
        """Run tools, or return confirmation errors for blocked writes."""
        if not find_unconfirmed_write_calls(state["messages"]):
            return tool_node.invoke(state)
        logger.info("Blocked unconfirmed guru write for user %s", state["user_id"])
        calls = state["messages"][-1].tool_calls
        return {"messages": [_confirmation_required_message(call) for call in calls]}

    return tools_node


def _confirmation_required_message(tool_call: dict[str, Any]) -> ToolMessage:
    """Build the error result for a tool call blocked by the guard.

    Args:
        tool_call: The blocked tool call.

    Returns:
        ToolMessage telling the model to get confirmation first.
    """
    return ToolMessage(
        content=(
            "ยังไม่ได้บันทึก: ต้องสรุปสิ่งที่จะบันทึกให้ผู้ใช้"
            f" แล้วรอให้ผู้ใช้พิมพ์ '{CONFIRMATION_WORD}' ก่อน"
        ),
        tool_call_id=tool_call["id"],
        name=tool_call["name"],
    )


def build_guru_context_prompt(state: GuruAgentState) -> str:
    """Build the full system prompt for the guru named in state.

    Args:
        state: Graph state with guru_id, user_id and db_session_factory.

    Returns:
        Date context + persona prompt + the user's risk profile (if any).

    Raises:
        UnknownGuruError: If state["guru_id"] is not registered.

    Example:
        >>> build_guru_context_prompt({"guru_id": "dalio", ...})
    """
    persona = get_guru_persona(state["guru_id"])
    risk_context = get_risk_profile_context(state["user_id"], state.get("db_session_factory"))
    return get_date_context() + risk_context + build_guru_system_prompt(persona)


def create_guru_llm_node(chat_model: BaseChatModel) -> Any:
    """Create the persona-aware LLM node for the guru agent.

    Args:
        chat_model: LangChain ChatModel to use.

    Returns:
        A callable node function for the LangGraph graph.

    Example:
        >>> node = create_guru_llm_node(chat_model)
    """
    model_with_tools = chat_model.bind_tools(GURU_TOOLS)

    def llm_node(state: GuruAgentState) -> dict[str, Any]:
        """Invoke the LLM with the active guru's system prompt.

        Args:
            state: Current agent state with messages and guru_id.

        Returns:
            Dict with updated messages list.
        """
        system_message = SystemMessage(content=build_guru_context_prompt(state))
        response = model_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    return llm_node


def build_guru_agent_graph(
    chat_model: BaseChatModel | None = None,
) -> CompiledStateGraph[GuruAgentState, None, Any, Any]:
    """Build the LangGraph StateGraph for the Guru Agent.

    Creates a ReAct-style graph: agent -> (tool_calls?) -> tools -> agent -> END.

    Args:
        chat_model: Optional ChatModel override. Uses factory if None.

    Returns:
        Compiled LangGraph StateGraph.

    Example:
        >>> graph = build_guru_agent_graph()
        >>> result = graph.invoke({
        ...     "messages": [("user", "ตลาดตกควรทำอย่างไร")],
        ...     "user_id": "abc-123", "db_session_factory": None, "guru_id": "buffett",
        ... })
    """
    if chat_model is None:
        from finance_ai.agents.llm_factory import create_chat_model

        chat_model = create_chat_model()

    graph = StateGraph(GuruAgentState)
    graph.add_node("agent", create_guru_llm_node(chat_model))
    graph.add_node("tools", create_guarded_tools_node())
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()
