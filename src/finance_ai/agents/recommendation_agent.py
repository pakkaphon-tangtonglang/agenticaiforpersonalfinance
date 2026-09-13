"""LangGraph Recommendation Agent for proactive financial analysis.

Uses a ReAct-style graph: LLM analyzes user query, calls recommendation
tools to gather and analyze all financial data, then presents
prioritized recommendations in Thai.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from finance_ai.agents.graph_utils import should_continue
from finance_ai.agents.market_data_tools import search_finance_news
from finance_ai.agents.prompts import RECOMMENDATION_AGENT_SYSTEM_PROMPT, get_date_context
from finance_ai.agents.psychology_tools import detect_psychological_cues
from finance_ai.agents.rag_tool import search_finance_knowledge
from finance_ai.agents.recommendation_tools import (
    generate_financial_recommendations,
    get_financial_health_score,
)
from finance_ai.agents.schemas import RecommendationAgentState
from finance_ai.core.logging import get_logger
from finance_ai.database.crud.risk_assessment_crud import RiskAssessmentCRUD
from finance_ai.agents.session_helper import get_tool_session
from finance_ai.tools.recommendation_guardrail import check_recommendation_response

logger = get_logger(__name__)

RECOMMENDATION_AGENT_TOOLS = [
    generate_financial_recommendations,
    get_financial_health_score,
    search_finance_knowledge,
    search_finance_news,
    detect_psychological_cues,
]


def get_risk_profile_context(user_id: str, db_session_factory: Any) -> str:
    """Build the Thai risk-profile context block from the user's latest assessment.

    Never raises: any failure (no DB session factory, connection error,
    missing table) is logged as a warning and an empty string is returned so
    that chat keeps working without personalization.

    Args:
        user_id: UUID string of the user.
        db_session_factory: Session factory used by get_tool_session.

    Returns:
        Thai context block ending with a blank line, or "" when unavailable.

    Example:
        >>> get_risk_profile_context("abc-123", session_factory)
        'บริบทผู้ใช้ ...'
    """
    try:
        with get_tool_session(db_session_factory) as session:
            assessment = RiskAssessmentCRUD().get_latest_by_user(session, user_id)
        if assessment is None:
            return ""
        return (
            "บริบทผู้ใช้ (แบบประเมินความเหมาะสมในการลงทุน):\n"
            f"ระดับความเสี่ยงที่รับได้: {assessment.risk_category} "
            f"(ระดับ {assessment.risk_level}, คะแนน {assessment.total_score})\n"
            "เมื่อให้คำแนะนำการลงทุน ให้เหมาะสมกับระดับความเสี่ยงนี้เสมอ\n\n"
        )
    except Exception as error:  # noqa: BLE001  # chat must never crash on context
        logger.warning("Failed to load risk profile context for user %s: %s", user_id, error)
        return ""


def get_user_risk_level(user_id: str, db_session_factory: Any) -> int | None:
    """Load the user's latest questionnaire risk level (never raises).

    Args:
        user_id: UUID string of the user.
        db_session_factory: Session factory used by get_tool_session.

    Returns:
        Risk level (1-5), or None when unavailable or on any error.

    Example:
        >>> get_user_risk_level("abc-123", session_factory)
        3
    """
    try:
        with get_tool_session(db_session_factory) as session:
            assessment = RiskAssessmentCRUD().get_latest_by_user(session, user_id)
    except Exception as error:  # noqa: BLE001  # chat must never crash on context
        logger.warning("Failed to load risk level for user %s: %s", user_id, error)
        return None
    return None if assessment is None else int(assessment.risk_level)


def create_guardrail_node() -> Any:
    """Create the deterministic post-generation guardrail node.

    Appends suitability/data-quality warnings to the agent's final
    answer based on the user's questionnaire risk level and whether the
    conversation used any tool results. Message id is preserved so
    LangGraph replaces the message in place instead of appending.

    Returns:
        A callable node function for the LangGraph graph.

    Example:
        >>> node = create_guardrail_node()
    """

    def guardrail_node(state: RecommendationAgentState) -> dict[str, Any]:
        """Append warnings to a clean final answer when needed.

        Args:
            state: Current agent state with messages and user context.

        Returns:
            Dict with a replaced final message, or empty messages.
        """
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or last_message.tool_calls:
            return {"messages": []}
        had_tool_results = any(isinstance(m, ToolMessage) for m in state["messages"])
        risk_level = get_user_risk_level(state["user_id"], state["db_session_factory"])
        answer = str(last_message.content)
        warning = check_recommendation_response(answer, risk_level, had_tool_results)
        if not warning:
            return {"messages": []}
        return {"messages": [AIMessage(id=last_message.id, content=f"{answer}\n\n{warning}")]}

    return guardrail_node


def create_llm_node(
    chat_model: BaseChatModel,
) -> Any:
    """Create the LLM reasoning node for the recommendation agent.

    Binds recommendation tools to the model and prepends the system prompt.

    Args:
        chat_model: LangChain ChatModel to use.

    Returns:
        A callable node function for the LangGraph graph.

    Example:
        >>> node = create_llm_node(chat_model)
    """
    model_with_tools = chat_model.bind_tools(RECOMMENDATION_AGENT_TOOLS)

    def _build_system_content(state: RecommendationAgentState) -> str:
        """Compose the system prompt: date, risk context, and base prompt.

        Args:
            state: Current agent state with user_id and db_session_factory.

        Returns:
            The full system prompt content for the LLM.
        """
        risk_context = get_risk_profile_context(state["user_id"], state["db_session_factory"])
        return get_date_context() + risk_context + RECOMMENDATION_AGENT_SYSTEM_PROMPT

    def llm_node(state: RecommendationAgentState) -> dict[str, Any]:
        """Invoke the LLM with current messages and system prompt.

        Args:
            state: Current agent state with messages.

        Returns:
            Dict with updated messages list.
        """
        messages = [SystemMessage(content=_build_system_content(state))] + state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    return llm_node


def build_recommendation_agent_graph(
    chat_model: BaseChatModel | None = None,
) -> CompiledStateGraph[RecommendationAgentState, None, Any, Any]:
    """Build the LangGraph StateGraph for the Recommendation Agent.

    Creates a ReAct-style graph: agent -> (tool_calls?) -> tools -> agent -> END.

    Args:
        chat_model: Optional ChatModel override. Uses factory if None.

    Returns:
        Compiled LangGraph StateGraph.

    Example:
        >>> graph = build_recommendation_agent_graph()
        >>> result = graph.invoke({
        ...     "messages": [("user", "วิเคราะห์การเงินของฉัน")],
        ...     "user_id": "abc-123",
        ... })
    """
    if chat_model is None:
        from finance_ai.agents.llm_factory import (
            create_chat_model,
        )  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        chat_model = create_chat_model()

    graph = StateGraph(RecommendationAgentState)
    graph.add_node("agent", create_llm_node(chat_model))
    graph.add_node("tools", ToolNode(RECOMMENDATION_AGENT_TOOLS))
    graph.add_node("guardrail", create_guardrail_node())
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": "guardrail"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("guardrail", END)
    return graph.compile()
