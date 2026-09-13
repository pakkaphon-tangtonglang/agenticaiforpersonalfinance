"""LangGraph Recommendation Agent for proactive financial analysis.

Uses a ReAct-style graph: LLM analyzes user query, calls recommendation
tools to gather and analyze all financial data, then presents
prioritized recommendations in Thai.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
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
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()
