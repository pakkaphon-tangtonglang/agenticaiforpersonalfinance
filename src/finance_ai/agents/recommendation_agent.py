"""LangGraph Recommendation Agent for proactive financial analysis.

Uses a ReAct-style graph: LLM analyzes user query, calls recommendation
tools to gather and analyze all financial data, then presents
prioritized recommendations in Thai.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

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

logger = get_logger(__name__)

RECOMMENDATION_AGENT_TOOLS = [
    generate_financial_recommendations,
    get_financial_health_score,
    search_finance_knowledge,
    search_finance_news,
    detect_psychological_cues,
]


def should_continue(state: RecommendationAgentState) -> str:
    """Determine next node: continue to tools or end.

    Args:
        state: Current agent state.

    Returns:
        "tools" if the last message has tool_calls, "end" otherwise.

    Example:
        >>> should_continue({"messages": [msg], "recommendation_result": None, "user_id": ""})
        'end'
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


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

    def llm_node(state: RecommendationAgentState) -> dict[str, Any]:
        """Invoke the LLM with current messages and system prompt.

        Args:
            state: Current agent state with messages.

        Returns:
            Dict with updated messages list.
        """
        messages = [
            SystemMessage(content=get_date_context() + RECOMMENDATION_AGENT_SYSTEM_PROMPT)
        ] + state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    return llm_node


def build_recommendation_agent_graph(
    chat_model: BaseChatModel | None = None,
) -> Any:
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
