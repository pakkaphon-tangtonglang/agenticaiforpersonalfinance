"""LangGraph Asset Monitoring Agent for news tracking and asset price lookup.

Uses a ReAct-style graph: LLM reasons about the query, calls tools
when needed, then formats the result in Thai for the user.

Tools (4):
  - search_finance_news   : News Search Tool
  - get_stock_price       : Stock Price Tool
  - search_finance_knowledge : RAG News Summarization Tool
  - manage_watchlist      : Asset Tracking Database Tool
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from finance_ai.agents.asset_monitoring_tools import manage_watchlist
from finance_ai.agents.market_data_tools import MARKET_DATA_TOOLS
from finance_ai.agents.prompts import ASSET_MONITORING_AGENT_SYSTEM_PROMPT, get_date_context
from finance_ai.agents.rag_tool import search_finance_knowledge
from finance_ai.agents.schemas import AssetMonitoringAgentState
from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

ASSET_MONITORING_TOOLS = [
    manage_watchlist,
    search_finance_knowledge,
] + MARKET_DATA_TOOLS  # [get_stock_price, search_finance_news]


def should_continue(state: AssetMonitoringAgentState) -> str:
    """Determine next node: continue to tools or end.

    Args:
        state: Current agent state.

    Returns:
        "tools" if the last message has tool_calls, "end" otherwise.

    Example:
        >>> should_continue({"messages": [msg], "user_id": ""})
        'end'
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


def create_llm_node(
    chat_model: BaseChatModel,
) -> Any:
    """Create the LLM reasoning node for the asset monitoring agent.

    Args:
        chat_model: LangChain ChatModel to use.

    Returns:
        A callable node function for the LangGraph graph.

    Example:
        >>> node = create_llm_node(chat_model)
    """
    model_with_tools = chat_model.bind_tools(ASSET_MONITORING_TOOLS)

    def llm_node(state: AssetMonitoringAgentState) -> dict[str, Any]:
        """Invoke the LLM with current messages and system prompt.

        Args:
            state: Current agent state with messages.

        Returns:
            Dict with updated messages list.
        """
        messages = [
            SystemMessage(content=get_date_context() + ASSET_MONITORING_AGENT_SYSTEM_PROMPT)
        ] + state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    return llm_node


def build_asset_monitoring_agent_graph(
    chat_model: BaseChatModel | None = None,
) -> Any:
    """Build the LangGraph StateGraph for the Asset Monitoring Agent.

    Creates a ReAct-style graph: agent -> (tool_calls?) -> tools -> agent -> END.

    Args:
        chat_model: Optional ChatModel override. Uses factory if None.

    Returns:
        Compiled LangGraph StateGraph.

    Example:
        >>> graph = build_asset_monitoring_agent_graph()
        >>> result = graph.invoke({
        ...     "messages": [("user", "ข่าว PTT.BK ล่าสุด")],
        ...     "user_id": "abc-123",
        ... })
    """
    if chat_model is None:
        from finance_ai.agents.llm_factory import create_chat_model  # noqa: PLC0415

        chat_model = create_chat_model()

    graph = StateGraph(AssetMonitoringAgentState)
    graph.add_node("agent", create_llm_node(chat_model))
    graph.add_node("tools", ToolNode(ASSET_MONITORING_TOOLS))
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()
