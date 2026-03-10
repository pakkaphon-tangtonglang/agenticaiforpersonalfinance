"""LangGraph Investment Agent for portfolio tracking and recommendations.

Uses a ReAct-style graph: LLM reasons about the query, calls investment tools
when needed, then formats the result in Thai for the user.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from finance_ai.agents.cross_agent_tools import INVESTMENT_CROSS_TOOLS
from finance_ai.agents.investment_tools import (
    add_holding,
    get_investment_advice,
    import_csv,
    lookup_holding,
    refresh_prices,
    view_portfolio,
)
from finance_ai.agents.market_data_tools import MARKET_DATA_TOOLS
from finance_ai.agents.prompts import INVESTMENT_AGENT_SYSTEM_PROMPT
from finance_ai.agents.rag_tool import search_finance_knowledge
from finance_ai.agents.schemas import InvestmentAgentState
from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

INVESTMENT_TOOLS = (
    [
        view_portfolio,
        add_holding,
        import_csv,
        refresh_prices,
        lookup_holding,
        get_investment_advice,
        search_finance_knowledge,
    ]
    + INVESTMENT_CROSS_TOOLS
    + MARKET_DATA_TOOLS
)


def should_continue(state: InvestmentAgentState) -> str:
    """Determine next node: continue to tools or end.

    Args:
        state: Current agent state.

    Returns:
        "tools" if the last message has tool_calls, "end" otherwise.

    Example:
        >>> should_continue({"messages": [msg], "investment_result": None, "user_id": ""})
        'end'
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


def create_llm_node(
    chat_model: BaseChatModel,
) -> Any:
    """Create the LLM reasoning node for the investment agent.

    Binds investment tools to the model and prepends the system prompt on each call.

    Args:
        chat_model: LangChain ChatModel to use.

    Returns:
        A callable node function for the LangGraph graph.

    Example:
        >>> node = create_llm_node(chat_model)
    """
    model_with_tools = chat_model.bind_tools(INVESTMENT_TOOLS)

    def llm_node(state: InvestmentAgentState) -> dict[str, Any]:
        """Invoke the LLM with current messages and system prompt.

        Args:
            state: Current agent state with messages.

        Returns:
            Dict with updated messages list.
        """
        messages = [SystemMessage(content=INVESTMENT_AGENT_SYSTEM_PROMPT)] + state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    return llm_node


def build_investment_agent_graph(
    chat_model: BaseChatModel | None = None,
) -> Any:
    """Build the LangGraph StateGraph for the Investment Agent.

    Creates a ReAct-style graph: agent -> (tool_calls?) -> tools -> agent -> END.

    Args:
        chat_model: Optional ChatModel override. Uses factory if None.

    Returns:
        Compiled LangGraph StateGraph.

    Example:
        >>> graph = build_investment_agent_graph()
        >>> result = graph.invoke({
        ...     "messages": [("user", "ดูพอร์ตของฉัน")],
        ...     "user_id": "abc-123",
        ... })
    """
    if chat_model is None:
        from finance_ai.agents.llm_factory import create_chat_model  # noqa: PLC0415

        chat_model = create_chat_model()

    graph = StateGraph(InvestmentAgentState)
    graph.add_node("agent", create_llm_node(chat_model))
    graph.add_node("tools", ToolNode(INVESTMENT_TOOLS))
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()
