"""LangGraph Planning Agent for financial goal management.

Uses a ReAct-style graph: LLM reasons about the query, calls planning tools
when needed, then formats the result in Thai for the user.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from finance_ai.agents.cross_agent_tools import PLANNING_CROSS_TOOLS
from finance_ai.agents.planning_tools import (
    calculate_saving_plan,
    create_financial_goal,
    update_goal_progress,
    view_financial_goals,
)
from finance_ai.agents.prompts import PLANNING_AGENT_SYSTEM_PROMPT
from finance_ai.agents.rag_tool import search_finance_knowledge
from finance_ai.agents.schemas import PlanningAgentState
from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

PLANNING_TOOLS = [
    create_financial_goal,
    view_financial_goals,
    update_goal_progress,
    calculate_saving_plan,
    search_finance_knowledge,
] + PLANNING_CROSS_TOOLS


def should_continue(state: PlanningAgentState) -> str:
    """Determine next node: continue to tools or end.

    Args:
        state: Current agent state.

    Returns:
        "tools" if the last message has tool_calls, "end" otherwise.

    Example:
        >>> should_continue({"messages": [msg], "planning_result": None, "user_id": ""})
        'end'
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


def create_llm_node(
    chat_model: BaseChatModel,
) -> Any:
    """Create the LLM reasoning node for the planning agent.

    Binds planning tools to the model and prepends the system prompt on each call.

    Args:
        chat_model: LangChain ChatModel to use.

    Returns:
        A callable node function for the LangGraph graph.

    Example:
        >>> node = create_llm_node(chat_model)
    """
    model_with_tools = chat_model.bind_tools(PLANNING_TOOLS)

    def llm_node(state: PlanningAgentState) -> dict[str, Any]:
        """Invoke the LLM with current messages and system prompt.

        Args:
            state: Current agent state with messages.

        Returns:
            Dict with updated messages list.
        """
        messages = [SystemMessage(content=PLANNING_AGENT_SYSTEM_PROMPT)] + state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    return llm_node


def build_planning_agent_graph(
    chat_model: BaseChatModel | None = None,
) -> Any:
    """Build the LangGraph StateGraph for the Planning Agent.

    Creates a ReAct-style graph: agent -> (tool_calls?) -> tools -> agent -> END.

    Args:
        chat_model: Optional ChatModel override. Uses factory if None.

    Returns:
        Compiled LangGraph StateGraph.

    Example:
        >>> graph = build_planning_agent_graph()
        >>> result = graph.invoke({
        ...     "messages": [("user", "อยากออมเงิน 100,000 บาท")],
        ...     "user_id": "abc-123",
        ... })
    """
    if chat_model is None:
        from finance_ai.agents.llm_factory import create_chat_model

        chat_model = create_chat_model()

    graph = StateGraph(PlanningAgentState)
    graph.add_node("agent", create_llm_node(chat_model))
    graph.add_node("tools", ToolNode(PLANNING_TOOLS))
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()
