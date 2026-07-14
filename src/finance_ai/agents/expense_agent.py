"""LangGraph Expense Agent for personal expense tracking.

Uses a ReAct-style graph: LLM reasons about the query, calls expense tools
when needed, then formats the result in Thai for the user.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from finance_ai.agents.cross_agent_tools import EXPENSE_CROSS_TOOLS
from finance_ai.agents.expense_tools import (
    add_expense,
    add_income,
    query_expenses_by_category,
    summarize_monthly_expenses,
)
from finance_ai.agents.graph_utils import should_continue
from finance_ai.agents.prompts import EXPENSE_AGENT_SYSTEM_PROMPT, get_date_context
from finance_ai.agents.rag_tool import search_finance_knowledge
from finance_ai.agents.schemas import ExpenseAgentState
from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

EXPENSE_TOOLS = [
    add_expense,
    add_income,
    summarize_monthly_expenses,
    query_expenses_by_category,
    search_finance_knowledge,
] + EXPENSE_CROSS_TOOLS


def create_llm_node(
    chat_model: BaseChatModel,
) -> Any:
    """Create the LLM reasoning node for the expense agent.

    Binds expense tools to the model and prepends the system prompt on each call.

    Args:
        chat_model: LangChain ChatModel to use.

    Returns:
        A callable node function for the LangGraph graph.

    Example:
        >>> node = create_llm_node(chat_model)
    """
    model_with_tools = chat_model.bind_tools(EXPENSE_TOOLS)

    def llm_node(state: ExpenseAgentState) -> dict[str, Any]:
        """Invoke the LLM with current messages and system prompt.

        Args:
            state: Current agent state with messages.

        Returns:
            Dict with updated messages list.
        """
        messages = [
            SystemMessage(content=get_date_context() + EXPENSE_AGENT_SYSTEM_PROMPT)
        ] + state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    return llm_node


def build_expense_agent_graph(
    chat_model: BaseChatModel | None = None,
) -> Any:
    """Build the LangGraph StateGraph for the Expense Agent.

    Creates a ReAct-style graph: agent -> (tool_calls?) -> tools -> agent -> END.

    Args:
        chat_model: Optional ChatModel override. Uses factory if None.

    Returns:
        Compiled LangGraph StateGraph.

    Example:
        >>> graph = build_expense_agent_graph()
        >>> result = graph.invoke({
        ...     "messages": [("user", "จ่ายค่ากาแฟ 80 บาท")],
        ...     "user_id": "abc-123",
        ... })
    """
    if chat_model is None:
        from finance_ai.agents.llm_factory import create_chat_model

        chat_model = create_chat_model()

    graph = StateGraph(ExpenseAgentState)
    graph.add_node("agent", create_llm_node(chat_model))
    graph.add_node("tools", ToolNode(EXPENSE_TOOLS))
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()
