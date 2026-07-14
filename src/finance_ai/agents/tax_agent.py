"""LangGraph Tax Agent for Thai personal income tax calculations.

Uses a ReAct-style graph: LLM reasons about the query, calls calculate_thai_tax
tool when needed, then formats the result in Thai for the user.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from finance_ai.agents.cross_agent_tools import TAX_CROSS_TOOLS
from finance_ai.agents.graph_utils import should_continue
from finance_ai.agents.prompts import TAX_AGENT_SYSTEM_PROMPT, get_date_context
from finance_ai.agents.rag_tool import search_finance_knowledge
from finance_ai.agents.schemas import TaxAgentState
from finance_ai.agents.tax_tools import calculate_thai_tax
from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

TAX_TOOLS = [calculate_thai_tax, search_finance_knowledge] + TAX_CROSS_TOOLS


def _create_first_turn_node(
    chat_model: BaseChatModel,
) -> Any:
    """Create LLM node that forces tool calling on first turn.

    Forces the model to call a tool (e.g., calculate_thai_tax)
    instead of computing tax by itself, which produces wrong results.

    Args:
        chat_model: LangChain ChatModel to use.

    Returns:
        A callable node function for the LangGraph graph.
    """
    model_force_tool = chat_model.bind_tools([calculate_thai_tax], tool_choice="any")

    def first_turn_node(state: TaxAgentState) -> dict[str, Any]:
        """Invoke LLM with forced tool calling.

        Args:
            state: Current agent state with messages.

        Returns:
            Dict with updated messages list.
        """
        messages = [SystemMessage(content=get_date_context() + TAX_AGENT_SYSTEM_PROMPT)] + state[
            "messages"
        ]
        response = model_force_tool.invoke(messages)
        return {"messages": [response]}

    return first_turn_node


def _create_respond_node(
    chat_model: BaseChatModel,
) -> Any:
    """Create LLM node for responding after tool results.

    After tools have run, this node formats the results
    in Thai without being forced to call tools again.

    Args:
        chat_model: LangChain ChatModel to use.

    Returns:
        A callable node function for the LangGraph graph.
    """
    model_with_tools = chat_model.bind_tools(TAX_TOOLS)

    def respond_node(state: TaxAgentState) -> dict[str, Any]:
        """Invoke LLM to format tool results.

        Args:
            state: Current agent state with messages.

        Returns:
            Dict with updated messages list.
        """
        messages = [SystemMessage(content=get_date_context() + TAX_AGENT_SYSTEM_PROMPT)] + state[
            "messages"
        ]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    return respond_node


def build_tax_agent_graph(
    chat_model: BaseChatModel | None = None,
) -> CompiledStateGraph[TaxAgentState, Any]:
    """Build the LangGraph StateGraph for the Tax Agent.

    Graph flow: first_turn (force tool) -> tools -> respond -> END.
    The first turn always calls a tool to ensure accurate calculation.
    After tool results, the respond node formats the answer.

    Args:
        chat_model: Optional ChatModel override. Uses factory if None.

    Returns:
        Compiled LangGraph StateGraph.

    Example:
        >>> graph = build_tax_agent_graph()
        >>> result = graph.invoke({"messages": [("user", "คำนวณภาษี...")]})
    """
    if chat_model is None:
        from finance_ai.agents.llm_factory import create_chat_model

        chat_model = create_chat_model()

    graph = StateGraph(TaxAgentState)
    graph.add_node("first_turn", _create_first_turn_node(chat_model))
    graph.add_node("tools", ToolNode(TAX_TOOLS))
    graph.add_node("respond", _create_respond_node(chat_model))
    graph.set_entry_point("first_turn")
    graph.add_edge("first_turn", "tools")
    graph.add_edge("tools", "respond")
    graph.add_conditional_edges(
        "respond",
        should_continue,
        {"tools": "tools", "end": END},
    )
    return graph.compile()
