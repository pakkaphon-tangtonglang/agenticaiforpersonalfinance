"""LangGraph Report Agent for autonomous financial report generation.

Uses a ReAct-style graph: LLM gathers all financial data via tools,
then formats a comprehensive report covering all domains in Thai.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from finance_ai.agents.prompts import REPORT_AGENT_SYSTEM_PROMPT
from finance_ai.agents.rag_tool import search_finance_knowledge
from finance_ai.agents.report_tools import (
    generate_financial_report_tool,
    get_financial_summary,
)
from finance_ai.agents.schemas import ReportAgentState
from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

REPORT_AGENT_TOOLS = [
    generate_financial_report_tool,
    get_financial_summary,
    search_finance_knowledge,
]


def should_continue(state: ReportAgentState) -> str:
    """Determine next node: continue to tools or end.

    Args:
        state: Current agent state.

    Returns:
        "tools" if the last message has tool_calls, "end" otherwise.

    Example:
        >>> should_continue({"messages": [msg], "report_result": None, "user_id": ""})
        'end'
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


def create_llm_node(
    chat_model: BaseChatModel,
) -> Any:
    """Create the LLM reasoning node for the report agent.

    Binds report tools to the model and prepends the system prompt.

    Args:
        chat_model: LangChain ChatModel to use.

    Returns:
        A callable node function for the LangGraph graph.

    Example:
        >>> node = create_llm_node(chat_model)
    """
    model_with_tools = chat_model.bind_tools(REPORT_AGENT_TOOLS)

    def llm_node(state: ReportAgentState) -> dict[str, Any]:
        """Invoke the LLM with current messages and system prompt.

        Args:
            state: Current agent state with messages.

        Returns:
            Dict with updated messages list.
        """
        messages = [SystemMessage(content=REPORT_AGENT_SYSTEM_PROMPT)] + state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    return llm_node


def build_report_agent_graph(
    chat_model: BaseChatModel | None = None,
) -> Any:
    """Build the LangGraph StateGraph for the Report Agent.

    Creates a ReAct-style graph: agent -> (tool_calls?) -> tools -> agent -> END.

    Args:
        chat_model: Optional ChatModel override. Uses factory if None.

    Returns:
        Compiled LangGraph StateGraph.

    Example:
        >>> graph = build_report_agent_graph()
        >>> result = graph.invoke({
        ...     "messages": [("user", "สร้างรายงานการเงิน")],
        ...     "user_id": "abc-123",
        ... })
    """
    if chat_model is None:
        from finance_ai.agents.llm_factory import (  # noqa: PLC0415
            create_chat_model,
        )

        chat_model = create_chat_model()

    graph = StateGraph(ReportAgentState)
    graph.add_node("agent", create_llm_node(chat_model))
    graph.add_node("tools", ToolNode(REPORT_AGENT_TOOLS))
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()
