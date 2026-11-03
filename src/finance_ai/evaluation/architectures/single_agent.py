"""Single-agent design: one ReAct agent with every specialist's tools and prompt.

No routing step. The six production specialist prompts are joined under
headings (the shared anti-hallucination rules appear once), all 27 unique
tools are bound, and the recommendation guardrail checks the final answer,
since that safety check is not a routing feature.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from finance_ai.agents.asset_monitoring_agent import ASSET_MONITORING_TOOLS
from finance_ai.agents.expense_agent import EXPENSE_TOOLS
from finance_ai.agents.graph_utils import build_tool_node, should_continue
from finance_ai.agents.planning_agent import PLANNING_TOOLS
from finance_ai.agents.prompts import ANTI_HALLUCINATION_RULES, get_date_context
from finance_ai.agents.recommendation_agent import (
    RECOMMENDATION_AGENT_TOOLS,
    create_guardrail_node,
)
from finance_ai.agents.report_agent import REPORT_AGENT_TOOLS
from finance_ai.agents.schemas import RecommendationAgentState
from finance_ai.agents.tax_agent import TAX_TOOLS
from finance_ai.evaluation.architectures.design_context import (
    DesignContext,
    DesignOutcome,
    content_to_text,
)
from finance_ai.evaluation.architectures.peer_to_peer import AGENT_PROMPTS

_SECTION_TITLES: dict[str, str] = {
    "tax": "ด้านภาษี",
    "expense": "ด้านรายรับรายจ่าย",
    "asset_monitoring": "ด้านติดตามสินทรัพย์",
    "planning": "ด้านวางแผนการเงิน",
    "recommendation": "ด้านคำแนะนำการเงิน",
    "report": "ด้านรายงานการเงิน",
}


def all_tools() -> list[Any]:
    """Return every specialist tool once, in a stable order.

    Returns:
        Unique tools from all six specialist agents.

    Example:
        >>> len(all_tools())
        27
    """
    unique: dict[str, Any] = {}
    for group in (
        TAX_TOOLS,
        EXPENSE_TOOLS,
        ASSET_MONITORING_TOOLS,
        PLANNING_TOOLS,
        RECOMMENDATION_AGENT_TOOLS,
        REPORT_AGENT_TOOLS,
    ):
        for tool in group:
            unique.setdefault(tool.name, tool)
    return list(unique.values())


def build_single_agent_prompt() -> str:
    """Join the six specialist prompts into one system prompt.

    Returns:
        Thai system prompt covering every domain.
    """
    sections = "\n\n".join(
        f"## {_SECTION_TITLES[agent]}\n{prompt.replace(ANTI_HALLUCINATION_RULES, '').strip()}"
        for agent, prompt in AGENT_PROMPTS.items()
    )
    return (
        "คุณเป็นผู้ช่วยการเงินส่วนบุคคลที่ดูแลทุกด้านด้วยตัวเอง"
        " ใช้คำแนะนำของด้านที่ตรงกับคำถาม และเรียกเครื่องมือที่เหมาะสม\n"
        "ถ้าข้อความกว้างเกินไปจนไม่รู้ว่าผู้ใช้ต้องการอะไร ให้ถามกลับก่อน\n\n"
        f"{sections}\n{ANTI_HALLUCINATION_RULES}"
    )


def build_single_agent_graph(
    chat_model: BaseChatModel,
) -> CompiledStateGraph[RecommendationAgentState, None, Any, Any]:
    """Build the single agent: agent ⇄ tools, then guardrail.

    Args:
        chat_model: Model for the agent.

    Returns:
        Compiled graph.
    """
    tools = all_tools()
    model_with_tools = chat_model.bind_tools(tools)
    system_prompt = build_single_agent_prompt()

    def agent_node(state: RecommendationAgentState) -> dict[str, Any]:
        """Call the model with the joined prompt and all tools."""
        messages = [SystemMessage(content=get_date_context() + system_prompt)] + state["messages"]
        return {"messages": [model_with_tools.invoke(messages)]}

    graph = StateGraph(RecommendationAgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", build_tool_node(tools))
    graph.add_node("guardrail", create_guardrail_node())
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": "guardrail"})
    graph.add_edge("tools", "agent")
    graph.add_edge("guardrail", END)
    return graph.compile()


def run_single_agent(query: str, context: DesignContext) -> DesignOutcome:
    """Answer a query with the single agent.

    Args:
        query: User question.
        context: Model, user and database for the run.

    Returns:
        Answer with an empty route (no routing step).

    Example:
        >>> run_single_agent("จ่ายค่ากาแฟ 80 บาท", context).route  # doctest: +SKIP
        []
    """
    graph = build_single_agent_graph(context.chat_model)
    state = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "user_id": context.user_id,
            "db_session_factory": context.db_session_factory,
        }
    )
    return DesignOutcome(answer=content_to_text(state["messages"][-1].content), route=[])
