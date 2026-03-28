"""Agent implementations for the multi-agent system.

Provides LangGraph-based agents for financial query processing:
- Orchestrator Agent: Classifies user queries and routes to specialized agents.
- Tax Agent: Calculates Thai personal income tax using LangGraph ReAct pattern.
- Expense Agent: Tracks and summarizes personal expenses using LangGraph ReAct pattern.
- Asset Monitoring Agent: Tracks portfolio and monitors asset news/events.
"""

from finance_ai.agents.expense_agent import build_expense_agent_graph
from finance_ai.agents.asset_monitoring_agent import build_asset_monitoring_agent_graph
from finance_ai.agents.llm_factory import create_chat_model
from finance_ai.agents.router_agent import orchestrate_query
from finance_ai.agents.tax_agent import build_tax_agent_graph

__all__ = [
    "build_expense_agent_graph",
    "build_asset_monitoring_agent_graph",
    "build_tax_agent_graph",
    "create_chat_model",
    "orchestrate_query",
]
