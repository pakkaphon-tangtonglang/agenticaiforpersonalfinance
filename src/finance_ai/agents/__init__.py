"""Agent implementations for the multi-agent system.

Provides LangGraph-based agents for financial query processing:
- Router Agent: Classifies user queries and routes to specialized agents.
- Tax Agent: Calculates Thai personal income tax using LangGraph ReAct pattern.
- Expense Agent: Tracks and summarizes personal expenses using LangGraph ReAct pattern.
- Investment Agent: Tracks portfolio and provides investment recommendations.
"""

from finance_ai.agents.expense_agent import build_expense_agent_graph
from finance_ai.agents.investment_agent import build_investment_agent_graph
from finance_ai.agents.llm_factory import create_chat_model
from finance_ai.agents.router_agent import route_query
from finance_ai.agents.tax_agent import build_tax_agent_graph

__all__ = [
    "build_expense_agent_graph",
    "build_investment_agent_graph",
    "build_tax_agent_graph",
    "create_chat_model",
    "route_query",
]
