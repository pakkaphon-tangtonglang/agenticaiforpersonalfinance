"""Graph caching module for LangGraph agent graphs.

Caches compiled graphs per (intent, chat_model) pair to avoid
rebuilding the graph on every query. This eliminates redundant
StateGraph construction and LLM model creation.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

_graph_cache: dict[str, CompiledStateGraph] = {}

_BUILDER_MAP: dict[str, str] = {
    "tax": "finance_ai.agents.tax_agent",
    "expense": "finance_ai.agents.expense_agent",
    "asset_monitoring": "finance_ai.agents.asset_monitoring_agent",
    "planning": "finance_ai.agents.planning_agent",
    "recommendation": "finance_ai.agents.recommendation_agent",
    "report": "finance_ai.agents.report_agent",
}

_BUILDER_FN: dict[str, str] = {
    "tax": "build_tax_agent_graph",
    "expense": "build_expense_agent_graph",
    "asset_monitoring": "build_asset_monitoring_agent_graph",
    "planning": "build_planning_agent_graph",
    "recommendation": "build_recommendation_agent_graph",
    "report": "build_report_agent_graph",
}


def get_compiled_graph(
    intent: str,
    chat_model: BaseChatModel,
) -> CompiledStateGraph | None:
    """Get a cached compiled graph for the given intent.

    Builds and caches the graph on first call per intent.
    Returns None for unrecognized intents.

    Args:
        intent: Agent intent string (tax, expense, etc.).
        chat_model: LangChain chat model to bind to the graph.

    Returns:
        CompiledStateGraph or None if intent is unknown.

    Example:
        >>> graph = get_compiled_graph("tax", model)
    """
    if intent not in _BUILDER_MAP:
        return None
    cache_key = intent
    if cache_key not in _graph_cache:
        logger.info("Building and caching graph for intent: %s", intent)
        builder = _load_builder(intent)
        _graph_cache[cache_key] = builder(chat_model)
    return _graph_cache[cache_key]


def clear_graph_cache() -> None:
    """Clear all cached graphs (for testing).

    Example:
        >>> clear_graph_cache()
    """
    _graph_cache.clear()


def _load_builder(intent: str) -> Any:
    """Dynamically load the graph builder function.

    Args:
        intent: Agent intent string.

    Returns:
        Graph builder callable.
    """
    import importlib

    module_path = _BUILDER_MAP[intent]
    fn_name = _BUILDER_FN[intent]
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)
