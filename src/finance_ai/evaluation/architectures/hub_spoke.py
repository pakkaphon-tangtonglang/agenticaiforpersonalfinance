"""Hub-and-spoke design: the production router (one LLM call) picks one specialist."""

from finance_ai.agents.router_agent import orchestrate_query
from finance_ai.evaluation.architectures.design_context import (
    GENERAL_AGENT,
    DesignContext,
    DesignError,
    DesignOutcome,
    content_to_text,
)

_GENERAL_INTENTS = frozenset({"general", "general_chat", "unknown"})


def run_hub_spoke(query: str, context: DesignContext) -> DesignOutcome:
    """Answer a query exactly as production does.

    Args:
        query: User question.
        context: Model, user and database for the run.

    Returns:
        Answer and a one-step route (agent key or ``clarify``).

    Raises:
        DesignError: When production orchestration reports an error.

    Example:
        >>> run_hub_spoke("จ่ายค่ากาแฟ 80 บาท", context).route  # doctest: +SKIP
        ['expense']
    """
    result = orchestrate_query(
        query,
        chat_model=context.chat_model,
        user_id=context.user_id,
        db_session_factory=context.db_session_factory,
    )
    intent = str(result.get("intent", ""))
    answer = content_to_text(result.get("response", ""))
    if intent == "error":
        raise DesignError(answer)
    route = GENERAL_AGENT if intent in _GENERAL_INTENTS else intent
    return DesignOutcome(answer=answer, route=[route])
