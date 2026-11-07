"""Hub-and-spoke designs on the production router.

``hub_spoke`` sends every message to one specialist (the original design).
``hub_multi`` lets the router split a multi-topic message into one task per
specialist and joins the answers (production multi-intent routing).
"""

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
    return _run_production_router(query, context, allow_multi_intent=False)


def run_hub_multi(query: str, context: DesignContext) -> DesignOutcome:
    """Answer a query with multi-intent routing (one agent per topic).

    Args:
        query: User question.
        context: Model, user and database for the run.

    Returns:
        Answer and the agents that answered, in order (or ``clarify``).

    Raises:
        DesignError: When production orchestration reports an error.

    Example:
        >>> run_hub_multi("ภาษีเท่าไร แล้วควรออมเท่าไร", context).route  # doctest: +SKIP
        ['tax', 'planning']
    """
    return _run_production_router(query, context, allow_multi_intent=True)


def _run_production_router(
    query: str, context: DesignContext, allow_multi_intent: bool
) -> DesignOutcome:
    """Run orchestrate_query and turn its result into a design outcome."""
    result = orchestrate_query(
        query,
        chat_model=context.chat_model,
        user_id=context.user_id,
        db_session_factory=context.db_session_factory,
        allow_multi_intent=allow_multi_intent,
    )
    intent = str(result.get("intent", ""))
    answer = content_to_text(result.get("response", ""))
    if intent == "error":
        raise DesignError(answer)
    intents = [str(each) for each in result.get("intents") or [intent]]
    route = [GENERAL_AGENT if each in _GENERAL_INTENTS else each for each in intents]
    return DesignOutcome(answer=answer, route=route)
