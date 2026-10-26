"""Stream a Guru Agent answer for the Guru tab.

The Guru tab skips the router: the user already picked a guru, so the
query goes straight to the cached guru graph with guru_id in state.
"""

from collections.abc import Iterator
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from finance_ai.agents.graph_cache import get_compiled_graph
from finance_ai.agents.guru_personas import get_guru_persona
from finance_ai.agents.stream_utils import StreamEvent, stream_agent_response


def guru_intent(guru_id: str) -> str:
    """Return the intent label stored on guru messages.

    Args:
        guru_id: Persona id.

    Returns:
        Intent string such as "guru:buffett".

    Example:
        >>> guru_intent("buffett")
        'guru:buffett'
    """
    return f"guru:{guru_id}"


def stream_guru_query(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    query: str,
    guru_id: str,
    user_id: str,
    db_session_factory: Any,
    chat_model: BaseChatModel | None,
    chat_history: list[tuple[str, str]] | None = None,
) -> Iterator[StreamEvent]:
    """Stream one guru's answer to a query.

    Args:
        query: The user's question.
        guru_id: Persona id (e.g. "dalio").
        user_id: UUID of the user.
        db_session_factory: Session factory injected into tools.
        chat_model: Chat model; the factory default is used if None.
        chat_history: Earlier (role, content) turns of this guru chat.

    Yields:
        StreamEvent token and complete events.

    Raises:
        UnknownGuruError: If guru_id is not registered.
        RuntimeError: If the guru graph is missing from the graph cache.

    Example:
        >>> events = list(stream_guru_query("ตลาดตก", "buffett", "u1", factory, model))
    """
    get_guru_persona(guru_id)
    graph = get_compiled_graph("guru", chat_model)
    if graph is None:
        raise RuntimeError("Guru graph is not registered in graph_cache._BUILDER_MAP.")
    state = {
        "messages": list(chat_history or []) + [("user", query)],
        "user_id": user_id,
        "db_session_factory": db_session_factory,
        "guru_id": guru_id,
    }
    yield from stream_agent_response(graph, state, guru_intent(guru_id))
