"""Peer-to-peer design: no router; agents forward the question among themselves.

The entry agent (round-robin by task position) reads the question with its own
production prompt plus a list of its peers and replies ``handle``,
``forward`` to a peer, or ``ask_user``. Up to MAX_HOPS forwards; the agent
that accepts runs its production graph.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from finance_ai.agents import prompts
from finance_ai.agents.llm_factory import router_chat_model
from finance_ai.agents.router_agent import _build_clarify_response  # pylint: disable=W0212
from finance_ai.evaluation.architectures.design_context import (
    DesignContext,
    DesignOutcome,
    parse_json_reply,
    run_specialist,
)
from finance_ai.evaluation.architectures.intent_catalog import SPECIALIST_AGENTS, describe_intents
from finance_ai.evaluation.architectures.run_result import CLARIFY_ROUTE

MAX_HOPS = 3

AGENT_PROMPTS: dict[str, str] = {
    "tax": prompts.TAX_AGENT_SYSTEM_PROMPT,
    "expense": prompts.EXPENSE_AGENT_SYSTEM_PROMPT,
    "asset_monitoring": prompts.ASSET_MONITORING_AGENT_SYSTEM_PROMPT,
    "planning": prompts.PLANNING_AGENT_SYSTEM_PROMPT,
    "recommendation": prompts.RECOMMENDATION_AGENT_SYSTEM_PROMPT,
    "report": prompts.REPORT_AGENT_SYSTEM_PROMPT,
}


def entry_agent_for(task_index: int) -> str:
    """Pick the entry agent round-robin, so each agent starts equally often.

    Args:
        task_index: Task position in the task file.

    Returns:
        One of the six specialist agents.

    Example:
        >>> entry_agent_for(7)
        'expense'
    """
    return SPECIALIST_AGENTS[task_index % len(SPECIALIST_AGENTS)]


def build_handoff_prompt(agent: str) -> str:
    """Build the decision prompt for one agent: its own prompt + its peers.

    Args:
        agent: The agent deciding.

    Returns:
        Thai system prompt asking for an ``action`` JSON reply.
    """
    peers = [peer for peer in SPECIALIST_AGENTS if peer != agent]
    return (
        f"{AGENT_PROMPTS[agent]}\n\n"
        "ก่อนตอบ ให้ตัดสินใจก่อนว่าคำถามล่าสุดอยู่ในหน้าที่ของคุณหรือไม่\n"
        f"เพื่อนร่วมทีมของคุณ:\n{describe_intents(peers)}\n\n"
        '- ถ้าเป็นหน้าที่ของคุณ ตอบ {"action": "handle"}\n'
        '- ถ้าเป็นหน้าที่ของเพื่อนร่วมทีม ตอบ {"action": "forward", "to": "<ชื่อ>"}\n'
        '- ถ้าข้อความกว้างเกินไปจนไม่รู้ว่าผู้ใช้ต้องการอะไร ตอบ {"action": "ask_user"}\n'
        "ตอบเป็น JSON เท่านั้น"
    )


def run_peer_to_peer(query: str, context: DesignContext) -> DesignOutcome:
    """Pass the query between agents until one accepts, then answer.

    Args:
        query: User question.
        context: Model, user, database and task position for the run.

    Returns:
        Answer and the route through the agents (ends with the answering agent
        or ``clarify``).

    Example:
        >>> run_peer_to_peer("จ่ายค่ากาแฟ 80 บาท", context).route  # doctest: +SKIP
        ['tax', 'expense']
    """
    current = entry_agent_for(context.task_index)
    route = [current]
    for _ in range(MAX_HOPS):
        action, target = _decide(current, query, context)
        if action == "ask_user":
            return DesignOutcome(
                str(_build_clarify_response()["response"]), route + [CLARIFY_ROUTE]
            )
        if action != "forward" or target not in SPECIALIST_AGENTS or target == current:
            break
        current = target
        route.append(current)
    return DesignOutcome(run_specialist(current, query, context), route)


def _decide(agent: str, query: str, context: DesignContext) -> tuple[str, str]:
    """Ask one agent whether to handle, forward or ask back."""
    model = router_chat_model(context.chat_model)
    reply = model.invoke(
        [SystemMessage(content=build_handoff_prompt(agent)), HumanMessage(content=query)]
    )
    parsed = parse_json_reply(reply.content)
    return str(parsed.get("action", "handle")), str(parsed.get("to", ""))
