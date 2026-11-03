"""Hierarchical design: level 1 picks a group, level 2 picks an agent inside it.

Both levels reuse the production router's intent descriptions, few-shot
examples, thinking level and 0.7 ask-back threshold, so the only change
from hub-and-spoke is the two-step routing structure.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from finance_ai.agents.llm_factory import router_chat_model
from finance_ai.agents.router_agent import _build_clarify_response  # pylint: disable=W0212
from finance_ai.evaluation.architectures.design_context import (
    CONFIDENCE_THRESHOLD,
    GENERAL_AGENT,
    DesignContext,
    DesignOutcome,
    parse_confidence,
    parse_json_reply,
    run_specialist,
)
from finance_ai.evaluation.architectures.intent_catalog import (
    describe_intents,
    examples_for,
    intent_examples,
)
from finance_ai.evaluation.architectures.run_result import CLARIFY_ROUTE

GROUPS: dict[str, tuple[str, ...]] = {
    "operations": ("tax", "expense"),
    "analysis": ("recommendation", "report"),
    "planning": ("planning", "asset_monitoring"),
    GENERAL_AGENT: (GENERAL_AGENT, "unknown"),
}

_VAGUE_RULE = (
    "ถ้าข้อความกว้างเกินไปจนเดาไม่ได้ (เช่น 'เงิน', 'การเงิน') ให้ confidence ไม่เกิน 0.5"
    " แต่ถ้าระบุหัวข้อชัดเจนแม้จะสั้น ให้จำแนกตามปกติ"
)


def build_group_prompt() -> str:
    """Build the level-1 prompt listing the four groups and their intents.

    Returns:
        Thai system prompt asking for ``{"group": ..., "confidence": ...}``.
    """
    group_lines = "\n\n".join(
        f'กลุ่ม "{group}":\n{describe_intents(list(intents))}' for group, intents in GROUPS.items()
    )
    group_of = {intent: group for group, intents in GROUPS.items() for intent in intents}
    examples = "\n".join(f'- "{query}" → {group_of[intent]}' for query, intent in intent_examples())
    return (
        "คุณเป็นตัวจัดกลุ่มคำถามทางการเงินระดับแรก เลือกกลุ่มใหญ่ที่คำถามอยู่\n\n"
        f"{group_lines}\n\n{_VAGUE_RULE}\n\nตัวอย่าง:\n{examples}\n\n"
        'ตอบเป็น JSON เท่านั้น: {"group": "<group>", "confidence": <0.0-1.0>}'
    )


def build_agent_prompt(group: str) -> str:
    """Build the level-2 prompt limited to the intents of one group.

    Args:
        group: Group key from GROUPS.

    Returns:
        Thai system prompt asking for ``{"intent": ..., "confidence": ...}``.
    """
    intents = [intent for intent in GROUPS[group] if intent != "unknown"]
    examples = examples_for(intents)
    example_block = f"\n\nตัวอย่าง:\n{examples}" if examples else ""
    return (
        "คุณเป็นตัวจำแนกคำถามทางการเงินระดับที่สอง เลือกหมวดจากรายการนี้เท่านั้น\n\n"
        f"{describe_intents(intents)}\n\n{_VAGUE_RULE}{example_block}\n\n"
        'ตอบเป็น JSON เท่านั้น: {"intent": "<category>", "confidence": <0.0-1.0>}'
    )


def run_hierarchical(query: str, context: DesignContext) -> DesignOutcome:
    """Route a query in two steps, then run the chosen specialist.

    Args:
        query: User question.
        context: Model, user and database for the run.

    Returns:
        Answer and route (agent key or ``clarify``).

    Example:
        >>> run_hierarchical("จ่ายค่ากาแฟ 80 บาท", context).route  # doctest: +SKIP
        ['expense']
    """
    group, group_confident = _classify(build_group_prompt(), "group", query, context)
    if not group_confident or group not in GROUPS:
        return _clarify()
    if group == GENERAL_AGENT:
        return DesignOutcome(run_specialist(GENERAL_AGENT, query, context), [GENERAL_AGENT])
    agent, agent_confident = _classify(build_agent_prompt(group), "intent", query, context)
    if not agent_confident:
        return _clarify()
    if agent not in GROUPS[group]:
        agent = GENERAL_AGENT
    return DesignOutcome(run_specialist(agent, query, context), [agent])


def _classify(prompt: str, key: str, query: str, context: DesignContext) -> tuple[str, bool]:
    """Make one routing call and return (choice, confidence >= threshold)."""
    model = router_chat_model(context.chat_model)
    reply = model.invoke([SystemMessage(content=prompt), HumanMessage(content=query)])
    parsed = parse_json_reply(reply.content)
    confident = parse_confidence(parsed.get("confidence")) >= CONFIDENCE_THRESHOLD
    return str(parsed.get(key, "")), confident


def _clarify() -> DesignOutcome:
    """Return the production ask-back message."""
    return DesignOutcome(str(_build_clarify_response()["response"]), [CLARIFY_ROUTE])
