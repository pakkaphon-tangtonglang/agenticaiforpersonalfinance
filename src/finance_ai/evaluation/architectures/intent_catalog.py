"""Intent descriptions and few-shot examples taken from the production router prompt.

The hierarchical and peer-to-peer designs reuse this exact wording so that
the only difference from hub-and-spoke is the routing structure, not the
prompt quality.
"""

import re

from finance_ai.agents.prompts import ORCHESTRATOR_EXAMPLES_SECTION, ORCHESTRATOR_PROMPT_HEAD

SPECIALIST_AGENTS: tuple[str, ...] = (
    "tax",
    "expense",
    "asset_monitoring",
    "planning",
    "recommendation",
    "report",
)

_DESCRIPTION_PATTERN = re.compile(r'^- "(\w+)": (.*?)(?=^- "|^\n|\Z)', re.MULTILINE | re.DOTALL)
_EXAMPLE_PATTERN = re.compile(r'^- "(.+?)" → \{"intent": "(\w+)".*$', re.MULTILINE)


def intent_descriptions() -> dict[str, str]:
    """Return {intent: Thai description} from the production router prompt.

    Returns:
        One entry per intent listed in the router prompt.

    Example:
        >>> "tax" in intent_descriptions()
        True
    """
    return {
        intent: " ".join(text.split())
        for intent, text in _DESCRIPTION_PATTERN.findall(ORCHESTRATOR_PROMPT_HEAD)
    }


def intent_examples() -> list[tuple[str, str]]:
    """Return (query, intent) few-shot pairs from the production router prompt.

    Returns:
        Examples in prompt order.

    Example:
        >>> ("จ่ายค่ากาแฟ 80 บาท", "expense") in intent_examples()
        True
    """
    return list(_EXAMPLE_PATTERN.findall(ORCHESTRATOR_EXAMPLES_SECTION))


def describe_intents(intents: list[str]) -> str:
    """Format the descriptions of the given intents as a Thai bullet list.

    Args:
        intents: Intent keys to include, in order.

    Returns:
        Lines like ``- "tax": คำถามเกี่ยวกับภาษี ...``.

    Example:
        >>> describe_intents(["tax"]).startswith('- "tax":')
        True
    """
    descriptions = intent_descriptions()
    return "\n".join(f'- "{intent}": {descriptions[intent]}' for intent in intents)


def examples_for(intents: list[str]) -> str:
    """Format the router's few-shot examples whose intent is in the list.

    Args:
        intents: Intent keys to keep.

    Returns:
        Lines like ``- "จ่ายค่ากาแฟ 80 บาท" → expense``; empty if none match.

    Example:
        >>> "expense" in examples_for(["expense"])
        True
    """
    return "\n".join(
        f'- "{query}" → {intent}' for query, intent in intent_examples() if intent in intents
    )
