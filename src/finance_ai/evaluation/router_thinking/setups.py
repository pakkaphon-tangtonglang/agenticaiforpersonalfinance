"""The four router setups: two Gemini models × two thinking levels."""

from typing import Any, Literal

from langchain_core.runnables import Runnable
from pydantic import BaseModel

from finance_ai.agents.llm_factory import create_google_chat_model
from finance_ai.core.config import get_settings

SetupName = Literal["flash_minimal", "flash_default", "lite_minimal", "lite_high"]
ThinkingLevel = Literal["minimal", "low", "medium", "high", "default"]


class RouterSetup(BaseModel):
    """One model + thinking level combination ("default" = send no thinking setting).

    The thinking probe (thinking_probe.py) showed that "low" spends 0 thinking
    tokens on routing, like "minimal", so the study compares minimal with the
    model default (how the router ran before the latency fix) and with "high".

    Example:
        >>> SETUPS["flash_minimal"].model
        'gemini-3.5-flash'
    """

    name: SetupName
    model: str
    thinking_level: ThinkingLevel


SETUPS: dict[str, RouterSetup] = {
    "flash_minimal": RouterSetup(
        name="flash_minimal", model="gemini-3.5-flash", thinking_level="minimal"
    ),
    "flash_default": RouterSetup(
        name="flash_default", model="gemini-3.5-flash", thinking_level="default"
    ),
    "lite_minimal": RouterSetup(
        name="lite_minimal", model="gemini-3.5-flash-lite", thinking_level="minimal"
    ),
    "lite_high": RouterSetup(
        name="lite_high", model="gemini-3.5-flash-lite", thinking_level="high"
    ),
}
PRODUCTION_SETUP = "flash_minimal"
COMPARISONS: list[tuple[str, str, str]] = [
    ("flash_minimal", "flash_default", "ch.3 claim: minimal vs no setting (before the fix)"),
    ("lite_minimal", "lite_high", "Does thinking help flash-lite?"),
    ("lite_minimal", "flash_minimal", "Could flash-lite replace the production router?"),
]


def build_router_model(setup: RouterSetup) -> Runnable[Any, Any]:
    """Create the Gemini model for a setup with its thinking level bound.

    The result is a binding, not a ChatGoogleGenerativeAI instance, so the
    production ``router_chat_model`` passes it through unchanged instead of
    re-binding the configured router level.

    Args:
        setup: Model and thinking level.

    Returns:
        Runnable that takes router messages and returns an AIMessage.
    """
    settings = get_settings().model_copy(
        update={"google_model": setup.model, "llm_thinking_level": None}
    )
    model = create_google_chat_model(settings)
    if setup.thinking_level == "default":
        return model.bind()
    return model.bind(thinking_level=setup.thinking_level)
