"""Post-generation guardrail for the Recommendation Agent.

Deterministic checks applied to the agent's final answer before it is
shown to the user. Deterministic code cannot hallucinate, so it acts as
the safety floor on top of the LLM:

1. Risk mismatch — a low-risk user (per the SEC suitability
   questionnaire) reading about high-risk instruments gets a warning.
2. Unbacked numbers — specific return percentages promised without any
   tool call in that turn get an "estimates only" disclaimer.
"""

from finance_ai.tools.recommendation_constants import (
    HIGH_RISK_INSTRUMENT_KEYWORDS,
    LOW_RISK_LEVEL_THRESHOLD,
    RETURN_PROMISE_PATTERN,
)

RISK_MISMATCH_WARNING: str = (
    "⚠️ ข้อควรระวัง: คำแนะนำบางส่วนอาจไม่เหมาะกับระดับความเสี่ยงที่คุณรับได้ "
    "(ระดับ {risk_level}) กรุณาพิจารณาอย่างรอบคอบและศึกษาข้อมูลก่อนตัดสินใจลงทุน"
)
UNBACKED_NUMBERS_WARNING: str = (
    "⚠️ หมายเหตุ: ตัวเลขผลตอบแทนในคำแนะนำเป็นการประมาณการ ไม่ใช่ผลตอบแทนที่รับประกัน "
    "กรุณาตรวจสอบราคาและข้อมูลล่าสุดก่อนตัดสินใจ"
)


def _is_low_risk_level(risk_level: int | None) -> bool:
    """Check whether the user's risk level counts as low risk.

    Args:
        risk_level: Risk level 1-5 from the questionnaire, or None.

    Returns:
        True only for known levels at or below the low-risk threshold.

    Example:
        >>> _is_low_risk_level(1)
        True
    """
    return risk_level is not None and risk_level <= LOW_RISK_LEVEL_THRESHOLD


def _mentions_high_risk_instruments(answer: str) -> bool:
    """Check whether the answer mentions any high-risk instrument.

    Args:
        answer: The agent's final answer text.

    Returns:
        True when any high-risk keyword appears (case-insensitive).

    Example:
        >>> _mentions_high_risk_instruments("แนะนำคริปโต")
        True
    """
    lowered = answer.lower()
    return any(keyword in lowered for keyword in HIGH_RISK_INSTRUMENT_KEYWORDS)


def _promises_returns(answer: str) -> bool:
    """Check whether the answer promises specific return percentages.

    Args:
        answer: The agent's final answer text.

    Returns:
        True when a return/profit percentage claim is found.

    Example:
        >>> _promises_returns("ผลตอบแทน 15% ต่อปี")
        True
    """
    return RETURN_PROMISE_PATTERN.search(answer) is not None


def check_recommendation_response(
    answer: str,
    risk_level: int | None,
    had_tool_results: bool,
) -> str:
    """Run all guardrails on the agent's final answer.

    Args:
        answer: The agent's final answer text.
        risk_level: User's latest questionnaire risk level (1-5), or None.
        had_tool_results: Whether tool results exist in the conversation.

    Returns:
        Warning text to append ("" when the answer is clean).

    Example:
        >>> check_recommendation_response("แนะนำคริปโต", 1, True)
        '⚠️ ข้อควรระวัง: ...'
    """
    warnings: list[str] = []
    if _is_low_risk_level(risk_level) and _mentions_high_risk_instruments(answer):
        warnings.append(RISK_MISMATCH_WARNING.format(risk_level=risk_level))
    if _promises_returns(answer) and not had_tool_results:
        warnings.append(UNBACKED_NUMBERS_WARNING)
    return "\n\n".join(warnings)
