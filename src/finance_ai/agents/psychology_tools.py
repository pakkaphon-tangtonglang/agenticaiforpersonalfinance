"""LangGraph tool for psychological cue detection.

Provides a @tool-decorated function that agents can use to detect
behavioral biases and sentiment in user messages.
"""

from typing import Any

from langchain_core.tools import tool


@tool
def detect_psychological_cues(user_input: str) -> dict[str, Any]:
    """ตรวจจับ sentiment และ behavioral bias ของผู้ใช้.

    ใช้เมื่อผู้ใช้แสดงอารมณ์ ความกังวล หรือพฤติกรรมทางการเงินที่อาจมี bias
    เช่น กลัวขาดทุน (loss aversion), ใจร้อน (present bias),
    มั่นใจเกินไป (overconfidence), ตามกระแส (trend chasing)

    Args:
        user_input: ข้อความของผู้ใช้ที่ต้องการวิเคราะห์.

    Returns:
        Dict with sentiment, detected_biases, tone_recommendation, summary.
    """
    from finance_ai.tools.psychology_service import (  # noqa: PLC0415
        analyze_psychological_cues,
    )

    result = analyze_psychological_cues(user_input)
    return result.model_dump(mode="json")
