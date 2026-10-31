"""User-facing Thai messages for LLM provider failures.

Agents call the LLM provider for every reply. When that call fails
(quota exhausted, provider outage), the user should get a clear message
instead of a broken stream or an HTTP 500.
"""

from finance_ai.rag.rate_limit_retry import is_rate_limit_error

LLM_QUOTA_MESSAGE: str = (
    "ขออภัยครับ โควตาการใช้งาน AI เต็มชั่วคราว " "กรุณาลองใหม่อีกครั้งในภายหลัง"
)

LLM_GENERIC_ERROR_MESSAGE: str = (
    "ขออภัยครับ ระบบเกิดข้อผิดพลาด "
    "กรุณาลองถามใหม่อีกครั้ง\n\n"
    "ลองถามในรูปแบบอื่น เช่น:\n"
    '- "ช่วยวางแผนออมเงินซื้อรถ'
    'ราคา 800,000 บาท"\n'
    '- "คำนวณภาษีเงินได้ 500,000 บาท"'
)


def describe_llm_error(error: Exception) -> str:
    """Pick the Thai message that explains an LLM failure to the user.

    Args:
        error: The exception raised by the LLM call or agent graph.

    Returns:
        The quota message for rate-limit errors, else the generic message.

    Example:
        >>> describe_llm_error(RuntimeError("429 RESOURCE_EXHAUSTED")) == LLM_QUOTA_MESSAGE
        True
    """
    if is_rate_limit_error(error):
        return LLM_QUOTA_MESSAGE
    return LLM_GENERIC_ERROR_MESSAGE
