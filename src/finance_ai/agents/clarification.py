"""Tool results that tell the agent to ask the user instead of guessing.

A tool returns `clarification(...)` when the input is unknown, ambiguous or
missing. The agent prompt (CLARIFICATION_RULES in prompts.py) makes the LLM
ask `ask_user`, offering `options` when there are any.
"""

from typing import Any

from finance_ai.tools.symbol_guard import SymbolResolutionError, SymbolSearchUnavailableError

ASK_FOR_FULL_NAME = "ขอชื่อเต็มหรือตัวย่อของสินทรัพย์อีกครั้ง เช่น PTT, AAPL, ทองคำ"
ASK_WHICH_ONE = "หมายถึงตัวไหน"


def clarification(problem: str, ask_user: str, options: list[str] | None = None) -> dict[str, Any]:
    """Build a result that tells the agent to ask the user one short question.

    Args:
        problem: What is wrong, for the agent (and logs).
        ask_user: The question to ask, in Thai.
        options: Choices to offer the user, if any.

    Returns:
        {"error", "needs_clarification": True, "ask_user", "options"?}.

    Example:
        >>> clarification("time missing", "แจ้งกี่โมง")["needs_clarification"]
        True
    """
    result: dict[str, Any] = {"error": problem, "needs_clarification": True, "ask_user": ask_user}
    if options:
        result["options"] = options
    return result


def symbol_problem(error: SymbolResolutionError) -> dict[str, Any]:
    """Turn a symbol lookup failure into a retry note or a question.

    Search outages are not the user's fault, so they are not turned into a
    question; ambiguous input offers the candidates; unknown input asks again.

    Example:
        >>> symbol_problem(SymbolResolutionError("not found"))["ask_user"]
        'ขอชื่อเต็มหรือตัวย่อของสินทรัพย์อีกครั้ง เช่น PTT, AAPL, ทองคำ'
    """
    if isinstance(error, SymbolSearchUnavailableError):
        return {"error": str(error), "retry_later": True}
    if error.candidates:
        options = [f"{match.symbol} — {match.name}" for match in error.candidates]
        return clarification(str(error), ASK_WHICH_ONE, options)
    return clarification(str(error), ASK_FOR_FULL_NAME)
