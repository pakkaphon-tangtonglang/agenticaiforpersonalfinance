"""Turn a vague fund query into words likely to appear in Thai fund names (LLM).

Used only when a direct SEC search finds nothing, e.g. "กองทุนหุ้นอเมริกา"
→ ["US", "S&P 500", "ยูเอส", "NASDAQ"]. The model only suggests search terms;
every result is still looked up in SEC data, so it cannot invent a fund.
"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

MAX_TERMS = 5
MIN_TERM_LENGTH = 2
MAX_TERM_LENGTH = 40

EXPANSION_PROMPT = (
    "ผู้ใช้กำลังค้นหากองทุนรวมไทยด้วยคำว่า: {query}\n"
    "เสนอคำค้นสั้นๆ ไม่เกิน {limit} คำ ที่น่าจะปรากฏในชื่อกองทุนรวมไทยหรือรหัสกองทุน"
    " (ภาษาไทย อังกฤษ หรือตัวย่อ) เช่น 'หุ้นอเมริกา' → US, S&P 500, ยูเอส, NASDAQ;"
    " 'ลดหย่อนภาษี' → SSF, RMF, เพื่อการเลี้ยงชีพ; 'ทองคำ' → GOLD, ทองคำ\n"
    "ห้ามเดารหัสกองทุนเฉพาะเจาะจงที่ไม่แน่ใจ ให้ใช้คำทั่วไปที่อยู่ในชื่อกองทุน\n"
    "ถ้าคำค้นไม่มีความหมาย พิมพ์มั่ว หรือไม่ได้พูดถึงการลงทุนเลย ให้ตอบ meaningful=false"
    " และ terms เป็นลิสต์ว่าง"
)


class FundSearchTerms(BaseModel):
    """Search terms suggested by the model, or meaningful=False for gibberish."""

    meaningful: bool = Field(
        default=True, description="False when the query has no understandable investment meaning"
    )
    terms: list[str] = Field(default_factory=list, description="Short words found in fund names")


def expand_fund_query(query: str, model_factory: Callable[[], Any] | None = None) -> list[str]:
    """Ask the LLM for fund-name search terms; [] when the model is unavailable.

    Args:
        query: What the user typed (e.g. "กองทุนหุ้นอเมริกา").
        model_factory: Builds the chat model (defaults to the app's model).

    Returns:
        Up to MAX_TERMS cleaned, de-duplicated terms (never the original query).

    Example:
        >>> expand_fund_query("กองทุนหุ้นอเมริกา")  # doctest: +SKIP
        ['US', 'S&P 500', 'ยูเอส']
    """
    try:
        model = (model_factory or _default_model)()
        structured = model.with_structured_output(FundSearchTerms)
        result = structured.invoke(EXPANSION_PROMPT.format(query=query, limit=MAX_TERMS))
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        # Broad by design: a failed expansion only means "no extra results".
        logger.warning("Fund query expansion failed for %r: %s", query, exc)
        return []
    if not isinstance(result, FundSearchTerms) or not result.meaningful:
        return []  # gibberish: ask the user instead of guessing related funds
    return clean_terms(result.terms, query)


def clean_terms(terms: list[str], query: str) -> list[str]:
    """Trim, drop too short/long or repeated terms and the original query.

    Example:
        >>> clean_terms([" US ", "us", "x", "หุ้นอเมริกา"], "หุ้นอเมริกา")
        ['US']
    """
    seen = {query.strip().casefold()}
    cleaned: list[str] = []
    for term in terms:
        text = " ".join(str(term).split())
        folded = text.casefold()
        if not MIN_TERM_LENGTH <= len(text) <= MAX_TERM_LENGTH or folded in seen:
            continue
        seen.add(folded)
        cleaned.append(text)
    return cleaned[:MAX_TERMS]


def _default_model() -> Any:
    """The app's configured chat model (imported lazily: heavy dependency)."""
    from finance_ai.agents.llm_factory import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        create_chat_model,
    )

    return create_chat_model()
