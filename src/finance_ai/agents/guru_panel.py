"""Guru panel: ask several gurus one question, then compare their answers.

Gurus answer in parallel (bounded to respect LLM rate limits); one failing
guru only produces an error event. A single structured LLM call then builds
the stance comparison, instead of multi-round debates.
"""

from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from finance_ai.agents.graph_cache import get_compiled_graph
from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

PANEL_MAX_WORKERS = 2
GURU_ERROR_MESSAGE = "กูรูท่านนี้ตอบไม่สำเร็จ กรุณาลองใหม่อีกครั้ง"
PANEL_ANSWER_INSTRUCTION = (
    "\n\n(โหมดถามทั้งวง: ตอบสั้นไม่เกิน 150 คำ"
    " บรรทัดแรกขึ้นต้นด้วย 'จุดยืน:' ตามด้วยเหตุผลหลักและความเสี่ยงที่ควรระวัง)"
)

PanelEventType = Literal["guru_answer", "guru_error", "synthesis", "synthesis_error"]


class PanelEvent(BaseModel):
    """One event streamed to the panel view."""

    event_type: PanelEventType
    guru_id: str | None = None
    content: str = ""
    synthesis: "PanelSynthesis | None" = None


class GuruStance(BaseModel):
    """One guru's position, extracted for the comparison table."""

    guru_id: str = Field(description="Persona id, e.g. buffett")
    stance: str = Field(description="จุดยืนสั้นๆ ภาษาไทย")
    key_reason: str = Field(description="เหตุผลหลัก ภาษาไทย")
    risk: str = Field(description="ความเสี่ยงที่กูรูเตือน ภาษาไทย")


class PanelSynthesis(BaseModel):
    """Comparison of the gurus' answers."""

    stances: list[GuruStance]
    consensus: str = Field(description="จุดที่กูรูเห็นตรงกัน ภาษาไทย")
    differences: str = Field(description="จุดที่กูรูเห็นต่างกัน ภาษาไทย")


def synthesis_to_markdown(synthesis: PanelSynthesis) -> str:
    """Render the comparison as a Thai markdown table for chat history.

    Args:
        synthesis: The structured panel comparison.

    Returns:
        Markdown with one table row per guru, then consensus and differences.

    Example:
        >>> synthesis_to_markdown(synthesis).splitlines()[0]
        '| กูรู | จุดยืน | เหตุผลหลัก | ความเสี่ยง |'
    """
    rows = [f"| {s.guru_id} | {s.stance} | {s.key_reason} | {s.risk} |" for s in synthesis.stances]
    return "\n".join(
        ["| กูรู | จุดยืน | เหตุผลหลัก | ความเสี่ยง |", "|---|---|---|---|", *rows]
        + ["", f"**เห็นตรงกัน:** {synthesis.consensus}", f"**เห็นต่าง:** {synthesis.differences}"]
    )


def _final_answer_text(final_state: dict[str, Any]) -> str:
    """Return the text of the last AI message in a graph result."""
    for message in reversed(final_state.get("messages", [])):
        if isinstance(message, AIMessage) and message.content:
            return str(message.text)
    return ""


def answer_as_guru(
    query: str,
    guru_id: str,
    user_id: str,
    db_session_factory: Any,
    chat_model: BaseChatModel | None,
) -> str:
    """Get one guru's short panel answer (no chat history).

    Args:
        query: The user's question.
        guru_id: Persona id.
        user_id: UUID of the user.
        db_session_factory: Session factory injected into tools.
        chat_model: Chat model for the guru graph.

    Returns:
        The guru's final answer text.

    Raises:
        Exception: Whatever the graph raises; run_guru_panel reports it.

    Example:
        >>> answer_as_guru("ตลาดตก", "buffett", "u1", factory, model)
    """
    graph = get_compiled_graph("guru", chat_model)
    if graph is None:
        raise RuntimeError("Guru graph is not registered in graph_cache._BUILDER_MAP.")
    state = {
        "messages": [("user", query + PANEL_ANSWER_INSTRUCTION)],
        "user_id": user_id,
        "db_session_factory": db_session_factory,
        "guru_id": guru_id,
    }
    return _final_answer_text(graph.invoke(state))


def _build_synthesis_prompt(query: str, answers: dict[str, str]) -> str:
    """Build the comparison prompt from each guru's answer."""
    sections = "\n\n".join(f"[{guru_id}]\n{answer}" for guru_id, answer in answers.items())
    return (
        "สรุปคำตอบของกูรูแต่ละคนต่อคำถามเดียวกัน เป็นภาษาไทย\n"
        "ใช้เฉพาะข้อมูลในคำตอบด้านล่าง ห้ามเพิ่มความเห็นใหม่\n"
        f"คำถาม: {query}\n\n{sections}"
    )


def synthesize_panel(
    query: str,
    answers: dict[str, str],
    chat_model: BaseChatModel,
) -> PanelSynthesis | None:
    """Compare the gurus' answers with one structured LLM call.

    Args:
        query: The user's question.
        answers: Answer text keyed by guru_id.
        chat_model: Chat model supporting with_structured_output.

    Returns:
        PanelSynthesis, or None when the model call fails.

    Example:
        >>> synthesize_panel("ตลาดตก", {"buffett": "..."}, model).consensus
    """
    try:
        structured_model = chat_model.with_structured_output(PanelSynthesis)
        result = structured_model.invoke(_build_synthesis_prompt(query, answers))
    except Exception as error:  # pylint: disable=broad-exception-caught
        # Broad by design: a failed summary must not hide the gurus' answers.
        logger.warning("Panel synthesis failed: %s", error)
        return None
    return result if isinstance(result, PanelSynthesis) else None


def _collect_answers(
    futures: dict[Future[str], str],
    answers: dict[str, str],
) -> Iterator[PanelEvent]:
    """Yield an event per guru as each future finishes, filling answers."""
    for future in as_completed(futures):
        guru_id = futures[future]
        try:
            answers[guru_id] = future.result()
        except Exception as error:  # pylint: disable=broad-exception-caught
            logger.warning("Panel guru %s failed: %s", guru_id, error)
            # Details stay in the log; the client only gets a generic message.
            yield PanelEvent(event_type="guru_error", guru_id=guru_id, content=GURU_ERROR_MESSAGE)
            continue
        yield PanelEvent(event_type="guru_answer", guru_id=guru_id, content=answers[guru_id])


def run_guru_panel(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    query: str,
    guru_ids: list[str],
    user_id: str,
    db_session_factory: Any,
    chat_model: BaseChatModel | None,
) -> Iterator[PanelEvent]:
    """Ask every guru the same question, then compare their answers.

    Args:
        query: The user's question.
        guru_ids: Personas to ask.
        user_id: UUID of the user.
        db_session_factory: Session factory injected into tools.
        chat_model: Chat model for the gurus and the synthesis.

    Yields:
        guru_answer / guru_error events as gurus finish, then one
        synthesis or synthesis_error event (skipped if nobody answered).

    Example:
        >>> for event in run_guru_panel("ตลาดตก", ["buffett", "dalio"], "u1", f, m): ...
    """
    answers: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=PANEL_MAX_WORKERS) as pool:
        futures = {
            pool.submit(
                answer_as_guru, query, guru_id, user_id, db_session_factory, chat_model
            ): guru_id
            for guru_id in guru_ids
        }
        yield from _collect_answers(futures, answers)
    if not answers:
        return
    synthesis = synthesize_panel(query, answers, chat_model) if chat_model else None
    if synthesis is None:
        yield PanelEvent(event_type="synthesis_error", content="สรุปเปรียบเทียบไม่สำเร็จ")
        return
    yield PanelEvent(event_type="synthesis", synthesis=synthesis)
