"""Multi-intent routing: one message about several topics gets one complete reply.

The router may split a message into tasks (one per agent, in the order the
user asked). There is no fixed cap; the bound is the number of agents,
because tasks for the same agent are merged. Each task runs its specialist
agent in turn, later agents see the earlier answers, and the answers are
joined as headed sections, followed by a note on unsure parts and a
summary of everything that was saved.
"""

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal

from finance_ai.agents.citation_sources import CitationSource
from finance_ai.agents.schemas import RoutedTask
from finance_ai.agents.tool_trace import ToolTrace
from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

# Same bar as single-topic routing (router_agent.ROUTER_CONFIDENCE_THRESHOLD).
TASK_CONFIDENCE_THRESHOLD = Decimal("0.7")
GENERAL_INTENT = "general"
EARLIER_ANSWER_MAX_CHARS = 1500

SECTION_TITLES: dict[str, str] = {
    "tax": "ภาษี",
    "expense": "รายรับ–รายจ่าย",
    "asset_monitoring": "การลงทุน",
    "planning": "วางแผนการเงิน",
    "recommendation": "คำแนะนำการเงิน",
    "report": "รายงานการเงิน",
    GENERAL_INTENT: "ความรู้การเงินทั่วไป",
}

WRITE_TOOL_LABELS: dict[str, str] = {
    "add_expense": "รายจ่าย",
    "add_income": "รายรับ",
    "add_portfolio_holding": "สินทรัพย์ในพอร์ต",
    "add_recurring_entry": "รายการประจำเดือน",
    "cancel_recurring_entry_tool": "ยกเลิกรายการประจำเดือน",
    "create_financial_goal": "เป้าหมายใหม่",
    "update_goal_progress": "ความคืบหน้าเป้าหมาย",
    "update_savings_goal_cross": "ความคืบหน้าเป้าหมาย",
    "manage_watchlist": "รายการติดตาม",
    "pay_debt": "จ่ายหนี้",
    "record_debt_tool": "หนี้",
    "update_debt_balance": "ยอดหนี้คงเหลือ",
    "record_social_security_cross": "ประกันสังคม",
    "record_tax_deduction_cross": "ค่าลดหย่อนภาษี",
    "undo_tax_deduction_cross": "ยกเลิกค่าลดหย่อนภาษี",
}
_AMOUNT_KEYS = (
    "amount",
    "added_amount",
    "target_amount",
    "current_amount",
    "remaining_balance",
    "monthly_salary",
)
_NAME_KEYS = ("symbol", "name", "creditor", "deduction_type", "search_term")
_WATCHLIST_WRITE_ACTIONS = frozenset({"add", "remove"})

FAILED_SECTION_TEXT = "ส่วนนี้ตอบไม่สำเร็จ ลองถามเรื่องนี้อีกครั้งได้ไหมคะ"
UNSURE_HEADER = "ส่วนที่ยังไม่แน่ใจว่าต้องการอะไร ช่วยพิมพ์รายละเอียดเพิ่มได้ไหมคะ:"
WRITE_SUMMARY_HEADER = "บันทึกแล้ว:"

_CITATION_MARKER = re.compile(r"\[(\d{1,3})\]")
_UNFINISHED_MARKER = re.compile(r"\[\d{0,3}$")

AgentRunner = Callable[[str, str], tuple[str, list[ToolTrace]]]


@dataclass(frozen=True)
class TaskSection:
    """The answer to one part of a multi-topic message.

    Attributes:
        intent: Agent that answered.
        text: Answer text.
        tools: Tool calls the agent made.
        failed: True when the agent raised instead of answering.

    Example:
        >>> TaskSection(intent="tax", text="ภาษี 5,000 บาท").failed
        False
    """

    intent: str
    text: str
    tools: tuple[ToolTrace, ...] = ()
    failed: bool = False


def merge_tasks_by_agent(tasks: Sequence[RoutedTask]) -> list[RoutedTask]:
    """Merge tasks that go to the same agent, keeping the user's order.

    ``unknown`` parts go to general chat. Merged parts join their
    sub-questions and keep the higher confidence.

    Args:
        tasks: Tasks from the router.

    Returns:
        At most one task per agent.

    Example:
        >>> [t.intent for t in merge_tasks_by_agent(tasks)]  # doctest: +SKIP
        ['tax', 'planning']
    """
    merged: dict[str, RoutedTask] = {}
    for task in tasks:
        intent = GENERAL_INTENT if task.intent == "unknown" else task.intent
        earlier = merged.get(intent)
        if earlier is None:
            merged[intent] = task.model_copy(update={"intent": intent})
            continue
        merged[intent] = earlier.model_copy(
            update={
                "sub_query": f"{earlier.sub_query}\n{task.sub_query}".strip(),
                "confidence": max(earlier.confidence, task.confidence),
            }
        )
    return list(merged.values())


def split_by_confidence(tasks: Sequence[RoutedTask]) -> tuple[list[RoutedTask], list[RoutedTask]]:
    """Separate parts the router is sure about from parts it is not.

    Args:
        tasks: Merged tasks.

    Returns:
        (confident tasks, unsure tasks), each in the user's order.

    Example:
        >>> split_by_confidence([])
        ([], [])
    """
    confident = [task for task in tasks if task.confidence >= TASK_CONFIDENCE_THRESHOLD]
    unsure = [task for task in tasks if task.confidence < TASK_CONFIDENCE_THRESHOLD]
    return confident, unsure


def section_title(intent: str) -> str:
    """Return the Thai heading for an agent's section.

    Args:
        intent: Agent key.

    Returns:
        Heading text; general chat's title for unknown keys.

    Example:
        >>> section_title("tax")
        'ภาษี'
    """
    return SECTION_TITLES.get(intent, SECTION_TITLES[GENERAL_INTENT])


def build_task_query(task: RoutedTask, original_query: str, earlier: Sequence[TaskSection]) -> str:
    """Build the question one agent receives.

    The router's self-contained sub-question is used (the full message when
    it is missing). Earlier answers from the same message are attached as
    context, so a later part can build on them.

    Args:
        task: The part to answer.
        original_query: The user's whole message.
        earlier: Sections already answered in this message.

    Returns:
        Thai question for the agent.

    Example:
        >>> build_task_query(task, "ภาษีเท่าไร แล้วควรออมเท่าไร", [])  # doctest: +SKIP
        'ภาษีเท่าไร'
    """
    question = task.sub_query.strip() or original_query
    answered = [section for section in earlier if not section.failed]
    if not answered:
        return question
    context = "\n\n".join(
        f"[{section_title(section.intent)}]\n{section.text[:EARLIER_ANSWER_MAX_CHARS]}"
        for section in answered
    )
    return (
        f"{question}\n\n"
        "(คำตอบส่วนก่อนหน้าในข้อความเดียวกัน ใช้ประกอบได้ถ้าจำเป็น"
        f" ไม่ต้องตอบซ้ำ:\n{context})"
    )


def run_tasks(
    tasks: Sequence[RoutedTask], original_query: str, run_agent: AgentRunner
) -> list[TaskSection]:
    """Run each task's agent in the user's order; one failure does not stop the rest.

    Tasks run one after another because agents may write data and a later
    part may depend on an earlier one.

    Args:
        tasks: Confident, merged tasks.
        original_query: The user's whole message.
        run_agent: Runs (intent, question) and returns (answer, tool calls).

    Returns:
        One section per task, in order.

    Example:
        >>> run_tasks(tasks, query, runner)  # doctest: +SKIP
    """
    sections: list[TaskSection] = []
    for task in tasks:
        question = build_task_query(task, original_query, sections)
        try:
            text, tools = run_agent(task.intent, question)
        except Exception:  # pylint: disable=broad-exception-caught  # noqa: BLE001
            logger.error("Multi-intent part failed: %s", task.intent, exc_info=True)
            sections.append(failed_section(task.intent))
            continue
        if not text.strip():
            logger.warning("Multi-intent part returned no text: %s", task.intent)
            sections.append(failed_section(task.intent, tools))
            continue
        sections.append(TaskSection(intent=task.intent, text=text, tools=tuple(tools)))
    return sections


def failed_section(intent: str, tools: Sequence[ToolTrace] = ()) -> TaskSection:
    """Return the apology section for a part that raised or gave no text.

    Args:
        intent: Agent that failed.
        tools: Tool calls it made before failing (writes still count as saved).

    Returns:
        A failed section with the short apology.

    Example:
        >>> failed_section("tax").failed
        True
    """
    return TaskSection(intent=intent, text=FAILED_SECTION_TEXT, tools=tuple(tools), failed=True)


def section_heading(intent: str, is_first: bool) -> str:
    """Return the markdown heading that opens a section.

    Args:
        intent: Agent key.
        is_first: True for the first section (no blank line before it).

    Returns:
        Heading line ending with a newline.

    Example:
        >>> section_heading("tax", is_first=True)
        '## ภาษี\\n'
    """
    prefix = "" if is_first else "\n\n"
    # Level 2, above the level-3 headings agents use inside their own answers.
    return f"{prefix}## {section_title(intent)}\n"


def combine_sections(sections: Sequence[TaskSection], unsure: Sequence[RoutedTask]) -> str:
    """Join the sections into one reply.

    A lone section gets no heading. Unsure parts and saved items are listed
    at the end.

    Args:
        sections: Answered parts, in order.
        unsure: Parts the router was not sure about.

    Returns:
        The full reply text.

    Example:
        >>> combine_sections([TaskSection("tax", "ภาษี 0 บาท")], [])
        'ภาษี 0 บาท'
    """
    if len(sections) == 1:
        body = sections[0].text
    else:
        body = "".join(
            section_heading(section.intent, index == 0) + section.text
            for index, section in enumerate(sections)
        )
    return body + closing_notes(sections, unsure)


def closing_notes(sections: Sequence[TaskSection], unsure: Sequence[RoutedTask]) -> str:
    """Return the unsure-parts note and the saved-items summary.

    Args:
        sections: Answered parts.
        unsure: Parts the router was not sure about.

    Returns:
        Text starting with a blank line, or "" when there is nothing to add.

    Example:
        >>> closing_notes([], [])
        ''
    """
    notes: list[str] = []
    if unsure:
        notes.append(UNSURE_HEADER + "".join(f"\n- {task.sub_query}" for task in unsure))
    summary = write_summary(tool for section in sections for tool in section.tools)
    if summary:
        notes.append(summary)
    return "".join(f"\n\n{note}" for note in notes)


def write_summary(tools: Iterable[ToolTrace]) -> str:
    """List every successful write made while answering.

    Args:
        tools: Tool calls from all sections.

    Returns:
        "บันทึกแล้ว:" followed by one line per write, or "" when nothing was saved.

    Example:
        >>> write_summary([ToolTrace(name="add_expense", args={"amount": "120"})])
        'บันทึกแล้ว:\\n- รายจ่าย 120 บาท'
    """
    lines = [line for line in (describe_write(tool) for tool in tools) if line]
    if not lines:
        return ""
    return WRITE_SUMMARY_HEADER + "".join(f"\n- {line}" for line in lines)


def describe_write(tool: ToolTrace) -> str | None:
    """Describe one write tool call for the summary.

    Args:
        tool: A finished tool call.

    Returns:
        Short Thai line, or None for reads, failures and watchlist listing.

    Example:
        >>> describe_write(ToolTrace(name="add_expense", args={"amount": "80"}))
        'รายจ่าย 80 บาท'
    """
    label = WRITE_TOOL_LABELS.get(tool.name)
    if label is None or tool.failed:
        return None
    if tool.name == "manage_watchlist" and tool.args.get("action") not in _WATCHLIST_WRITE_ACTIONS:
        return None
    details = [tool.args[key] for key in _NAME_KEYS if tool.args.get(key)][:1]
    details += [f"{tool.args[key]} บาท" for key in _AMOUNT_KEYS if tool.args.get(key)][:1]
    return " ".join([label, *details])


def merge_citations(
    turn_citations: Sequence[CitationSource], section_citations: Sequence[CitationSource]
) -> tuple[list[CitationSource], dict[int, int]]:
    """Give one section's citations turn-wide numbers.

    Each agent numbers its sources from [1], so two sections can both cite
    [1] for different documents. A document already cited keeps its number.

    Args:
        turn_citations: Citations already numbered in this reply.
        section_citations: One section's citations with its own numbers.

    Returns:
        (all citations so far, section number → turn-wide number).

    Example:
        >>> merge_citations([], [])
        ([], {})
    """
    merged = list(turn_citations)
    number_by_key = {citation.document_key: citation.index for citation in merged}
    mapping: dict[int, int] = {}
    for citation in section_citations:
        if citation.document_key not in number_by_key:
            number_by_key[citation.document_key] = max(number_by_key.values(), default=0) + 1
            merged.append(
                citation.model_copy(update={"index": number_by_key[citation.document_key]})
            )
        mapping[citation.index] = number_by_key[citation.document_key]
    return merged, mapping


class CitationRenumberer:
    """Rewrite a section's [n] markers to turn-wide numbers while text streams in.

    A marker may arrive split across tokens ("[", "2", "]"), so an
    unfinished marker at the end of a token is held back until it completes.

    Example:
        >>> renumberer = CitationRenumberer({1: 3})
        >>> renumberer.feed("ลดหย่อน [") + renumberer.feed("1] บาท") + renumberer.flush()
        'ลดหย่อน [3] บาท'
    """

    def __init__(self, mapping: dict[int, int] | None = None) -> None:
        """Start with a section-number → turn-number mapping (identity when empty)."""
        self.mapping: dict[int, int] = dict(mapping or {})
        self._held = ""

    def feed(self, text: str) -> str:
        """Return the text that is safe to send, with markers rewritten.

        Args:
            text: Next streamed piece.

        Returns:
            Rewritten text, minus any unfinished marker at the end.
        """
        buffer = self._held + text
        unfinished = _UNFINISHED_MARKER.search(buffer)
        cut = unfinished.start() if unfinished else len(buffer)
        self._held = buffer[cut:]
        return self.rewrite(buffer[:cut])

    def flush(self) -> str:
        """Return whatever is still held back (the stream ended)."""
        held, self._held = self._held, ""
        return self.rewrite(held)

    def rewrite(self, text: str) -> str:
        """Rewrite every complete marker in a finished text.

        Args:
            text: Text with section-local markers.

        Returns:
            Text with turn-wide markers.
        """
        return _CITATION_MARKER.sub(
            lambda match: f"[{self.mapping.get(int(match.group(1)), int(match.group(1)))}]", text
        )
