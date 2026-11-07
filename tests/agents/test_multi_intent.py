"""Tests for multi-intent helpers: merging, ordering, joining, summaries, citations."""

from decimal import Decimal

import pytest

from finance_ai.agents.citation_sources import CitationSource
from finance_ai.agents.multi_intent import (
    EARLIER_ANSWER_MAX_CHARS,
    FAILED_SECTION_TEXT,
    TASK_CONFIDENCE_THRESHOLD,
    UNSURE_HEADER,
    WRITE_SUMMARY_HEADER,
    CitationRenumberer,
    TaskSection,
    build_task_query,
    combine_sections,
    describe_write,
    merge_citations,
    merge_tasks_by_agent,
    run_tasks,
    section_heading,
    section_title,
    split_by_confidence,
    write_summary,
)
from finance_ai.agents.router_agent import ROUTER_CONFIDENCE_THRESHOLD
from finance_ai.agents.schemas import RoutedTask
from finance_ai.agents.tool_trace import ToolTrace


def task(intent: str, sub_query: str = "", confidence: str = "0.9") -> RoutedTask:
    """Build a routed task."""
    return RoutedTask(intent=intent, sub_query=sub_query, confidence=Decimal(confidence))  # type: ignore[arg-type]


def citation(index: int, key: str) -> CitationSource:
    """Build a citation for one document."""
    return CitationSource(index=index, document_key=key, title=f"doc {key}")


class TestMergeTasksByAgent:
    """Tasks are merged per agent, in the user's order, with no cap."""

    def test_keeps_every_distinct_agent(self) -> None:
        """Six different agents give six tasks (no cap)."""
        intents = ["tax", "expense", "asset_monitoring", "planning", "recommendation", "report"]
        merged = merge_tasks_by_agent([task(intent) for intent in intents])
        assert [item.intent for item in merged] == intents

    def test_same_agent_parts_are_joined(self) -> None:
        """Two tax parts become one task with both questions and the higher confidence."""
        merged = merge_tasks_by_agent(
            [task("tax", "ภาษีปีนี้", "0.8"), task("planning", "ออม"), task("tax", "RMF", "0.95")]
        )
        assert [item.intent for item in merged] == ["tax", "planning"]
        assert merged[0].sub_query == "ภาษีปีนี้\nRMF"
        assert merged[0].confidence == Decimal("0.95")

    def test_unknown_goes_to_general(self) -> None:
        """Unknown parts are answered by general chat."""
        merged = merge_tasks_by_agent([task("unknown", "สูตรอาหาร"), task("general", "ดอกเบี้ย")])
        assert [item.intent for item in merged] == ["general"]

    def test_empty(self) -> None:
        """No tasks stay no tasks."""
        assert not merge_tasks_by_agent([])


class TestSplitByConfidence:
    """Parts below the routing threshold are separated."""

    def test_threshold_matches_single_topic_router(self) -> None:
        """Multi-topic parts use the same bar as single-topic routing."""
        assert TASK_CONFIDENCE_THRESHOLD == ROUTER_CONFIDENCE_THRESHOLD

    def test_split(self) -> None:
        """0.7 is confident; 0.69 is not."""
        confident, unsure = split_by_confidence(
            [task("tax", confidence="0.7"), task("report", confidence="0.69")]
        )
        assert [item.intent for item in confident] == ["tax"]
        assert [item.intent for item in unsure] == ["report"]


class TestBuildTaskQuery:
    """Each agent gets its own question, plus earlier answers as context."""

    def test_first_part_gets_sub_query_only(self) -> None:
        """The first part has no earlier context."""
        assert build_task_query(task("tax", "ภาษีเท่าไร"), "ทั้งข้อความ", []) == "ภาษีเท่าไร"

    def test_missing_sub_query_falls_back_to_message(self) -> None:
        """A part without a sub-question gets the whole message."""
        assert build_task_query(task("tax", "  "), "ทั้งข้อความ", []) == "ทั้งข้อความ"

    def test_earlier_answers_are_attached(self) -> None:
        """A later part sees earlier answers under their headings."""
        earlier = [TaskSection(intent="tax", text="ภาษี 5,000 บาท")]
        query = build_task_query(task("planning", "ควรออมเท่าไร"), "ข้อความ", earlier)
        assert query.startswith("ควรออมเท่าไร")
        assert "[ภาษี]\nภาษี 5,000 บาท" in query

    def test_failed_sections_are_not_context(self) -> None:
        """A failed part's apology is not passed on."""
        earlier = [TaskSection(intent="tax", text=FAILED_SECTION_TEXT, failed=True)]
        assert build_task_query(task("planning", "ออม"), "ข้อความ", earlier) == "ออม"

    def test_long_answers_are_trimmed(self) -> None:
        """Earlier answers are cut so the context stays small."""
        earlier = [TaskSection(intent="report", text="ก" * (EARLIER_ANSWER_MAX_CHARS + 500))]
        query = build_task_query(task("planning", "ออม"), "ข้อความ", earlier)
        assert "ก" * EARLIER_ANSWER_MAX_CHARS in query
        assert "ก" * (EARLIER_ANSWER_MAX_CHARS + 1) not in query


class TestRunTasks:
    """Parts run one after another; one failure does not stop the rest."""

    def test_runs_in_order_with_earlier_context(self) -> None:
        """The second agent receives the first agent's answer."""
        calls: list[tuple[str, str]] = []

        def runner(intent: str, question: str) -> tuple[str, list[ToolTrace]]:
            calls.append((intent, question))
            return f"ตอบ {intent}", [ToolTrace(name=f"{intent}_tool")]

        sections = run_tasks([task("tax", "ภาษี"), task("planning", "ออม")], "ข้อความ", runner)
        assert [section.intent for section in sections] == ["tax", "planning"]
        assert calls[0] == ("tax", "ภาษี")
        assert "ตอบ tax" in calls[1][1]
        assert sections[1].tools[0].name == "planning_tool"

    def test_empty_answer_counts_as_failed(self) -> None:
        """An agent that returns no text gets the apology, not an empty section."""
        sections = run_tasks([task("tax"), task("planning")], "ข้อความ", lambda i, q: ("  ", []))
        assert sections[0].failed and sections[0].text == FAILED_SECTION_TEXT

    def test_failure_is_isolated(self) -> None:
        """A failing agent becomes a short apology; later parts still run."""

        def runner(intent: str, question: str) -> tuple[str, list[ToolTrace]]:
            if intent == "tax":
                raise RuntimeError("provider down")
            return "ตอบแผน", []

        sections = run_tasks([task("tax"), task("planning")], "ข้อความ", runner)
        assert sections[0].failed and sections[0].text == FAILED_SECTION_TEXT
        assert sections[1].text == "ตอบแผน" and not sections[1].failed


class TestCombineSections:
    """Sections are joined under headings with closing notes."""

    def test_single_section_has_no_heading(self) -> None:
        """One answered part reads like a normal reply."""
        assert combine_sections([TaskSection("tax", "ภาษี 0 บาท")], []) == "ภาษี 0 บาท"

    def test_sections_get_headings_in_order(self) -> None:
        """Each part opens with its agent's Thai heading."""
        reply = combine_sections([TaskSection("tax", "A"), TaskSection("planning", "B")], [])
        assert reply == "## ภาษี\nA\n\n## วางแผนการเงิน\nB"

    def test_unsure_parts_are_listed(self) -> None:
        """Parts below the threshold are named so the user can clarify them."""
        reply = combine_sections(
            [TaskSection("tax", "A")], [task("report", "สรุปอะไรสักอย่าง", "0.4")]
        )
        assert reply == f"A\n\n{UNSURE_HEADER}\n- สรุปอะไรสักอย่าง"

    def test_saved_items_are_summarised(self) -> None:
        """Writes from every section are listed at the end."""
        sections = [
            TaskSection(
                "expense", "A", tools=(ToolTrace(name="add_expense", args={"amount": "120"}),)
            ),
            TaskSection(
                "planning",
                "B",
                tools=(
                    ToolTrace(
                        name="update_savings_goal_cross",
                        args={"search_term": "ฉุกเฉิน", "added_amount": "40000"},
                    ),
                ),
            ),
        ]
        reply = combine_sections(sections, [])
        assert reply.endswith(
            f"{WRITE_SUMMARY_HEADER}\n- รายจ่าย 120 บาท\n- ความคืบหน้าเป้าหมาย ฉุกเฉิน 40000 บาท"
        )


class TestDescribeWrite:
    """Only successful writes appear in the summary."""

    @pytest.mark.parametrize(
        ("trace", "expected"),
        [
            (ToolTrace(name="add_expense", args={"amount": "80"}), "รายจ่าย 80 บาท"),
            (
                ToolTrace(
                    name="add_portfolio_holding", args={"symbol": "PTT.BK", "quantity": "100"}
                ),
                "สินทรัพย์ในพอร์ต PTT.BK",
            ),
            (
                ToolTrace(
                    name="create_financial_goal",
                    args={"name": "ดาวน์บ้าน", "target_amount": "300000"},
                ),
                "เป้าหมายใหม่ ดาวน์บ้าน 300000 บาท",
            ),
            (
                ToolTrace(name="manage_watchlist", args={"action": "add", "symbol": "BDMS.BK"}),
                "รายการติดตาม BDMS.BK",
            ),
        ],
    )
    def test_writes(self, trace: ToolTrace, expected: str) -> None:
        """Write tools get a label plus their name and amount."""
        assert describe_write(trace) == expected

    @pytest.mark.parametrize(
        "trace",
        [
            ToolTrace(name="calculate_thai_tax", args={"annual_income": "600000"}),
            ToolTrace(name="add_expense", args={"amount": "80"}, failed=True),
            ToolTrace(name="manage_watchlist", args={"action": "list"}),
        ],
    )
    def test_reads_failures_and_listing_are_skipped(self, trace: ToolTrace) -> None:
        """Reads, failed writes and watchlist listing are not 'saved'."""
        assert describe_write(trace) is None

    def test_nothing_saved_gives_empty_summary(self) -> None:
        """No writes, no summary."""
        assert write_summary([ToolTrace(name="get_stock_price")]) == ""


class TestHeadings:
    """Section headings."""

    def test_unknown_key_uses_general_title(self) -> None:
        """An unexpected key still gets a heading."""
        assert section_title("mystery") == section_title("general")

    def test_later_heading_starts_with_blank_line(self) -> None:
        """Headings after the first are separated from the previous section."""
        assert section_heading("report", is_first=False) == "\n\n## รายงานการเงิน\n"


class TestMergeCitations:
    """Citation numbers are unique across the whole reply."""

    def test_first_section_keeps_numbers(self) -> None:
        """With nothing cited yet, numbers stay as they are."""
        merged, mapping = merge_citations([], [citation(1, "a"), citation(2, "b")])
        assert [item.index for item in merged] == [1, 2]
        assert mapping == {1: 1, 2: 2}

    def test_new_documents_continue_numbering(self) -> None:
        """A second section's [1] for a new document becomes [3]."""
        turn = [citation(1, "a"), citation(2, "b")]
        merged, mapping = merge_citations(turn, [citation(1, "c")])
        assert mapping == {1: 3}
        assert merged[-1].index == 3 and merged[-1].document_key == "c"

    def test_same_document_keeps_its_number(self) -> None:
        """A document cited again keeps its first number."""
        merged, mapping = merge_citations([citation(1, "a")], [citation(1, "b"), citation(2, "a")])
        assert mapping == {1: 2, 2: 1}
        assert [item.document_key for item in merged] == ["a", "b"]


class TestCitationRenumberer:
    """Markers are rewritten even when split across streamed tokens."""

    def test_split_marker(self) -> None:
        """'[' and '1]' in separate tokens still become [3]."""
        renumberer = CitationRenumberer({1: 3})
        out = renumberer.feed("ลดหย่อน [") + renumberer.feed("1") + renumberer.feed("] บาท")
        assert out + renumberer.flush() == "ลดหย่อน [3] บาท"

    def test_unknown_numbers_are_unchanged(self) -> None:
        """Numbers outside the mapping are left as written."""
        assert CitationRenumberer({1: 2}).rewrite("[1][5]") == "[2][5]"

    def test_held_text_is_flushed(self) -> None:
        """An unfinished marker at the end of the stream is sent as it is."""
        renumberer = CitationRenumberer()
        assert renumberer.feed("ราคา [") == "ราคา "
        assert renumberer.flush() == "["

    def test_mapping_can_change_mid_stream(self) -> None:
        """Numbers learned after a tool call apply to later text."""
        renumberer = CitationRenumberer()
        first = renumberer.feed("[1] ")
        renumberer.mapping = {1: 4}
        assert first + renumberer.feed("[1]") == "[1] [4]"
