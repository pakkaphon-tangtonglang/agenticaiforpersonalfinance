"""Tests for numbered citation sources built from RAG results."""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from finance_ai.agents.citation_sources import (
    SNIPPET_MAX_CHARS,
    CitationSource,
    build_citations,
    collect_turn_citations,
    format_citations_as_context,
)
from finance_ai.rag.document_models import RetrievalResult


def _result(source_file: str, content: str = "เนื้อหา", **fields: object) -> RetrievalResult:
    """Build a retrieval result with sensible defaults."""
    base = {"content": content, "source_file": source_file, "domain": "tax", "relevance_score": 0.5}
    return RetrievalResult.model_validate({**base, **fields})


def _tool_message(
    citations: list[CitationSource], name: str = "search_guru_knowledge"
) -> ToolMessage:
    """Wrap citations like ToolNode serialises a dict tool result."""
    payload = {"citations": [citation.model_dump() for citation in citations]}
    return ToolMessage(content=json.dumps(payload, ensure_ascii=False), tool_call_id="t", name=name)


class TestBuildCitations:
    """Numbering, de-duplication and display fields."""

    def test_numbers_each_document_once(self) -> None:
        """Two chunks of one document share a number."""
        results = [_result("a.md", "ก"), _result("a.md", "ข"), _result("b.md", "ค")]

        citations = build_citations(results, [])

        assert [citation.index for citation in citations] == [1, 2]
        assert citations[0].snippet == "ก"

    def test_continues_numbering_after_existing(self) -> None:
        """A second search in the same turn keeps earlier numbers."""
        first = build_citations([_result("a.md")], [])

        second = build_citations([_result("a.md"), _result("c.md")], first)

        assert [citation.index for citation in second] == [1, 2]
        assert second[0].document_key == first[0].document_key

    def test_display_fields_never_use_filename(self) -> None:
        """Title, publisher, url, domain and kind come from metadata."""
        result = _result(
            "tax_guide.md",
            title="คู่มือภาษี",
            source_title="คู่มือภาษีเงินได้",
            source_publisher="กรมสรรพากร",
            source_url="https://www.rd.go.th/546.html",
        )

        citation = build_citations([result], [])[0]

        assert citation.title == "คู่มือภาษี"
        assert citation.publisher == "กรมสรรพากร"
        assert citation.domain == "rd.go.th"
        assert citation.kind == "compiled"
        assert "tax_guide" not in citation.model_dump_json()

    def test_missing_title_falls_back_to_source_title_then_generic(self) -> None:
        """Old index entries without a title still get a readable label."""
        with_source = build_citations([_result("a.md", source_title="Owner's Manual")], [])[0]
        bare = build_citations([_result("b.md")], [])[0]

        assert with_source.title == "Owner's Manual"
        assert bare.title == "เอกสารอ้างอิง"

    def test_snippet_is_truncated(self) -> None:
        """Long chunks are shortened for the popover."""
        citation = build_citations([_result("a.md", "x" * 1000)], [])[0]

        assert len(citation.snippet) <= SNIPPET_MAX_CHARS
        assert citation.snippet.endswith("…")

    def test_snippet_drops_markdown_syntax(self) -> None:
        """Headings, emphasis and table pipes do not leak into the popover."""
        chunk = "# กองทุนดัชนี\n## *คำแนะนำ*\n**ค่าธรรมเนียมต่ำ** | 90% S&P 500 |"

        citation = build_citations([_result("a.md", chunk)], [])[0]

        assert citation.snippet == "กองทุนดัชนี คำแนะนำ ค่าธรรมเนียมต่ำ 90% S&P 500"

    def test_snippet_skips_repeated_title_and_note(self) -> None:
        """The card already shows the title; the doc's '>' note is not content."""
        body = "ผลลัพธ์ทางการเงินขึ้นกับพฤติกรรมมากกว่าความรู้หรือความฉลาด"
        chunk = f"# จิตวิทยาการเงิน\n\n> เอกสารนี้เป็นการสรุปแนวคิด\n\n## แนวคิดหลัก\n{body}"

        citation = build_citations([_result("a.md", chunk, title="จิตวิทยาการเงิน")], [])[0]

        assert citation.snippet == f"แนวคิดหลัก {body}"

    def test_snippet_keeps_note_when_little_else_remains(self) -> None:
        """An intro-only chunk shows its note instead of a bare heading."""
        note = "เอกสารนี้เป็นการสรุปแนวคิดหลักด้วยถ้อยคำของทีมผู้พัฒนา"
        chunk = f"# จิตวิทยาการเงิน\n> {note}\n## แนวคิดหลัก"

        citation = build_citations([_result("a.md", chunk, title="จิตวิทยาการเงิน")], [])[0]

        assert citation.snippet == f"{note} แนวคิดหลัก"

    def test_snippet_falls_back_to_full_chunk(self) -> None:
        """A chunk that is only a short heading + note keeps everything."""
        chunk = "# คู่มือ\n> หมายเหตุ"

        citation = build_citations([_result("a.md", chunk, title="คู่มือ")], [])[0]

        assert citation.snippet == "คู่มือ หมายเหตุ"

    def test_empty_results(self) -> None:
        """No results means no citations."""
        assert build_citations([], []) == []


class TestFormatCitationsAsContext:
    """The model sees numbers and titles, not filenames."""

    def test_labels_chunks_with_numbers(self) -> None:
        """Every chunk is prefixed with its document's number and title."""
        results = [_result("a.md", "ก", title="หนี้"), _result("a.md", "ข", title="หนี้")]
        citations = build_citations(results, [])

        context = format_citations_as_context(results, citations)

        assert context == "[1] หนี้\nก\n\n[1] หนี้\nข"
        assert "a.md" not in context


class TestCollectTurnCitations:
    """Citations are read from search ToolMessages of the latest turn."""

    def test_collects_from_tool_messages_after_last_question(self) -> None:
        """Earlier turns are ignored; duplicates across searches collapse."""
        old = build_citations([_result("old.md")], [])
        current = build_citations([_result("a.md"), _result("b.md")], [])
        messages = [
            HumanMessage(content="ก่อนหน้า"),
            _tool_message(old),
            AIMessage(content="ตอบเก่า"),
            HumanMessage(content="คำถามใหม่"),
            _tool_message(current[:1]),
            _tool_message(current),
        ]

        citations = collect_turn_citations(messages)

        assert [citation.index for citation in citations] == [1, 2]
        assert citations[1].document_key == current[1].document_key

    def test_ignores_other_tools_and_bad_json(self) -> None:
        """Non-search tools and unparsable content are skipped."""
        messages = [
            HumanMessage(content="q"),
            ToolMessage(content="not json", tool_call_id="t", name="search_finance_knowledge"),
            ToolMessage(
                content='{"citations": "x"}', tool_call_id="t", name="search_finance_knowledge"
            ),
            ToolMessage(content='{"total": 5}', tool_call_id="t", name="get_expenses"),
        ]

        assert collect_turn_citations(messages) == []

    def test_accepts_tuple_messages_and_empty_list(self) -> None:
        """Input-state tuples are not ToolMessages and are skipped."""
        assert collect_turn_citations([("user", "q")]) == []
        assert collect_turn_citations([]) == []
