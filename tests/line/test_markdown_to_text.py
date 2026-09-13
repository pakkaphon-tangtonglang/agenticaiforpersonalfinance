"""Tests for markdown → LINE plain-text conversion.

LINE text bubbles render markdown literally (users see raw #, **, |),
so agent replies are stripped to clean plain text at the push boundary.
"""

from finance_ai.line.markdown_to_text import markdown_to_line_text


class TestMarkdownToLineText:
    """Tests for markdown_to_line_text."""

    def test_strips_heading_markers(self) -> None:
        """# / ## markers are removed, heading text kept."""
        assert markdown_to_line_text("# ราคาหุ้น PTT") == "ราคาหุ้น PTT"
        assert markdown_to_line_text("## สรุป") == "สรุป"

    def test_strips_emphasis(self) -> None:
        """Bold/italic/bold-italic markers are removed, text kept."""
        assert markdown_to_line_text("**42.00 บาท**") == "42.00 บาท"
        assert markdown_to_line_text("*เน้นข้อความ*") == "เน้นข้อความ"
        assert markdown_to_line_text("__หนา__") == "หนา"
        assert markdown_to_line_text("`CODE`") == "CODE"

    def test_converts_table_to_lines(self) -> None:
        """Table rows become plain lines; separator rows are dropped."""
        markdown = "| รายการ | ข้อมูล |\n" "|---|---|\n" "| **ราคาปัจจุบัน** | 42.00 บาท |\n"
        result = markdown_to_line_text(markdown)

        assert "รายการ | ข้อมูล" in result
        assert "ราคาปัจจุบัน | 42.00 บาท" in result
        assert "---" not in result

    def test_bullets_use_dot_marker(self) -> None:
        """Markdown '- ' bullets render as '• '."""
        assert markdown_to_line_text("- รายการแรก") == "• รายการแรก"

    def test_strips_blockquote_marker(self) -> None:
        """'> quoted' keeps the text without the marker."""
        assert markdown_to_line_text("> หมายเหตุ") == "หมายเหตุ"

    def test_horizontal_rule_becomes_divider(self) -> None:
        """A lone '---' rule renders as a line divider."""
        assert markdown_to_line_text("ก่อน\n---\nหลัง") == "ก่อน\n————————\nหลัง"

    def test_collapses_extra_blank_lines(self) -> None:
        """Runs of 3+ newlines collapse to one blank line."""
        assert markdown_to_line_text("a\n\n\n\nb") == "a\n\nb"

    def test_trims_edges(self) -> None:
        """Leading/trailing blank lines are removed."""
        assert markdown_to_line_text("\n\nสวัสดี\n") == "สวัสดี"

    def test_plain_text_unchanged(self) -> None:
        """Text without markdown passes through untouched."""
        assert markdown_to_line_text("สวัสดีครับ ราคา 42.00 บาท") == ("สวัสดีครับ ราคา 42.00 บาท")

    def test_full_stock_reply(self) -> None:
        """A realistic agent reply converts to readable plain text."""
        markdown = (
            "## ราคาหุ้น PTT (PTT.BK) 🇹🇭\n\n"
            "| รายการ | ข้อมูล |\n"
            "|---|---|\n"
            "| **ราคาปัจจุบัน** | **42.00 บาท** |\n\n"
            "- **ชื่อ**: PTT\n"
            "- **ตลาด**: SET\n"
        )
        result = markdown_to_line_text(markdown)

        assert "##" not in result
        assert "**" not in result
        assert "|---|" not in result
        assert "ราคาหุ้น PTT (PTT.BK) 🇹🇭" in result
        assert "ราคาปัจจุบัน | 42.00 บาท" in result
        assert "• ชื่อ: PTT" in result
