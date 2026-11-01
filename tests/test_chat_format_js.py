"""Tests for static/chat_format.js (chat text normalisation before markdown).

marked runs with ``breaks: false``, so a line that is not a markdown list
item is joined onto the line above. These tests pin the fix for plan
item 3.4 (missing newline between bullets). They run the real JS in node
and are skipped when node is not installed.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

CHAT_FORMAT_JS = (
    Path(__file__).resolve().parents[1] / "src" / "finance_ai" / "static" / "chat_format.js"
)
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def normalize(text: str) -> str:
    """Run ChatFormat.normalizeForChat in node and return its output.

    Args:
        text: Raw model output.

    Returns:
        The normalised markdown string.
    """
    script = (
        f"const f = require({json.dumps(str(CHAT_FORMAT_JS))});"
        "let s = ''; process.stdin.on('data', (d) => (s += d));"
        "process.stdin.on('end', () => process.stdout.write("
        "JSON.stringify(f.normalizeForChat(JSON.parse(s)))));"
    )
    completed = subprocess.run(
        [str(NODE), "-e", script],
        input=json.dumps(text),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return str(json.loads(completed.stdout))


class TestExistingCleanup:
    """Behaviour kept from the old inline normalizeForChat."""

    def test_crlf_collapsed_runs_and_trim(self) -> None:
        """CRLF becomes LF, 3+ newlines become one blank line, edges trimmed."""
        assert normalize("  a\r\n\r\n\r\n\r\nb  \n") == "a\n\nb"

    def test_none_like_empty_input(self) -> None:
        """Empty input stays empty."""
        assert normalize("") == ""

    def test_plain_soft_wrap_untouched(self) -> None:
        """An ordinary continuation line is still a soft wrap."""
        assert normalize("ประโยคแรก\nต่อบรรทัด") == "ประโยคแรก\nต่อบรรทัด"


class TestBulletCharacters:
    """Non-markdown bullet glyphs become real list items."""

    @pytest.mark.parametrize("glyph", ["•", "·", "▪", "–", "—"])
    def test_glyph_becomes_dash_item(self, glyph: str) -> None:
        """'• item' lines become '- item' so marked renders a list."""
        text = f"รายจ่าย:\n{glyph} ค่าอาหาร\n{glyph} ค่าเดินทาง"
        assert normalize(text) == "รายจ่าย:\n- ค่าอาหาร\n- ค่าเดินทาง"

    def test_indented_glyph_keeps_indent(self) -> None:
        """Nested bullets keep their indentation."""
        assert normalize("- a\n  • b") == "- a\n  - b"

    def test_glyph_mid_line_untouched(self) -> None:
        """A bullet glyph inside a sentence is not a list marker."""
        assert normalize("ราคา • 30 บาท") == "ราคา • 30 บาท"


class TestLineStarts:
    """Lines that read as items keep their own line (hard break before them)."""

    def test_emoji_lines_keep_own_line(self) -> None:
        """'✅ a' / '⚠️ b' lines each start on a new line."""
        assert normalize("สรุป:\n✅ มีเงินสำรอง\n⚠️ หนี้สูง") == (
            "สรุป:  \n✅ มีเงินสำรอง  \n⚠️ หนี้สูง"
        )

    def test_bold_label_lines_keep_own_line(self) -> None:
        """'**label:** value' lines each start on a new line."""
        assert normalize("**รายได้:** 40,000\n**รายจ่าย:** 30,000") == (
            "**รายได้:** 40,000  \n**รายจ่าย:** 30,000"
        )

    def test_markdown_lists_and_tables_untouched(self) -> None:
        """Real markdown structure needs no help."""
        text = "ขั้นตอน:\n1. ออม\n2. ลงทุน\n\n| a | b |\n|---|---|\n| 1 | 2 |"
        assert normalize(text) == text

    def test_code_fence_untouched(self) -> None:
        """Text inside ``` fences is left exactly as written."""
        text = "```\n• keep\n**keep**\n```"
        assert normalize(text) == text
