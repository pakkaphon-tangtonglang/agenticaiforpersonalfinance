"""Convert agent markdown replies to LINE-friendly plain text.

LINE text bubbles render markdown literally — users would see raw
``#``, ``**`` and table pipes. This module strips the markup at the
push boundary, so the saved conversation history keeps markdown for
the web UI while LINE receives clean readable text.
"""

import re
from typing import Optional

_TABLE_SEPARATOR_CELL = re.compile(r":?-{1,}:?")
_HEADING = re.compile(r"^#{1,6}\s*")
_HORIZONTAL_RULE = re.compile(r"^-{3,}$")
_BULLET = re.compile(r"^-\s+")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_BOLD_UNDERSCORE = re.compile(r"__(.+?)__")
_ITALIC = re.compile(r"\*([^*\n]+)\*")
_ITALIC_UNDERSCORE = re.compile(r"_([^_\n]+)_")

_LINE_DIVIDER = "————————"


def markdown_to_line_text(markdown: str) -> str:
    """Convert a markdown reply to plain text suitable for LINE.

    Strips headings, emphasis, and blockquote markers; converts tables
    to pipe-separated lines (dropping separator rows); renders bullets
    with a dot marker and horizontal rules as a divider line.

    Args:
        markdown: Agent response text (GitHub-flavored markdown).

    Returns:
        Plain-text version with collapsed blank lines, trimmed edges.

    Example:
        >>> markdown_to_line_text("**หุ้น PTT** ราคา 42.00")
        'หุ้น PTT ราคา 42.00'
    """
    converted_lines: list[str] = []
    for line in markdown.splitlines():
        converted = _convert_line(line)
        if converted is not None:
            converted_lines.append(converted)
    text = "\n".join(converted_lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _convert_line(line: str) -> Optional[str]:
    """Convert one markdown line, or None to drop it (separator rows).

    Args:
        line: Single raw markdown line.

    Returns:
        Converted plain-text line, or None when the line should be
        dropped entirely.
    """
    stripped = line.strip()
    if not stripped:
        return line
    if stripped.startswith(">"):
        stripped = stripped[1:].strip()
    stripped = _HEADING.sub("", stripped)
    stripped = _strip_emphasis(stripped)
    if stripped.startswith("|"):
        return _convert_table_line(stripped)
    if _HORIZONTAL_RULE.fullmatch(stripped):
        return _LINE_DIVIDER
    return _BULLET.sub("• ", stripped)


def _strip_emphasis(text: str) -> str:
    """Remove bold/italic markers and inline code backticks.

    Args:
        text: Line text possibly containing emphasis markup.

    Returns:
        Text with emphasis markers removed.
    """
    text = _BOLD.sub(r"\1", text)
    text = _BOLD_UNDERSCORE.sub(r"\1", text)
    text = _ITALIC.sub(r"\1", text)
    text = _ITALIC_UNDERSCORE.sub(r"\1", text)
    return text.replace("`", "")


def _convert_table_line(line: str) -> Optional[str]:
    """Convert one markdown table row to a plain pipe-separated line.

    Args:
        line: Table row starting and ending with ``|``.

    Returns:
        Pipe-separated cells, or None for the ``|---|`` separator row.
    """
    cells = [cell.strip() for cell in line.strip("|").split("|")]
    non_empty_cells = [cell for cell in cells if cell]
    if not non_empty_cells:
        return None
    if all(_TABLE_SEPARATOR_CELL.fullmatch(cell) for cell in non_empty_cells):
        return None
    return " | ".join(non_empty_cells)
