"""Convert architecture_comparison.md to a styled PDF using fpdf2 with Thai font.

Usage:
    python scripts/md_to_pdf.py

Output:
    data/evaluation/results/architecture_comparison.pdf
"""

from __future__ import annotations

import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fpdf import FPDF

# ──────────────────────── Config ────────────────────────

INPUT_MD = "data/evaluation/results/architecture_comparison.md"
OUTPUT_PDF = "data/evaluation/results/architecture_comparison.pdf"
THAI_FONT_PATH = "C:/Windows/Fonts/cordia.ttc"

# Colors
COLOR_HEADER_BG = (41, 128, 185)      # blue
COLOR_HEADER_TEXT = (255, 255, 255)   # white
COLOR_ROW_ALT = (236, 240, 241)       # light gray
COLOR_ROW_NORMAL = (255, 255, 255)    # white
COLOR_H1 = (44, 62, 80)              # dark navy
COLOR_H2 = (41, 128, 185)            # blue
COLOR_GRID = (189, 195, 199)          # light gray border
COLOR_SUCCESS_BG = (212, 239, 223)    # light green
COLOR_SUCCESS_TEXT = (30, 100, 60)    # dark green

# Sizes
PAGE_W = 297   # A4 landscape
PAGE_H = 210
MARGIN = 12


# ──────────────────────── PDF class ────────────────────────

class ArchitecturePDF(FPDF):
    """Custom PDF renderer with Thai font and styled components."""

    def __init__(self) -> None:
        super().__init__(orientation="L", unit="mm", format="A4")
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.set_auto_page_break(auto=True, margin=MARGIN)
        self.add_font("Thai", "", THAI_FONT_PATH)
        self.add_font("Thai", "B", THAI_FONT_PATH)

    def header(self) -> None:
        """Page header with subtle title."""
        self.set_font("Thai", size=8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 6, "เปรียบเทียบสถาปัตยกรรม Multi-Agent", align="R")
        self.ln(4)

    def footer(self) -> None:
        """Page number footer."""
        self.set_y(-10)
        self.set_font("Thai", size=8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 6, f"หน้า {self.page_no()}", align="C")

    def add_h1(self, text: str) -> None:
        """Render a level-1 heading."""
        self.set_font("Thai", "B", 18)
        self.set_text_color(*COLOR_H1)
        self.ln(2)
        self.multi_cell(0, 10, _break_long_words(text.lstrip("# ").strip()))
        self.set_draw_color(*COLOR_H2)
        self.set_line_width(0.5)
        self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
        self.ln(4)

    def add_h2(self, text: str) -> None:
        """Render a level-2 heading."""
        self.ln(3)
        self.set_font("Thai", "B", 13)
        self.set_text_color(*COLOR_H2)
        self.multi_cell(0, 8, _break_long_words(text.lstrip("# ").strip()))
        self.ln(2)

    def add_paragraph(self, text: str) -> None:
        """Render a paragraph with Thai font."""
        clean = _strip_markdown_formatting(text)
        if not clean.strip():
            return
        self.set_font("Thai", size=10)
        self.set_text_color(50, 50, 50)
        self.set_x(self.l_margin)
        self.multi_cell(0, 6, _break_long_words(clean))
        self.ln(2)

    def add_blockquote(self, text: str) -> None:
        """Render a blockquote (conclusion box) with green background."""
        clean = _strip_markdown_formatting(text.lstrip("> "))
        self.set_fill_color(*COLOR_SUCCESS_BG)
        self.set_draw_color(*COLOR_SUCCESS_TEXT)
        self.set_line_width(0.3)
        x = self.get_x()
        y = self.get_y()
        # Draw left bar
        self.set_fill_color(*COLOR_H2)
        self.rect(x, y, 2, 12, "F")
        self.set_x(x + 4)
        self.set_font("Thai", size=10)
        self.set_text_color(*COLOR_SUCCESS_TEXT)
        self.multi_cell(0, 6, _break_long_words(clean))
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def add_table(self, headers: list[str], rows: list[list[str]]) -> None:
        """Render a styled table with alternating row colors.

        Args:
            headers: Column header strings.
            rows: List of row data (each row is a list of strings).
        """
        if not headers:
            return

        usable_w = PAGE_W - 2 * MARGIN
        col_widths = _compute_col_widths(headers, rows, usable_w)

        # Header row
        self.set_font("Thai", "B", 9)
        self.set_fill_color(*COLOR_HEADER_BG)
        self.set_text_color(*COLOR_HEADER_TEXT)
        self.set_draw_color(*COLOR_GRID)
        self.set_line_width(0.2)

        row_h = 7
        for i, (header, w) in enumerate(zip(headers, col_widths)):
            self.cell(w, row_h, header.strip(), border=1, fill=True, align="C")
        self.ln()

        # Data rows
        self.set_font("Thai", size=8)
        self.set_text_color(30, 30, 30)
        for r_idx, row in enumerate(rows):
            fill_color = COLOR_ROW_ALT if r_idx % 2 == 0 else COLOR_ROW_NORMAL
            self.set_fill_color(*fill_color)
            for i, (cell_val, w) in enumerate(zip(row, col_widths)):
                text = cell_val.strip().replace("**", "").replace("✓", "v").replace("✗", "x")
                align = "R" if _is_numeric(text) else "L"
                self.cell(w, row_h, text, border=1, fill=True, align=align)
            self.ln()

        self.ln(3)


# ──────────────────────── Markdown Parser ────────────────────────

def _break_long_words(text: str, max_chars: int = 30) -> str:
    """Insert spaces into unbreakable words longer than max_chars.

    fpdf2 raises an error when a single word exceeds the available width.
    This prevents that by splitting very long tokens with a space.

    Args:
        text: Input text that may contain long unbreakable tokens.
        max_chars: Maximum token length before inserting a break.

    Returns:
        Text with long tokens split.
    """
    import re as _re
    tokens = _re.split(r"(\s+)", text)
    result = []
    for token in tokens:
        if len(token) > max_chars and " " not in token:
            # Insert a space every max_chars characters
            parts = [token[i:i + max_chars] for i in range(0, len(token), max_chars)]
            result.append(" ".join(parts))
        else:
            result.append(token)
    return "".join(result)


def _strip_markdown_formatting(text: str) -> str:
    """Remove bold/italic markdown syntax from text.

    Args:
        text: Raw markdown text.

    Returns:
        Plain text string.
    """
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    return text


def _is_numeric(text: str) -> bool:
    """Return True if text looks like a number or measurement.

    Args:
        text: Cell text.

    Returns:
        True if numeric.
    """
    cleaned = text.replace(",", "").replace("ms", "").replace(" ", "")
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def _compute_col_widths(
    headers: list[str],
    rows: list[list[str]],
    usable_w: float,
) -> list[float]:
    """Compute column widths proportional to content length.

    Args:
        headers: Column headers.
        rows: Table data rows.
        usable_w: Available page width.

    Returns:
        List of column widths in mm.
    """
    n = len(headers)
    if n == 0:
        return []

    # Estimate character counts per column
    char_counts = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row[:n]):
            char_counts[i] = max(char_counts[i], len(cell))

    total_chars = sum(char_counts) or 1
    min_w = usable_w / (n * 2)

    widths = []
    for c in char_counts:
        raw = (c / total_chars) * usable_w
        widths.append(max(raw, min_w))

    # Scale to exactly usable_w
    scale = usable_w / sum(widths)
    return [w * scale for w in widths]


def _parse_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    """Parse markdown table lines into headers and rows.

    Args:
        lines: Raw markdown table lines (including separator row).

    Returns:
        Tuple of (headers, rows).
    """
    if len(lines) < 2:
        return [], []

    def split_row(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    headers = split_row(lines[0])
    rows = []
    for line in lines[2:]:   # skip separator
        if line.strip().startswith("|"):
            rows.append(split_row(line))

    return headers, rows


# ──────────────────────── Main Renderer ────────────────────────

def render_markdown_to_pdf(md_path: str, pdf_path: str) -> None:
    """Read a markdown file and render it as a styled PDF.

    Args:
        md_path: Path to the input markdown file.
        pdf_path: Path for the output PDF.
    """
    with open(md_path, encoding="utf-8") as f:
        lines = f.readlines()

    pdf = ArchitecturePDF()
    pdf.add_page()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # H1
        if line.startswith("# ") and not line.startswith("## "):
            pdf.add_h1(line)
            i += 1

        # H2
        elif line.startswith("## "):
            pdf.add_h2(line)
            i += 1

        # Horizontal rule
        elif line.strip() == "---":
            pdf.set_draw_color(*COLOR_GRID)
            pdf.set_line_width(0.3)
            pdf.line(MARGIN, pdf.get_y(), PAGE_W - MARGIN, pdf.get_y())
            pdf.ln(3)
            i += 1

        # Table (starts with |)
        elif line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].rstrip())
                i += 1
            headers, rows = _parse_table(table_lines)
            pdf.add_table(headers, rows)

        # Blockquote
        elif line.startswith(">"):
            pdf.add_blockquote(line)
            i += 1

        # Numbered/bullet list item (handles optional indentation like "   - ")
        elif re.match(r"^\s*\d+\.", line) or re.match(r"^\s*-\s", line):
            clean = _strip_markdown_formatting(
                re.sub(r"^\s*\d+\.\s*|\s*-\s", "", line, count=1)
            )
            if clean.strip():
                pdf.set_font("Thai", size=10)
                pdf.set_text_color(50, 50, 50)
                pdf.set_x(pdf.l_margin + 4)
                pdf.multi_cell(0, 6, _break_long_words(clean))
            i += 1

        # Empty line
        elif not line.strip():
            pdf.ln(1)
            i += 1

        # Normal paragraph
        else:
            pdf.add_paragraph(line)
            i += 1

    pdf.output(pdf_path)
    print(f"PDF saved: {os.path.abspath(pdf_path)}")


# ──────────────────────── Entry Point ────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert markdown to styled PDF")
    parser.add_argument("input", nargs="?", default=INPUT_MD, help="Input .md file")
    parser.add_argument("output", nargs="?", default=OUTPUT_PDF, help="Output .pdf file")
    args = parser.parse_args()
    render_markdown_to_pdf(args.input, args.output)
