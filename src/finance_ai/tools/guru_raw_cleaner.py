"""Trim fetched guru pages (data/guru_raw) down to the article body.

Per-site rules cut menus, view counts, tag blocks and "related articles"
footers, then rejoin Thai text that inline HTML tags split into fragments.
Unknown sites (e.g. Berkshire PDFs) only lose bare URL lines; they still
need manual trimming. Output goes to data/guru_clean for human review.
No LLM calls.
"""

import re
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel

from finance_ai.rag.document_loader import parse_front_matter

FRAGMENT_LENGTH = 40
_URL_LINE = re.compile(r"^https?://\S+$")


class CleanRule(BaseModel):
    """Where the article body starts and ends on one site."""

    start_pattern: str | None = None
    start_offset: int = 1
    use_last_start_in_head: int = 0
    end_pattern: str | None = None
    drop_patterns: list[str] = []


SITE_RULES: dict[str, CleanRule] = {
    "finnomena.com": CleanRule(start_pattern=r"^[\d,]+$", end_pattern=r"^แท็ก:$"),
    "moneycoach.co.th": CleanRule(start_pattern=r"^< BACK$", start_offset=2),
    "thairath.co.th": CleanRule(
        start_pattern=r"^#$",
        start_offset=2,
        use_last_start_in_head=60,
        end_pattern=r"^อ่านข่าว",
        drop_patterns=[r" \| "],  # embedded video widget titles, e.g. "... | Digital Frontiers"
    ),
}


def rule_for_url(url: str) -> CleanRule | None:
    """Return the cleaning rule for a source URL's site, if any.

    Example:
        >>> rule_for_url("https://www.finnomena.com/dr-niwes/x") is not None
        True
    """
    host = urlsplit(url).netloc.removeprefix("www.")
    return SITE_RULES.get(host)


def _start_index(lines: list[str], rule: CleanRule) -> int:
    """Index of the first body line (0 when the start marker is missing)."""
    if rule.start_pattern is None:
        return 0
    head = lines[: rule.use_last_start_in_head] if rule.use_last_start_in_head else lines
    matches = [i for i, line in enumerate(head) if re.match(rule.start_pattern, line)]
    if not matches:
        return 0
    chosen = matches[-1] if rule.use_last_start_in_head else matches[0]
    return chosen + rule.start_offset


def _end_index(lines: list[str], start: int, rule: CleanRule) -> int:
    """Index just past the last body line (end of text when no marker)."""
    if rule.end_pattern is not None:
        for index in range(start, len(lines)):
            if re.match(rule.end_pattern, lines[index]):
                return index
    return len(lines)


def _is_dropped(line: str, rule: CleanRule) -> bool:
    """True for bare URLs and lines matching the site's drop patterns."""
    if _URL_LINE.match(line):
        return True
    return any(re.search(pattern, line) for pattern in rule.drop_patterns)


def _join_fragments(lines: list[str]) -> list[str]:
    """Rejoin short fragments (split by inline tags) with their neighbours."""
    joined: list[str] = []
    for line in lines:
        previous_is_short = bool(joined) and len(joined[-1]) < FRAGMENT_LENGTH
        if joined and (previous_is_short or len(line) < FRAGMENT_LENGTH):
            joined[-1] = f"{joined[-1]} {line}"
        else:
            joined.append(line)
    return joined


def clean_text(text: str, rule: CleanRule | None) -> str:
    """Cut a page's text down to its article body.

    Args:
        text: Extracted page text (one block per line).
        rule: Site rule, or None to only drop bare URL lines.

    Returns:
        Cleaned body text. Missing markers keep the whole text.

    Example:
        >>> clean_text("1,000\\nเนื้อหา\\nแท็ก:", rule_for_url("https://www.finnomena.com/a"))
        'เนื้อหา'
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if rule is not None:
        start = _start_index(lines, rule)
        lines = lines[start : _end_index(lines, start, rule)]
        lines = [line for line in lines if not _is_dropped(line, rule)]
        lines = _join_fragments(lines)
    else:
        lines = [line for line in lines if not _URL_LINE.match(line)]
    return "\n".join(lines)


def _strip_heading(body: str) -> str:
    """Remove the '# title' line the fetcher puts above the page text."""
    return re.sub(r"\A\s*# .*\n", "", body, count=1)


def clean_raw_file(raw_path: Path, output_root: Path) -> Path:
    """Write a cleaned copy of one fetched file for review.

    Args:
        raw_path: data/guru_raw/<guru_id>/<name>.md
        output_root: Root for cleaned copies (data/guru_clean).

    Returns:
        Path of the cleaned copy (the raw file is not modified).

    Example:
        >>> clean_raw_file(Path("data/guru_raw/niwes/a.md"), Path("data/guru_clean"))
    """
    front_matter, body = parse_front_matter(raw_path.read_text(encoding="utf-8"))
    cleaned = clean_text(
        _strip_heading(body), rule_for_url(str(front_matter.get("source_url", "")))
    )
    header = yaml.safe_dump({**front_matter, "cleaned": True}, allow_unicode=True, sort_keys=False)
    title = front_matter.get("source_title", raw_path.stem)
    target = output_root / raw_path.parent.name / raw_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"---\n{header}---\n# {title}\n\n{cleaned}\n", encoding="utf-8")
    return target
