"""Split a long reply into LINE text messages.

LINE allows at most 5,000 characters per text message and 5 messages per
push, so a long multi-topic reply is cut at paragraph breaks.
"""

LINE_TEXT_LIMIT = 5000
LINE_MAX_MESSAGES = 5
_PARAGRAPH_BREAK = "\n\n"
_ELLIPSIS = "…"


def split_line_text(
    text: str, limit: int = LINE_TEXT_LIMIT, max_messages: int = LINE_MAX_MESSAGES
) -> list[str]:
    """Split text into LINE-sized messages, preferring paragraph breaks.

    Args:
        text: Plain reply text.
        limit: Maximum characters per message.
        max_messages: Maximum messages in one push; extra text is cut with "…".

    Returns:
        One or more messages, each at most ``limit`` characters.

    Example:
        >>> split_line_text("ก" * 6, limit=4)
        ['กกกก', 'กก']
    """
    chunks = _pack_paragraphs(text.split(_PARAGRAPH_BREAK), limit)
    if len(chunks) <= max_messages:
        return chunks or [text]
    rest = _PARAGRAPH_BREAK.join(chunks[max_messages - 1 :])
    return chunks[: max_messages - 1] + [rest[: limit - len(_ELLIPSIS)] + _ELLIPSIS]


def _pack_paragraphs(paragraphs: list[str], limit: int) -> list[str]:
    """Join paragraphs into chunks of at most ``limit`` characters."""
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        joined = f"{current}{_PARAGRAPH_BREAK}{paragraph}" if current else paragraph
        if len(joined) <= limit:
            current = joined
            continue
        if current:
            chunks.append(current)
        pieces = _hard_split(paragraph, limit)
        chunks.extend(pieces[:-1])
        current = pieces[-1]
    if current:
        chunks.append(current)
    return chunks


def _hard_split(paragraph: str, limit: int) -> list[str]:
    """Cut one paragraph longer than ``limit`` into fixed-size pieces."""
    return [paragraph[start : start + limit] for start in range(0, len(paragraph), limit)] or [""]
