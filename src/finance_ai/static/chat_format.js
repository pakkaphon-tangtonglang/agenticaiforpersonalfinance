/* Personal Finance AI — chat text normalisation before markdown parsing.
   marked runs with breaks: false (single newlines soft-wrap, like GPT/Claude),
   so a line that is not a markdown list item is joined onto the line above.
   Models often write items as "• a", "– a", "✅ a" or "**label:** value",
   which then ran together on one line. This turns bullet glyphs into real
   list items and keeps item-like lines on their own line. Code fences are
   left untouched. Works in the browser (window.ChatFormat) and in node
   (module.exports) so it can be tested without a DOM.
*/
(function (root) {
  "use strict";

  const BULLET_GLYPH = /^(\s*)[•·▪–—](?=\s)\s*/;
  const ITEM_LIKE_START = /^\s*(\*\*|\p{Extended_Pictographic})/u;
  const FENCE = /^\s*(```|~~~)/;
  const HARD_BREAK = "  ";

  // A line that may end with a hard break: has text and is not a table row.
  function canBreakAfter(line) {
    const trimmed = line.trim();
    return trimmed !== "" && !trimmed.startsWith("|");
  }

  // Rewrite lines outside code fences; inFence tracks ``` / ~~~ blocks.
  function fixLines(lines) {
    let inFence = false;
    return lines.map((line, index) => {
      if (FENCE.test(line)) { inFence = !inFence; return line; }
      if (inFence) return line;
      const fixed = line.replace(BULLET_GLYPH, "$1- ");
      const next = lines[index + 1];
      const breakBefore = next !== undefined && !FENCE.test(next)
        && ITEM_LIKE_START.test(next.replace(BULLET_GLYPH, "$1- "));
      return breakBefore && canBreakAfter(fixed) ? fixed.trimEnd() + HARD_BREAK : fixed;
    });
  }

  // Collapse 3+ newlines, normalise CRLF, trim, then fix bullet lines.
  function normalizeForChat(text) {
    const cleaned = String(text == null ? "" : text)
      .replace(/\r\n/g, "\n")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
    return fixLines(cleaned.split("\n")).join("\n");
  }

  const api = { normalizeForChat };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ChatFormat = api;
})(typeof window !== "undefined" ? window : this);
