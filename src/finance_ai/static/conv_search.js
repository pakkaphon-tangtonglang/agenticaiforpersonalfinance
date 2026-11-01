/* Personal Finance AI — sidebar chat search (pure helpers)
   Matching is case-insensitive and ignores repeated spaces. The searched text
   is the chat title plus anything the caller adds (guru name, guru id), so
   "buffet" finds every Buffett chat on the Guru tab. Runs in the browser
   (window.ConvSearch) and in node for tests (module.exports).
*/
(function (root) {
  "use strict";

  function normalize(text) {
    return String(text || "").normalize("NFC").toLocaleLowerCase("th").replace(/\s+/g, " ").trim();
  }

  // True when every query character sequence appears in the text (empty query matches all).
  function matches(text, query) {
    const wanted = normalize(query);
    return !wanted || normalize(text).includes(wanted);
  }

  // Split a title around its first match so the match can be marked; null when none.
  function splitMatch(title, query) {
    const wanted = normalize(query);
    const text = String(title || "");
    if (!wanted) return null;
    const at = text.normalize("NFC").toLocaleLowerCase("th").indexOf(wanted);
    if (at < 0) return null;
    return { before: text.slice(0, at), match: text.slice(at, at + wanted.length), after: text.slice(at + wanted.length) };
  }

  const api = { matches, splitMatch };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ConvSearch = api;
})(typeof window !== "undefined" ? window : this);
