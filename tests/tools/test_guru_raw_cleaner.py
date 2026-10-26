"""Tests for cleaning fetched guru pages down to the article body."""

from pathlib import Path

from finance_ai.tools.guru_raw_cleaner import (
    clean_raw_file,
    clean_text,
    markers_found,
    rule_for_entry,
    rule_for_url,
)
from finance_ai.tools.guru_source_fetcher import SourceEntry

FINNOMENA_PAGE = """แจ้งเตือน
คลังความรู้
ยุคสินค้า Value (for money)
Dr.Niwes Hemvachiravarakorn
8 ก.ย. 68
1,192
ยุคของหุ้น “Value”
ในตลาดหุ้นไทย หลายคนอาจจะบอกว่ายุคของหุ้นคุณค่าได้ผ่านไปแล้วจริงหรือไม่ลองมาดูกัน
แท็ก:
Kept
บทความที่เกี่ยวข้อง
"""

THAIRATH_PAGE = """Personal Finance
กองบรรณาธิการ
Tag
#
อิสรภาพทางการเงิน
#
โค้ชหนุ่ม
ความเสี่ยงรอบใหม่ของ TSMC | Digital Frontiers EP.7
Stage 1 : เริ่มต้น
ออมก่อนใช้แล้วค่อยใช้ส่วนที่เหลือ เป็นหลักที่โค้ชหนุ่มย้ำมาตลอดในการบรรยายครั้งนี้
อ่านข่าวการเงินเพิ่มเติม
https://www.thairath.co.th/money
"""


class TestRuleForUrl:
    """Tests for picking a site rule."""

    def test_known_sites_have_rules(self) -> None:
        """Finnomena, Money Coach and Thairath are recognised."""
        assert rule_for_url("https://www.finnomena.com/dr-niwes/x") is not None
        assert rule_for_url("https://www.moneycoach.co.th/th/blog/y") is not None
        assert rule_for_url("https://www.thairath.co.th/money/z") is not None

    def test_unknown_site_has_no_rule(self) -> None:
        """Other sites (e.g. Berkshire PDFs) are left for manual trimming."""
        assert rule_for_url("https://www.berkshirehathaway.com/letters/2013ltr.pdf") is None


class TestCleanText:
    """Tests for clean_text."""

    def test_finnomena_keeps_only_article_body(self) -> None:
        """Menu, date, view count, tags and related links are removed."""
        cleaned = clean_text(FINNOMENA_PAGE, rule_for_url("https://www.finnomena.com/a"))

        assert cleaned.startswith("ยุคของหุ้น “Value”")
        assert "ในตลาดหุ้นไทย" in cleaned
        assert "แจ้งเตือน" not in cleaned
        assert "1,192" not in cleaned
        assert "บทความที่เกี่ยวข้อง" not in cleaned

    def test_thairath_starts_after_last_tag_and_drops_footer(self) -> None:
        """Tag block, 'read more' footer and bare URLs are removed."""
        cleaned = clean_text(THAIRATH_PAGE, rule_for_url("https://www.thairath.co.th/a"))

        assert cleaned.startswith("Stage 1 : เริ่มต้น")
        assert "ออมก่อนใช้" in cleaned
        assert "#" not in cleaned
        assert "https://" not in cleaned
        assert "Digital Frontiers" not in cleaned

    def test_short_fragments_are_joined(self) -> None:
        """Text split by inline HTML tags is rejoined into one line."""
        page = (
            "1,000\n“ขายระเบิด”\nและขยายสาขาอย่างรวดเร็วจนกลายเป็นร้านที่คนต่อคิวยาวทุกวัน\nแท็ก:\n"
        )

        cleaned = clean_text(page, rule_for_url("https://www.finnomena.com/a"))

        assert cleaned == "“ขายระเบิด” และขยายสาขาอย่างรวดเร็วจนกลายเป็นร้านที่คนต่อคิวยาวทุกวัน"

    def test_missing_markers_keep_whole_text(self) -> None:
        """If a site changes its layout, nothing is silently dropped."""
        cleaned = clean_text(
            "บทความยาวพอสมควรเกี่ยวกับการออมเงินและการลงทุนระยะยาวสำหรับคนทำงาน",
            rule_for_url("https://www.finnomena.com/a"),
        )

        assert cleaned.startswith("บทความยาว")


class TestCleanRawFile:
    """Tests for clean_raw_file."""

    def test_writes_cleaned_copy_with_front_matter(self, tmp_path: Path) -> None:
        """The copy keeps guru_id/source_url, marks cleaned, and leaves the raw file alone."""
        raw_dir = tmp_path / "raw" / "niwes"
        raw_dir.mkdir(parents=True)
        raw_file = raw_dir / "a.md"
        raw_text = (
            "---\nguru_id: niwes\nsource_title: T\nsource_url: https://www.finnomena.com/a\n"
            "kind: article\nreviewed: false\n---\n# T\n\n" + FINNOMENA_PAGE
        )
        raw_file.write_text(raw_text, encoding="utf-8")

        output = clean_raw_file(raw_file, tmp_path / "clean")

        content = output.read_text(encoding="utf-8")
        assert output == tmp_path / "clean" / "niwes" / "a.md"
        assert "guru_id: niwes" in content and "cleaned: true" in content
        assert "reviewed: false" in content
        assert "แจ้งเตือน" not in content
        assert raw_file.read_text(encoding="utf-8") == raw_text


PDF_LETTER = """Insurance results were fine.
Some Thoughts About Investing
Investment is most intelligent when it is most busi-
nesslike. Put 10% of the cash in short-term government
bonds and 90% in a very low-cost S&P 500 index fund.
17
************
Local and state financial problems are accelerating.
"""


class TestEntryMarkersAndPdf:
    """Per-source markers and PDF unwrapping (used for Berkshire letters)."""

    def _entry(self, **markers: str) -> SourceEntry:
        return SourceEntry(
            guru_id="buffett",
            url="https://www.berkshirehathaway.com/letters/2013ltr.pdf",
            **markers,
        )

    def test_entry_markers_override_site_rules(self) -> None:
        """start_marker/end_marker cut one section, keeping the heading."""
        rule = rule_for_entry(
            self._entry(
                start_marker="^Some Thoughts About Investing", end_marker="^Local and state"
            )
        )

        cleaned = clean_text(PDF_LETTER, rule, is_pdf=True)

        assert cleaned.startswith("Some Thoughts About Investing")
        assert "Insurance results" not in cleaned
        assert "Local and state" not in cleaned

    def test_pdf_lines_are_unwrapped_into_sentences(self) -> None:
        """Hard-wrapped PDF lines, hyphenation, page numbers and separators are fixed."""
        rule = rule_for_entry(self._entry(start_marker="^Some Thoughts", end_marker="^Local"))

        cleaned = clean_text(PDF_LETTER, rule, is_pdf=True)

        assert "most businesslike." in cleaned
        assert (
            "Put 10% of the cash in short-term government bonds and 90% in a very low-cost"
            " S&P 500 index fund." in cleaned
        )
        assert "17" not in cleaned.split()
        assert "****" not in cleaned

    def test_entry_without_markers_uses_site_rule(self) -> None:
        """Entries with no markers fall back to the site rule (or None)."""
        entry = SourceEntry(guru_id="niwes", url="https://www.finnomena.com/dr-niwes/x")

        assert rule_for_entry(entry) == rule_for_url(entry.url)

    def test_markers_found(self) -> None:
        """markers_found reports whether the start marker exists in the text."""
        rule = rule_for_entry(self._entry(start_marker="^Some Thoughts"))
        missing = rule_for_entry(self._entry(start_marker="^Nope"))

        assert markers_found(PDF_LETTER, rule)
        assert not markers_found(PDF_LETTER, missing)
        assert markers_found(PDF_LETTER, None)
