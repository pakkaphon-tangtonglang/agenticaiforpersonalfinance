"""Tests for cleaning fetched guru pages down to the article body."""

from pathlib import Path

from finance_ai.tools.guru_raw_cleaner import clean_raw_file, clean_text, rule_for_url

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
