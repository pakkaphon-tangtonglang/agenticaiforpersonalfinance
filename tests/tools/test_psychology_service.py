"""Tests for psychological cue detection service."""

from decimal import Decimal

from finance_ai.tools.psychology_service import (
    PsychologicalAnalysis,
    PsychologicalCue,
    analyze_psychological_cues,
    _calculate_confidence,
    _detect_biases,
    _detect_sentiment,
)


class TestDetectSentiment:
    """Tests for _detect_sentiment."""

    def test_anxious_sentiment(self) -> None:
        """Detects anxious sentiment from worry keywords."""
        assert _detect_sentiment("กังวลมากเรื่องหนี้") == "anxious"

    def test_frustrated_sentiment(self) -> None:
        """Detects frustrated sentiment from frustration keywords."""
        assert _detect_sentiment("เบื่อมาก ท้อแท้") == "frustrated"

    def test_hopeful_sentiment(self) -> None:
        """Detects hopeful sentiment from positive keywords."""
        assert _detect_sentiment("ตั้งใจจะออมเงิน มีเป้าหมาย") == "hopeful"

    def test_neutral_no_keywords(self) -> None:
        """Returns neutral when no sentiment keywords found."""
        assert _detect_sentiment("ขอคำนวณภาษีหน่อย") == "neutral"

    def test_empty_input(self) -> None:
        """Returns neutral for empty input."""
        assert _detect_sentiment("") == "neutral"

    def test_strongest_sentiment_wins(self) -> None:
        """Returns sentiment with most keyword matches."""
        text = "กังวล เครียด กลัว ไม่สบายใจ ตั้งใจ"
        assert _detect_sentiment(text) == "anxious"


class TestDetectBiases:
    """Tests for _detect_biases."""

    def test_loss_aversion(self) -> None:
        """Detects loss aversion from fear-of-loss keywords."""
        biases = _detect_biases("กลัวขาดทุน ไม่กล้าลงทุน")
        assert len(biases) == 1
        assert biases[0].bias_type == "loss_aversion"
        assert "กลัวขาดทุน" in biases[0].matched_keywords

    def test_present_bias(self) -> None:
        """Detects present bias from impulsive keywords."""
        biases = _detect_biases("อยากได้เดี๋ยวนี้ รอไม่ไหว")
        assert len(biases) == 1
        assert biases[0].bias_type == "present_bias"

    def test_overconfidence(self) -> None:
        """Detects overconfidence from certainty keywords."""
        biases = _detect_biases("ชัวร์ ได้แน่ๆ รวยแน่")
        bias_types = {b.bias_type for b in biases}
        assert "overconfidence" in bias_types

    def test_anchoring(self) -> None:
        """Detects anchoring bias from past-reference keywords."""
        biases = _detect_biases("เมื่อก่อนราคาหุ้นถูกกว่านี้")
        assert len(biases) == 1
        assert biases[0].bias_type == "anchoring"

    def test_trend_chasing(self) -> None:
        """Detects trend chasing from hype keywords."""
        biases = _detect_biases("กำลังฮิต ทุกคนซื้อ")
        assert len(biases) == 1
        assert biases[0].bias_type == "trend_chasing"

    def test_multiple_biases(self) -> None:
        """Detects multiple biases in a single message."""
        text = "กลัวขาดทุน แต่เห็นคนอื่นกำลังฮิตซื้อกัน"
        biases = _detect_biases(text)
        bias_types = {b.bias_type for b in biases}
        assert "loss_aversion" in bias_types
        assert "trend_chasing" in bias_types

    def test_no_biases(self) -> None:
        """Returns empty list when no biases detected."""
        biases = _detect_biases("ขอคำนวณภาษี")
        assert biases == []

    def test_empty_input(self) -> None:
        """Returns empty list for empty input."""
        assert _detect_biases("") == []


class TestCalculateConfidence:
    """Tests for _calculate_confidence."""

    def test_single_match(self) -> None:
        """Calculates confidence for single keyword match."""
        result = _calculate_confidence(1, 7)
        assert result == Decimal("0.14")

    def test_all_match(self) -> None:
        """Returns 1.0 when all keywords match."""
        result = _calculate_confidence(5, 5)
        assert result == Decimal("1.00")

    def test_no_match(self) -> None:
        """Returns 0 when no keywords match."""
        result = _calculate_confidence(0, 5)
        assert result == Decimal("0.00")

    def test_zero_total(self) -> None:
        """Returns 0 when total keywords is zero."""
        result = _calculate_confidence(0, 0)
        assert result == Decimal("0")


class TestAnalyzePsychologicalCues:
    """Tests for analyze_psychological_cues (integration)."""

    def test_returns_analysis_model(self) -> None:
        """Returns PsychologicalAnalysis instance."""
        result = analyze_psychological_cues("กลัวขาดทุนมาก")
        assert isinstance(result, PsychologicalAnalysis)

    def test_anxious_with_loss_aversion(self) -> None:
        """Detects both anxious sentiment and loss aversion."""
        result = analyze_psychological_cues("กังวล กลัวขาดทุนมาก")
        assert result.sentiment == "anxious"
        assert len(result.detected_biases) >= 1
        assert result.detected_biases[0].bias_type == "loss_aversion"

    def test_tone_recommendation_not_empty(self) -> None:
        """Generates tone recommendation when bias detected."""
        result = analyze_psychological_cues("กลัวขาดทุน")
        assert result.tone_recommendation != ""
        assert len(result.tone_recommendation) > 10

    def test_neutral_input(self) -> None:
        """Returns neutral analysis for normal input."""
        result = analyze_psychological_cues("ขอคำนวณภาษี 600000 บาท")
        assert result.sentiment == "neutral"
        assert result.detected_biases == []

    def test_empty_input(self) -> None:
        """Handles empty input gracefully."""
        result = analyze_psychological_cues("")
        assert result.sentiment == "neutral"
        assert result.detected_biases == []
        assert "ไม่พบ" in result.summary

    def test_summary_includes_bias_name(self) -> None:
        """Summary mentions detected bias in Thai."""
        result = analyze_psychological_cues("มั่นใจมาก ได้แน่ๆ")
        assert "มั่นใจเกินไป" in result.summary

    def test_summary_includes_sentiment(self) -> None:
        """Summary mentions detected sentiment in Thai."""
        result = analyze_psychological_cues("เครียดมาก กังวลเรื่องหนี้")
        assert "กังวล" in result.summary

    def test_default_tone_for_neutral(self) -> None:
        """Returns default tone for neutral input."""
        result = analyze_psychological_cues("สวัสดีครับ")
        assert "ตอบตามปกติ" in result.tone_recommendation

    def test_model_dump_serializable(self) -> None:
        """Result can be serialized to dict (for tool output)."""
        result = analyze_psychological_cues("กลัวเสียเงิน เครียด")
        dumped = result.model_dump(mode="json")
        assert isinstance(dumped, dict)
        assert "sentiment" in dumped
        assert "detected_biases" in dumped
        assert "tone_recommendation" in dumped
        assert "summary" in dumped
