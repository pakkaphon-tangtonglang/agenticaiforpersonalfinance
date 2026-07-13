"""Tests for LangGraph psychology tool wrapper."""

from finance_ai.agents.psychology_tools import detect_psychological_cues


class TestDetectPsychologicalCuesTool:
    """Tests for the detect_psychological_cues LangGraph tool."""

    def test_tool_returns_dict(self) -> None:
        """Tool invocation returns a dict."""
        result = detect_psychological_cues.invoke({"user_input": "กลัวขาดทุน"})
        assert isinstance(result, dict)

    def test_tool_has_required_fields(self) -> None:
        """Result contains all expected fields."""
        result = detect_psychological_cues.invoke({"user_input": "กลัวขาดทุน เครียดมาก"})
        assert "sentiment" in result
        assert "detected_biases" in result
        assert "tone_recommendation" in result
        assert "summary" in result

    def test_tool_detects_bias(self) -> None:
        """Tool detects loss aversion bias."""
        result = detect_psychological_cues.invoke({"user_input": "กลัวขาดทุน ไม่กล้าลงทุน"})
        assert len(result["detected_biases"]) > 0
        assert result["detected_biases"][0]["bias_type"] == "loss_aversion"

    def test_tool_neutral_input(self) -> None:
        """Tool handles neutral input."""
        result = detect_psychological_cues.invoke({"user_input": "คำนวณภาษี"})
        assert result["sentiment"] == "neutral"
        assert result["detected_biases"] == []

    def test_tool_empty_input(self) -> None:
        """Tool handles empty input."""
        result = detect_psychological_cues.invoke({"user_input": ""})
        assert result["sentiment"] == "neutral"

    def test_tool_sentiment_detection(self) -> None:
        """Tool detects anxious sentiment."""
        result = detect_psychological_cues.invoke({"user_input": "กังวลมาก เครียดเรื่องหนี้"})
        assert result["sentiment"] == "anxious"

    def test_tool_multiple_biases(self) -> None:
        """Tool detects multiple biases."""
        result = detect_psychological_cues.invoke({"user_input": "กลัวขาดทุน แต่เห็นกำลังฮิตก็อยากซื้อ"})
        bias_types = {b["bias_type"] for b in result["detected_biases"]}
        assert len(bias_types) >= 2
