"""Psychological cue detection service.

Rule-based analysis of user messages to detect behavioral biases
and sentiment, based on the BGRC framework (arXiv:2509.14180).
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel

from finance_ai.tools.psychology_constants import (
    BIAS_KEYWORDS,
    SENTIMENT_KEYWORDS,
    TONE_RECOMMENDATIONS,
)


class PsychologicalCue(BaseModel):
    """A single detected behavioral bias.

    Attributes:
        bias_type: Type of bias (e.g., 'loss_aversion').
        confidence: Detection confidence (0.0 - 1.0).
        matched_keywords: Keywords that triggered this detection.

    Example:
        >>> cue = PsychologicalCue(
        ...     bias_type="loss_aversion",
        ...     confidence=Decimal("0.6"),
        ...     matched_keywords=["กลัวขาดทุน"],
        ... )
    """

    bias_type: str
    confidence: Decimal
    matched_keywords: list[str]


class PsychologicalAnalysis(BaseModel):
    """Result of psychological cue analysis.

    Attributes:
        sentiment: Detected sentiment.
        detected_biases: List of detected biases.
        tone_recommendation: Suggested tone for response.
        summary: Brief summary in Thai.

    Example:
        >>> analysis = analyze_psychological_cues("กลัวขาดทุนมาก")
        >>> analysis.sentiment
        'anxious'
    """

    sentiment: str
    detected_biases: list[PsychologicalCue]
    tone_recommendation: str
    summary: str


def analyze_psychological_cues(
    user_input: str,
) -> PsychologicalAnalysis:
    """Analyze user message for sentiment and behavioral biases.

    Args:
        user_input: The user's message text.

    Returns:
        PsychologicalAnalysis with detected cues and recommendations.

    Example:
        >>> result = analyze_psychological_cues("กลัวเสียเงิน")
        >>> result.detected_biases[0].bias_type
        'loss_aversion'
    """
    sentiment = _detect_sentiment(user_input)
    biases = _detect_biases(user_input)
    tone = _build_tone_recommendation(sentiment, biases)
    summary = _build_summary(sentiment, biases)
    return PsychologicalAnalysis(
        sentiment=sentiment,
        detected_biases=biases,
        tone_recommendation=tone,
        summary=summary,
    )


def _detect_sentiment(text: str) -> str:
    """Detect sentiment from user text using keyword matching.

    Args:
        text: User message text.

    Returns:
        Sentiment label: 'neutral', 'anxious', 'frustrated', or 'hopeful'.
    """
    if not text:
        return "neutral"
    scores: dict[str, int] = {}
    for sentiment, keywords in SENTIMENT_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text)
        if count > 0:
            scores[sentiment] = count
    if not scores:
        return "neutral"
    return max(scores, key=scores.get)  # type: ignore[arg-type]


def _detect_biases(text: str) -> list[PsychologicalCue]:
    """Detect behavioral biases from user text.

    Args:
        text: User message text.

    Returns:
        List of detected PsychologicalCue objects.
    """
    if not text:
        return []
    detected: list[PsychologicalCue] = []
    for bias_type, keywords in BIAS_KEYWORDS.items():
        matched = [kw for kw in keywords if kw in text]
        if matched:
            confidence = _calculate_confidence(len(matched), len(keywords))
            detected.append(
                PsychologicalCue(
                    bias_type=bias_type,
                    confidence=confidence,
                    matched_keywords=matched,
                )
            )
    return detected


def _calculate_confidence(
    matched_count: int,
    total_keywords: int,
) -> Decimal:
    """Calculate confidence score based on keyword match ratio.

    Args:
        matched_count: Number of matched keywords.
        total_keywords: Total keywords for this bias type.

    Returns:
        Confidence as Decimal between 0.0 and 1.0.
    """
    if total_keywords == 0:
        return Decimal("0")
    raw = Decimal(str(matched_count)) / Decimal(str(total_keywords))
    return min(raw, Decimal("1.0")).quantize(Decimal("0.01"))


def _build_tone_recommendation(
    sentiment: str,
    biases: list[PsychologicalCue],
) -> str:
    """Build a combined tone recommendation string.

    Args:
        sentiment: Detected sentiment.
        biases: List of detected biases.

    Returns:
        Combined tone recommendation in Thai.
    """
    parts: list[str] = []
    if sentiment in TONE_RECOMMENDATIONS:
        parts.append(TONE_RECOMMENDATIONS[sentiment])
    for cue in biases:
        if cue.bias_type in TONE_RECOMMENDATIONS:
            parts.append(TONE_RECOMMENDATIONS[cue.bias_type])
    if not parts:
        return "ตอบตามปกติ ให้ข้อมูลที่ถูกต้องและครบถ้วน"
    return " | ".join(parts)


def _build_summary(
    sentiment: str,
    biases: list[PsychologicalCue],
) -> str:
    """Build a brief Thai summary of the analysis.

    Args:
        sentiment: Detected sentiment.
        biases: List of detected biases.

    Returns:
        Summary string in Thai.
    """
    if not biases and sentiment == "neutral":
        return "ไม่พบ bias หรือสัญญาณทางจิตวิทยาที่ชัดเจน"
    parts: list[str] = []
    if sentiment != "neutral":
        sentiment_thai = {
            "anxious": "กังวล/วิตกกังวล",
            "frustrated": "หงุดหงิด/ท้อแท้",
            "hopeful": "มีความหวัง/ตั้งใจ",
        }
        parts.append(f"อารมณ์: {sentiment_thai.get(sentiment, sentiment)}")
    for cue in biases:
        bias_thai = {
            "loss_aversion": "กลัวขาดทุน",
            "present_bias": "ชอบปัจจุบัน",
            "overconfidence": "มั่นใจเกินไป",
            "anchoring": "ยึดติดราคาเดิม",
            "trend_chasing": "ตามกระแส",
        }
        label = bias_thai.get(cue.bias_type, cue.bias_type)
        parts.append(f"bias: {label} ({cue.confidence})")
    return " | ".join(parts)
