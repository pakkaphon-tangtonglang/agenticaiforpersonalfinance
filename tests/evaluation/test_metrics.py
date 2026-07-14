"""Tests for evaluation metric functions."""

from decimal import Decimal

import pytest

from finance_ai.evaluation.metrics import (
    compute_confusion_matrix,
    compute_f1_score,
    compute_keyword_hit_rate,
    compute_mean_decimal,
    compute_mrr,
    compute_per_class_accuracy,
    compute_percentile,
    compute_precision_at_k,
    compute_recall_at_k,
    extract_thai_number,
    safe_mean,
)


class TestPrecisionAtK:
    """Tests for compute_precision_at_k."""

    def test_perfect_precision(self) -> None:
        """Test when all retrieved items are relevant."""
        result = compute_precision_at_k(["a", "b"], ["a", "b"], 2)
        assert result == 1.0

    def test_zero_precision(self) -> None:
        """Test when no retrieved items are relevant."""
        result = compute_precision_at_k(["x", "y"], ["a", "b"], 2)
        assert result == 0.0

    def test_partial_precision(self) -> None:
        """Test partial precision (1 of 3 relevant)."""
        result = compute_precision_at_k(["a", "x", "y"], ["a", "b"], 3)
        assert abs(result - 1 / 3) < 1e-10

    def test_k_larger_than_retrieved(self) -> None:
        """Test when k is larger than retrieved list."""
        result = compute_precision_at_k(["a"], ["a", "b"], 3)
        assert abs(result - 1 / 3) < 1e-10

    def test_k_zero_returns_zero(self) -> None:
        """Test that k=0 returns 0."""
        result = compute_precision_at_k(["a"], ["a"], 0)
        assert result == 0.0

    def test_empty_retrieved(self) -> None:
        """Test with empty retrieved list."""
        result = compute_precision_at_k([], ["a"], 3)
        assert result == 0.0


class TestRecallAtK:
    """Tests for compute_recall_at_k."""

    def test_perfect_recall(self) -> None:
        """Test when all relevant items are retrieved."""
        result = compute_recall_at_k(["a", "b"], ["a", "b"], 2)
        assert result == 1.0

    def test_zero_recall(self) -> None:
        """Test when no relevant items are retrieved."""
        result = compute_recall_at_k(["x", "y"], ["a", "b"], 2)
        assert result == 0.0

    def test_partial_recall(self) -> None:
        """Test partial recall (1 of 2 relevant found)."""
        result = compute_recall_at_k(["a", "x"], ["a", "b"], 2)
        assert result == 0.5

    def test_empty_relevant_returns_zero(self) -> None:
        """Test that empty relevant list returns 0."""
        result = compute_recall_at_k(["a"], [], 1)
        assert result == 0.0


class TestMRR:
    """Tests for compute_mrr."""

    def test_first_position(self) -> None:
        """Test when relevant item is at position 1."""
        result = compute_mrr(["a", "b", "c"], ["a"])
        assert result == 1.0

    def test_second_position(self) -> None:
        """Test when relevant item is at position 2."""
        result = compute_mrr(["x", "a", "b"], ["a"])
        assert result == 0.5

    def test_third_position(self) -> None:
        """Test when relevant item is at position 3."""
        result = compute_mrr(["x", "y", "a"], ["a"])
        assert abs(result - 1 / 3) < 1e-10

    def test_no_relevant_found(self) -> None:
        """Test when no relevant item is in retrieved list."""
        result = compute_mrr(["x", "y", "z"], ["a"])
        assert result == 0.0

    def test_empty_retrieved(self) -> None:
        """Test with empty retrieved list."""
        result = compute_mrr([], ["a"])
        assert result == 0.0

    def test_multiple_relevant(self) -> None:
        """Test with multiple relevant items (first match wins)."""
        result = compute_mrr(["x", "a", "b"], ["a", "b"])
        assert result == 0.5


class TestKeywordHitRate:
    """Tests for compute_keyword_hit_rate."""

    def test_all_keywords_found(self) -> None:
        """Test when all keywords are found."""
        result = compute_keyword_hit_rate("ลดหย่อน 60,000 บาท", ["60,000", "ลดหย่อน"])
        assert result == 1.0

    def test_no_keywords_found(self) -> None:
        """Test when no keywords are found."""
        result = compute_keyword_hit_rate("test text", ["ภาษี", "ลดหย่อน"])
        assert result == 0.0

    def test_partial_keywords(self) -> None:
        """Test when some keywords are found."""
        result = compute_keyword_hit_rate("ลดหย่อนส่วนตัว", ["ลดหย่อน", "60,000"])
        assert result == 0.5

    def test_empty_keywords_returns_one(self) -> None:
        """Test that empty keywords returns 1.0."""
        result = compute_keyword_hit_rate("any text", [])
        assert result == 1.0


class TestConfusionMatrix:
    """Tests for compute_confusion_matrix."""

    def test_simple_matrix(self) -> None:
        """Test building a simple confusion matrix."""
        pairs = [("a", "a"), ("a", "b"), ("b", "b")]
        matrix = compute_confusion_matrix(pairs)
        assert matrix["a"]["a"] == 1
        assert matrix["a"]["b"] == 1
        assert matrix["b"]["b"] == 1

    def test_empty_pairs(self) -> None:
        """Test with empty pairs list."""
        matrix = compute_confusion_matrix([])
        assert matrix == {}

    def test_all_correct(self) -> None:
        """Test when all predictions are correct."""
        pairs = [("a", "a"), ("b", "b"), ("c", "c")]
        matrix = compute_confusion_matrix(pairs)
        for label in ("a", "b", "c"):
            assert matrix[label][label] == 1


class TestPerClassAccuracy:
    """Tests for compute_per_class_accuracy."""

    def test_perfect_accuracy(self) -> None:
        """Test with perfect predictions."""
        matrix = {"a": {"a": 10}, "b": {"b": 5}}
        result = compute_per_class_accuracy(matrix)
        assert result["a"] == Decimal("1.0000")
        assert result["b"] == Decimal("1.0000")

    def test_partial_accuracy(self) -> None:
        """Test with some misclassifications."""
        matrix = {"a": {"a": 9, "b": 1}}
        result = compute_per_class_accuracy(matrix)
        assert result["a"] == Decimal("0.9000")

    def test_zero_accuracy(self) -> None:
        """Test with all wrong predictions."""
        matrix = {"a": {"b": 5}}
        result = compute_per_class_accuracy(matrix)
        assert result["a"] == Decimal("0.0000")


class TestF1Score:
    """Tests for compute_f1_score."""

    def test_perfect_f1(self) -> None:
        """Test F1 with perfect precision and recall."""
        result = compute_f1_score(Decimal("1.0"), Decimal("1.0"))
        assert result == Decimal("1.0000")

    def test_zero_f1(self) -> None:
        """Test F1 when both precision and recall are zero."""
        result = compute_f1_score(Decimal("0"), Decimal("0"))
        assert result == Decimal("0.0000")

    def test_typical_f1(self) -> None:
        """Test F1 with typical values."""
        result = compute_f1_score(Decimal("0.8"), Decimal("0.6"))
        assert result == Decimal("0.6857")


class TestPercentile:
    """Tests for compute_percentile."""

    def test_median(self) -> None:
        """Test 50th percentile (median)."""
        result = compute_percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50)
        assert result == 3.0

    def test_minimum(self) -> None:
        """Test 0th percentile (minimum)."""
        result = compute_percentile([1.0, 2.0, 3.0], 0)
        assert result == 1.0

    def test_maximum(self) -> None:
        """Test 100th percentile (maximum)."""
        result = compute_percentile([1.0, 2.0, 3.0], 100)
        assert result == 3.0

    def test_empty_list_returns_zero(self) -> None:
        """Test that empty list returns 0."""
        result = compute_percentile([], 50)
        assert result == 0.0

    def test_single_value(self) -> None:
        """Test with single value."""
        result = compute_percentile([5.0], 50)
        assert result == 5.0

    def test_p95(self) -> None:
        """Test 95th percentile."""
        values = list(range(1, 101))
        result = compute_percentile([float(v) for v in values], 95)
        assert result == pytest.approx(95.05, abs=0.1)


class TestMeanDecimal:
    """Tests for compute_mean_decimal."""

    def test_simple_mean(self) -> None:
        """Test mean of simple values."""
        result = compute_mean_decimal([Decimal("1"), Decimal("2"), Decimal("3")])
        assert result == Decimal("2.0000")

    def test_empty_list_returns_zero(self) -> None:
        """Test that empty list returns zero."""
        result = compute_mean_decimal([])
        assert result == Decimal("0.0000")

    def test_single_value(self) -> None:
        """Test mean of single value."""
        result = compute_mean_decimal([Decimal("4.5")])
        assert result == Decimal("4.5000")


class TestSafeMean:
    """Tests for safe_mean."""

    def test_safe_mean_empty_list(self) -> None:
        """Return 0.0 for empty list."""
        assert safe_mean([]) == 0.0

    def test_safe_mean_normal_values(self) -> None:
        """Return correct mean for normal values."""
        assert safe_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_safe_mean_single_value(self) -> None:
        """Return the value itself for single-element list."""
        assert safe_mean([5.0]) == pytest.approx(5.0)


class TestExtractThaiNumber:
    """Tests for extract_thai_number."""

    def test_simple_number(self) -> None:
        """Test extracting a simple number."""
        result = extract_thai_number("ภาษี 29,000 บาท", r"ภาษี\s+([\d,]+\.?\d*)\s*บาท")
        assert result == Decimal("29000")

    def test_number_with_decimals(self) -> None:
        """Test extracting a decimal number."""
        result = extract_thai_number("อัตรา 7.50%", r"อัตรา\s+([\d.]+)%")
        assert result == Decimal("7.50")

    def test_no_match_returns_none(self) -> None:
        """Test that non-matching text returns None."""
        result = extract_thai_number("no numbers", r"ภาษี\s+([\d,]+)")
        assert result is None

    def test_large_number_with_commas(self) -> None:
        """Test extracting a large comma-separated number."""
        result = extract_thai_number("รวม 1,234,567 บาท", r"รวม\s+([\d,]+)\s*บาท")
        assert result == Decimal("1234567")

    def test_invalid_number_returns_none(self) -> None:
        """Test that invalid number string returns None."""
        result = extract_thai_number("value abc", r"value\s+(\S+)")
        assert result is None
