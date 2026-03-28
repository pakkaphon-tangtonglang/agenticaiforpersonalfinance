"""Pure metric computation functions for evaluation framework.

All functions are stateless, under 20 lines, and independently testable.
"""

import re
from decimal import Decimal


def compute_precision_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int,
) -> float:
    """Compute Precision@k for retrieval results.

    Args:
        retrieved: List of retrieved item identifiers.
        relevant: List of relevant (ground truth) item identifiers.
        k: Number of top results to consider.

    Returns:
        Precision score between 0.0 and 1.0.

    Example:
        >>> compute_precision_at_k(["a", "b", "c"], ["a", "d"], 3)
        0.3333333333333333
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / k


def compute_recall_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int,
) -> float:
    """Compute Recall@k for retrieval results.

    Args:
        retrieved: List of retrieved item identifiers.
        relevant: List of relevant (ground truth) item identifiers.
        k: Number of top results to consider.

    Returns:
        Recall score between 0.0 and 1.0.

    Example:
        >>> compute_recall_at_k(["a", "b"], ["a", "c"], 2)
        0.5
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / len(relevant)


def compute_mrr(retrieved: list[str], relevant: list[str]) -> float:
    """Compute Mean Reciprocal Rank for a single query.

    Args:
        retrieved: List of retrieved item identifiers (ranked).
        relevant: List of relevant item identifiers.

    Returns:
        Reciprocal rank (1/position) of the first relevant result, or 0.0.

    Example:
        >>> compute_mrr(["x", "a", "b"], ["a"])
        0.5
    """
    relevant_set = set(relevant)
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_keyword_hit_rate(text: str, keywords: list[str]) -> float:
    """Compute fraction of expected keywords found in text.

    Args:
        text: Text content to search in.
        keywords: List of keywords to look for.

    Returns:
        Fraction of keywords found (0.0 to 1.0).

    Example:
        >>> compute_keyword_hit_rate("ลดหย่อน 60,000 บาท", ["60,000", "ลดหย่อน"])
        1.0
    """
    if not keywords:
        return 1.0
    hits = sum(1 for kw in keywords if kw in text)
    return hits / len(keywords)


def compute_confusion_matrix(
    pairs: list[tuple[str, str]],
) -> dict[str, dict[str, int]]:
    """Build a confusion matrix from (expected, predicted) pairs.

    Args:
        pairs: List of (expected_label, predicted_label) tuples.

    Returns:
        Nested dict: matrix[expected][predicted] = count.

    Example:
        >>> compute_confusion_matrix([("a", "a"), ("a", "b")])
        {'a': {'a': 1, 'b': 1}}
    """
    matrix: dict[str, dict[str, int]] = {}
    for expected, predicted in pairs:
        if expected not in matrix:
            matrix[expected] = {}
        matrix[expected][predicted] = matrix[expected].get(predicted, 0) + 1
    return matrix


def compute_per_class_accuracy(
    matrix: dict[str, dict[str, int]],
) -> dict[str, Decimal]:
    """Compute per-class accuracy from a confusion matrix.

    Args:
        matrix: Confusion matrix from compute_confusion_matrix.

    Returns:
        Dict mapping each class to its accuracy (0.0 to 1.0).

    Example:
        >>> compute_per_class_accuracy({"a": {"a": 9, "b": 1}})
        {'a': Decimal('0.9000')}
    """
    result: dict[str, Decimal] = {}
    for label, predictions in matrix.items():
        total = sum(predictions.values())
        correct = predictions.get(label, 0)
        result[label] = (Decimal(correct) / Decimal(total)).quantize(Decimal("0.0001"))
    return result


def compute_f1_score(precision: Decimal, recall: Decimal) -> Decimal:
    """Compute F1 score from precision and recall.

    Args:
        precision: Precision value (0.0 to 1.0).
        recall: Recall value (0.0 to 1.0).

    Returns:
        F1 score (harmonic mean of precision and recall).

    Example:
        >>> compute_f1_score(Decimal("0.8"), Decimal("0.6"))
        Decimal('0.6857')
    """
    total = precision + recall
    if total == Decimal("0"):
        return Decimal("0.0000")
    return (Decimal("2") * precision * recall / total).quantize(Decimal("0.0001"))


def compute_percentile(values: list[float], percentile: int) -> float:
    """Compute a percentile value from a list of floats.

    Args:
        values: List of numeric values.
        percentile: Percentile to compute (0-100).

    Returns:
        The percentile value.

    Example:
        >>> compute_percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50)
        3.0
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    index = (percentile / 100) * (len(sorted_vals) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_vals) - 1)
    fraction = index - lower
    return sorted_vals[lower] + fraction * (sorted_vals[upper] - sorted_vals[lower])


def compute_mean_decimal(values: list[Decimal]) -> Decimal:
    """Compute mean of a list of Decimal values.

    Args:
        values: List of Decimal values.

    Returns:
        Mean value rounded to 4 decimal places.

    Example:
        >>> compute_mean_decimal([Decimal("1"), Decimal("2"), Decimal("3")])
        Decimal('2.0000')
    """
    if not values:
        return Decimal("0.0000")
    total = sum(values, Decimal("0"))
    return (total / Decimal(len(values))).quantize(Decimal("0.0001"))


def extract_thai_number(text: str, pattern: str) -> Decimal | None:
    """Extract a Thai-formatted number from text using a regex pattern.

    The pattern should contain one capture group for the number.
    Handles comma-separated Thai number formats (e.g., "29,000.50").

    Args:
        text: Text to search in.
        pattern: Regex pattern with one capture group for the number.

    Returns:
        Extracted number as Decimal, or None if not found.

    Example:
        >>> extract_thai_number("ภาษี 29,000 บาท", r"ภาษี\\s+([\\d,]+\\.?\\d*)\\s*บาท")
        Decimal('29000')
    """
    match = re.search(pattern, text)
    if not match:
        return None
    number_str = match.group(1).replace(",", "")
    try:
        return Decimal(number_str)
    except (ArithmeticError, ValueError):
        return None
