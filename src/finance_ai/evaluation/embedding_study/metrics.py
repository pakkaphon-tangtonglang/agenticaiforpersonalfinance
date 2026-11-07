"""Ranking metrics (binary relevance) and a lexical leakage check."""

import math


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """1.0 when any relevant chunk is in the top k (hit rate), else 0.0.

    Example:
        >>> recall_at_k(["a", "b"], {"b"}, 1), recall_at_k(["a", "b"], {"b"}, 2)
        (0.0, 1.0)
    """
    return 1.0 if relevant.intersection(ranked[:k]) else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str], cutoff: int = 10) -> float:
    """1 / rank of the first relevant chunk within the cutoff, else 0.

    Example:
        >>> reciprocal_rank(["a", "b", "c"], {"c"})
        0.3333333333333333
    """
    for rank, chunk_id in enumerate(ranked[:cutoff], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    """Normalised discounted cumulative gain with binary relevance.

    Example:
        >>> ndcg_at_k(["a", "b"], {"a"})
        1.0
    """
    gain = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(ranked[:k], start=1)
        if chunk_id in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return gain / ideal if ideal else 0.0


def char_ngrams(text: str, size: int = 6) -> set[str]:
    """Character n-grams with spaces removed (works for Thai, which has no word spaces).

    Example:
        >>> sorted(char_ngrams("abcdefg", 6))
        ['abcdef', 'bcdefg']
    """
    compact = "".join(text.split())
    return {compact[i : i + size] for i in range(len(compact) - size + 1)}


def lexical_overlap(query: str, passage: str, size: int = 6) -> float:
    """Share of the query's character n-grams that also appear in a passage.

    A high share means the question copies the document's wording, which
    favours keyword search and inflates scores (leakage).

    Example:
        >>> lexical_overlap("abcdefg", "xxabcdefgxx")
        1.0
    """
    query_grams = char_ngrams(query, size)
    if not query_grams:
        return 0.0
    return len(query_grams & char_ngrams(passage, size)) / len(query_grams)
