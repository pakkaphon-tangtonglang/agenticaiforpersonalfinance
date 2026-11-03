"""Small, dependency-free statistics for the architecture study.

- Wilson score interval for success rates.
- Exact two-sided binomial test, used as McNemar's test (paired success)
  and as the sign test (paired latency).
- Percentiles and a bootstrap interval for the median paired difference.
- pass^k: share of tasks solved in every one of k rounds (tau-bench).
"""

import math
import random
from collections import defaultdict

Z_95 = 1.959964


def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Return the Wilson score interval for a proportion.

    Args:
        successes: Number of successes.
        total: Number of trials.
        z: Normal quantile (1.96 for 95 %).

    Returns:
        (low, high) in [0, 1]; (0, 0) when total is 0.

    Example:
        >>> low, high = wilson_interval(8, 10)
        >>> round(low, 3), round(high, 3)
        (0.49, 0.943)
    """
    if total == 0:
        return (0.0, 0.0)
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def exact_binomial_p(wins: int, losses: int) -> float:
    """Two-sided exact binomial test of wins vs losses with p = 0.5.

    Used as McNemar's exact test (discordant pairs) and as the sign test.

    Args:
        wins: Pairs where the first condition was better.
        losses: Pairs where the second condition was better.

    Returns:
        p-value in [0, 1]; 1.0 when there are no informative pairs.

    Example:
        >>> round(exact_binomial_p(9, 1), 4)
        0.0215
    """
    total = wins + losses
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(min(wins, losses) + 1)) / 2**total
    return float(min(1.0, 2 * tail))


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile.

    Args:
        values: Sample values (any order).
        fraction: Percentile as a fraction, e.g. 0.95.

    Returns:
        The percentile; 0.0 for an empty sample.

    Example:
        >>> percentile([1.0, 2.0, 3.0, 4.0], 0.5)
        2.5
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_median_interval(
    differences: list[float], resamples: int = 2000, seed: int = 7
) -> tuple[float, float]:
    """Return a 95 % percentile-bootstrap interval for the median.

    Args:
        differences: Paired differences.
        resamples: Bootstrap resamples.
        seed: Random seed (repeatable output).

    Returns:
        (low, high); (0, 0) for an empty sample.

    Example:
        >>> low, high = bootstrap_median_interval([1.0, 2.0, 3.0])
        >>> 1.0 <= low <= high <= 3.0
        True
    """
    if not differences:
        return (0.0, 0.0)
    rng = random.Random(seed)
    medians = [
        percentile(rng.choices(differences, k=len(differences)), 0.5) for _ in range(resamples)
    ]
    return (percentile(medians, 0.025), percentile(medians, 0.975))


def pass_hat_k(outcomes: list[tuple[str, bool]], rounds: int) -> float:
    """Return pass^k: the share of tasks that succeeded in all k rounds.

    Tasks with fewer than k recorded rounds are left out.

    Args:
        outcomes: (task_id, success) per run.
        rounds: k, the number of rounds.

    Returns:
        Share in [0, 1]; 0.0 when no task has k rounds.

    Example:
        >>> pass_hat_k([("a", True), ("a", True), ("b", True), ("b", False)], 2)
        0.5
    """
    by_task: dict[str, list[bool]] = defaultdict(list)
    for task_id, success in outcomes:
        by_task[task_id].append(success)
    complete = [results for results in by_task.values() if len(results) >= rounds]
    if not complete:
        return 0.0
    return sum(all(results[:rounds]) for results in complete) / len(complete)
