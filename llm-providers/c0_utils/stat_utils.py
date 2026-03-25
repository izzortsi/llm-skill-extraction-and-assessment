"""
stat_utils.py

Pure-Python statistical utilities for benchmark analysis.
Bootstrap confidence intervals and permutation tests.
"""

from __future__ import annotations

import random
import math
from typing import List, Tuple


def mean(values: List[float]) -> float:
    """Compute arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean.

    Args:
        values: Sample values
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed for reproducibility

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    if not values:
        return (0.0, 0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0], values[0])

    rng = random.Random(seed)
    n = len(values)
    point_estimate = mean(values)

    # Generate bootstrap means
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        bootstrap_means.append(mean(sample))

    bootstrap_means.sort()

    # Compute percentile CI
    alpha = 1.0 - confidence
    lower_idx = int(math.floor((alpha / 2.0) * n_bootstrap))
    upper_idx = int(math.ceil((1.0 - alpha / 2.0) * n_bootstrap)) - 1

    lower_idx = max(0, min(lower_idx, n_bootstrap - 1))
    upper_idx = max(0, min(upper_idx, n_bootstrap - 1))

    return (point_estimate, bootstrap_means[lower_idx], bootstrap_means[upper_idx])


def permutation_test(
    group_a: List[float],
    group_b: List[float],
    n_permutations: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Two-sided permutation test for difference in means.

    Args:
        group_a: First group of values
        group_b: Second group of values
        n_permutations: Number of permutations
        seed: Random seed for reproducibility

    Returns:
        (observed_difference, p_value)
        where observed_difference = mean(group_a) - mean(group_b)
    """
    if not group_a or not group_b:
        return (0.0, 1.0)

    rng = random.Random(seed)

    observed_diff = mean(group_a) - mean(group_b)
    abs_observed = abs(observed_diff)

    combined = group_a + group_b
    n_a = len(group_a)
    n_total = len(combined)

    count_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        perm_diff = abs(mean(perm_a) - mean(perm_b))
        if perm_diff >= abs_observed:
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)
    return (observed_diff, p_value)


def pass_rate(outcomes: List[bool]) -> float:
    """Compute pass rate from boolean outcomes.

    Args:
        outcomes: List of pass/fail booleans

    Returns:
        Pass rate as a float in [0, 1]
    """
    if not outcomes:
        return 0.0
    return sum(1.0 for o in outcomes if o) / len(outcomes)


def pass_rate_delta_pp(
    baseline_outcomes: List[bool],
    treatment_outcomes: List[bool],
) -> float:
    """Compute pass rate delta in percentage points.

    Args:
        baseline_outcomes: Baseline pass/fail outcomes
        treatment_outcomes: Treatment pass/fail outcomes

    Returns:
        Delta in percentage points (treatment - baseline) * 100
    """
    baseline_rate = pass_rate(baseline_outcomes)
    treatment_rate = pass_rate(treatment_outcomes)
    return (treatment_rate - baseline_rate) * 100.0
