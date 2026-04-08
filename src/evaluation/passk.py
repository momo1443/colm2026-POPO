"""
Pass@k metric computation for code/math generation evaluation.

Implements the unbiased estimator from:
    Chen et al. (2021) "Evaluating Large Language Models Trained on Code"

    Pass@k = 1 - C(n-c, k) / C(n, k)
"""

from typing import List, Dict

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute Pass@k using the unbiased estimator.

    Args:
        n: Total number of samples generated per problem.
        c: Number of correct samples for this problem.
        k: Number of samples to consider.

    Returns:
        Pass@k probability for this problem.
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def compute_pass_at_k_batch(
    results: List[Dict[str, int]],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """
    Compute Pass@k for multiple k values over a batch of problems.

    Args:
        results: List of dicts, each with 'n' (total samples) and
            'c' (correct samples) for one problem.
        k_values: List of k values to compute. Defaults to [1, 5, 10].

    Returns:
        Dict mapping 'pass@{k}' to the average Pass@k across problems.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    pass_at_k_results = {f"pass@{k}": [] for k in k_values}

    for result in results:
        n, c = result["n"], result["c"]
        for k in k_values:
            if k <= n:
                pass_at_k_results[f"pass@{k}"].append(pass_at_k(n, c, k))

    return {
        k: float(np.mean(v)) if v else 0.0
        for k, v in pass_at_k_results.items()
    }
