"""
Ave@k (Mean@k) metric computation for code/math generation evaluation.

Computes the expected fraction of correct samples in k random draws
(without replacement) from n total samples.

By linearity of expectation in the hypergeometric distribution:

    Ave@k = E[c_k / k] = c / n

where c is the number of correct samples in n total samples.
This holds for all valid k <= n.
"""

from typing import List, Dict

import numpy as np


def ave_at_k(n: int, c: int, k: int) -> float:
    """
    Compute Ave@k — the expected fraction of correct samples in k draws.

    By linearity of expectation (hypergeometric), the unbiased estimator
    is simply c/n, independent of k:

        Ave@k = E[c_k / k] = c / n

    Args:
        n: Total number of samples generated per problem.
        c: Number of correct samples for this problem.
        k: Number of samples to consider (must be <= n).

    Returns:
        Ave@k value for this problem.
    """
    if n == 0:
        return 0.0
    return c / n


def compute_ave_at_k_batch(
    results: List[Dict[str, int]],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """
    Compute Ave@k for multiple k values over a batch of problems.

    Args:
        results: List of dicts, each with 'n' (total samples) and
            'c' (correct samples) for one problem.
        k_values: List of k values to compute. Defaults to [1, 5, 10].

    Returns:
        Dict mapping 'ave@{k}' to the average Ave@k across problems.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    ave_at_k_results = {f"ave@{k}": [] for k in k_values}

    for result in results:
        n, c = result["n"], result["c"]
        for k in k_values:
            if k <= n:
                ave_at_k_results[f"ave@{k}"].append(ave_at_k(n, c, k))

    return {
        k: float(np.mean(v)) if v else 0.0
        for k, v in ave_at_k_results.items()
    }
