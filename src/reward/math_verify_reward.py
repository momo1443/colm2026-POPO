"""
Math verification reward function using the math-verify package.

Provides binary reward (1.0 / 0.0) for math reasoning tasks by
comparing model-generated answers against ground truth solutions
using symbolic math verification.

Note: math-verify is a hard dependency (always required).
No fallback to regex-only verification.

Canonical source: test_environment/train_grpo_v2.py lines 283-331.
"""

from typing import List

from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig


# Extraction configs for model predictions (more lenient, prioritize boxed)
_PRED_EXTRACTION_CONFIGS = [
    LatexExtractionConfig(boxed_match_priority=0),
    ExprExtractionConfig(),
]

# Extraction configs for gold answers (stricter)
_GOLD_EXTRACTION_CONFIGS = [
    LatexExtractionConfig(),
    ExprExtractionConfig(),
]


def math_verify_reward(
    completions: List[str],
    solution: List[str],
    **kwargs,
) -> List[float]:
    """
    Compute binary reward using math-verify symbolic verification.

    For each (completion, solution) pair:
    1. Parse both using math-verify's symbolic parser
    2. Verify symbolic equivalence
    3. Return 1.0 if correct, 0.0 if incorrect

    This is the TRL-compatible reward function signature, accepting
    completions and solutions as parallel lists.

    Args:
        completions: List of model-generated completion strings.
        solution: List of ground truth solution strings.
        **kwargs: Additional keyword arguments (ignored, for TRL compatibility).

    Returns:
        List of float rewards (1.0 for correct, 0.0 for incorrect).
    """
    rewards = []

    for completion, sol in zip(completions, solution):
        reward = _verify_single(completion, sol)
        rewards.append(reward)

    return rewards


def _verify_single(completion: str, solution: str) -> float:
    """
    Verify a single completion against a solution.

    Uses math-verify for symbolic comparison. Returns 0.0 if
    parsing fails or answer is incorrect.

    Args:
        completion: Model-generated text.
        solution: Ground truth solution.

    Returns:
        1.0 if correct, 0.0 otherwise.
    """
    try:
        pred_parsed = parse(completion, extraction_config=_PRED_EXTRACTION_CONFIGS)
        gold_parsed = parse(solution, extraction_config=_GOLD_EXTRACTION_CONFIGS)

        if pred_parsed is not None and gold_parsed is not None:
            is_correct = verify(gold_parsed, pred_parsed)
            return 1.0 if is_correct else 0.0
        else:
            return 0.0
    except Exception:
        return 0.0
