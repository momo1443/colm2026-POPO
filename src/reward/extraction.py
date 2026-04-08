"""
Answer extraction utilities for math reasoning tasks.

Provides regex-based extraction for various answer formats:
- \\boxed{} (LaTeX)
- #### (GSM8K)
- <answer> tags (R1 format)
- Last number fallback

Canonical source: test_environment/train_grpo_v2.py lines 232-277.
"""

import re
from typing import Optional


def extract_answer_simple(text: str) -> str:
    """
    Extract a math answer from model-generated text using regex patterns.

    Tries the following extraction strategies in order:
    1. \\boxed{...} (LaTeX boxed format)
    2. #### ... (GSM8K hash format)
    3. <answer>...</answer> (R1/reasoning format)
    4. Last number in text (fallback)

    Args:
        text: Model-generated completion text.

    Returns:
        Extracted answer string, or empty string if no answer found.
    """
    # 1. Look for boxed answer (handle nested braces)
    boxed_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    match = re.search(boxed_pattern, text)
    if match:
        return match.group(1).strip()

    # 2. Look for #### pattern (GSM8K format)
    match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    # 3. Look for <answer> tags (R1 format)
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        answer_text = match.group(1).strip()
        # Try to extract boxed from within answer tags
        boxed_in_answer = re.search(boxed_pattern, answer_text)
        if boxed_in_answer:
            return boxed_in_answer.group(1).strip()
        return answer_text

    # 4. Last number in text (fallback)
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def extract_answer_from_solution(solution: str) -> str:
    """
    Extract ground truth answer from a solution string.

    Handles:
    - GSM8K format: "... #### 42"
    - LaTeX boxed format: "... \\boxed{42}"
    - Plain text fallback

    Args:
        solution: Ground truth solution string.

    Returns:
        Extracted answer string.
    """
    # GSM8K format: ends with #### answer
    match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", solution)
    if match:
        return match.group(1).replace(",", "")

    # Boxed format
    boxed_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    match = re.search(boxed_pattern, solution)
    if match:
        return match.group(1).strip()

    # Return the whole solution if no pattern found
    return solution.strip()
