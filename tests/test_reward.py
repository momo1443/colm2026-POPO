"""Unit tests for reward and extraction modules."""

import pytest

from src.reward.extraction import extract_answer_simple, extract_answer_from_solution


class TestExtractAnswerSimple:
    """Tests for the answer extraction function."""

    def test_boxed_answer(self):
        """Test extraction from \\boxed{} format."""
        text = r"The answer is \boxed{42}."
        assert extract_answer_simple(text) == "42"

    def test_boxed_nested(self):
        """Test extraction from nested boxed format."""
        text = r"\boxed{\frac{1}{2}}"
        assert extract_answer_simple(text) == r"\frac{1}{2}"

    def test_hash_answer(self):
        """Test extraction from #### format (GSM8K)."""
        text = "So the total is #### 1234"
        assert extract_answer_simple(text) == "1234"

    def test_hash_with_commas(self):
        """Test extraction from #### format with comma separators."""
        text = "The answer is #### 1,234,567"
        assert extract_answer_simple(text) == "1234567"

    def test_answer_tags(self):
        """Test extraction from <answer> tags."""
        text = "Some reasoning <answer>42</answer>"
        assert extract_answer_simple(text) == "42"

    def test_last_number_fallback(self):
        """Test fallback to last number in text."""
        text = "The result is 7 and then 42."
        assert extract_answer_simple(text) == "42"

    def test_empty_text(self):
        """Test with empty string."""
        assert extract_answer_simple("") == ""

    def test_no_answer(self):
        """Test with text containing no numbers or patterns."""
        assert extract_answer_simple("no answer here") == ""

    def test_negative_number(self):
        """Test extraction of negative numbers."""
        text = "The answer is #### -42"
        assert extract_answer_simple(text) == "-42"


class TestExtractAnswerFromSolution:
    """Tests for ground truth extraction."""

    def test_gsm8k_format(self):
        """Test GSM8K solution format."""
        sol = "Step 1... Step 2... #### 42"
        assert extract_answer_from_solution(sol) == "42"

    def test_boxed_format(self):
        """Test boxed solution format."""
        sol = r"Therefore, \boxed{42}"
        assert extract_answer_from_solution(sol) == "42"

    def test_plain_text(self):
        """Test fallback for plain text solution."""
        sol = "42"
        assert extract_answer_from_solution(sol) == "42"
