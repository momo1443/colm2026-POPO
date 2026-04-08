"""Unit tests for POPO weight computation."""

import pytest
import torch

from src.popo.weights import compute_popo_weights, compute_sequence_log_probs


class TestComputeSequenceLogProbs:
    """Tests for sequence-level log probability computation."""

    def test_basic_computation(self):
        """Test mean log-prob computation."""
        logps = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.ones(2, 3)
        result = compute_sequence_log_probs(logps, mask)
        assert result.shape == (2,)
        assert torch.allclose(result, torch.tensor([2.0, 5.0]))

    def test_with_padding(self):
        """Test that padding tokens are excluded."""
        logps = torch.tensor([[1.0, 2.0, 999.0], [4.0, 5.0, 6.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        result = compute_sequence_log_probs(logps, mask)
        expected = torch.tensor([1.5, 5.0])
        assert torch.allclose(result, expected)


class TestComputePopoWeights:
    """Tests for POPO importance weight computation."""

    def test_single_prompt_mixed(self):
        """Test weights with one prompt, mixed positive/negative."""
        logps = torch.tensor([
            [-0.5, -0.3],  # sample 0 (positive)
            [-0.8, -0.9],  # sample 1 (negative)
            [-0.3, -0.2],  # sample 2 (positive)
            [-1.0, -1.2],  # sample 3 (negative)
        ])
        mask = torch.ones(4, 2)
        pos_mask = torch.tensor([True, False, True, False])

        weights = compute_popo_weights(logps, mask, pos_mask, num_generations=4)

        assert weights.shape == (4,)
        # Negative samples should have zero weight
        assert weights[1].item() == 0.0
        assert weights[3].item() == 0.0
        # Positive weights should sum to 1
        assert abs(weights[pos_mask].sum().item() - 1.0) < 1e-5

    def test_all_positive(self):
        """Test when all samples are positive."""
        logps = torch.randn(4, 8)
        mask = torch.ones(4, 8)
        pos_mask = torch.ones(4, dtype=torch.bool)

        weights = compute_popo_weights(logps, mask, pos_mask, num_generations=4)
        assert abs(weights.sum().item() - 1.0) < 1e-5

    def test_all_negative(self):
        """Test when all samples are negative (edge case)."""
        logps = torch.randn(4, 8)
        mask = torch.ones(4, 8)
        pos_mask = torch.zeros(4, dtype=torch.bool)

        weights = compute_popo_weights(logps, mask, pos_mask, num_generations=4)
        assert weights.sum().item() == 0.0

    def test_multi_prompt(self):
        """Test with multiple prompts (2 prompts, 4 generations each)."""
        logps = torch.randn(8, 16)
        mask = torch.ones(8, 16)
        pos_mask = torch.tensor(
            [True, False, True, True, False, True, False, False]
        )

        weights = compute_popo_weights(logps, mask, pos_mask, num_generations=4)

        assert weights.shape == (8,)
        # Check per-prompt normalization: weights sum to 1 per prompt
        w_prompt1 = weights[:4]
        w_prompt2 = weights[4:]
        assert abs(w_prompt1[pos_mask[:4]].sum().item() - 1.0) < 1e-5
        assert abs(w_prompt2[pos_mask[4:]].sum().item() - 1.0) < 1e-5

    def test_no_nan(self):
        """Test that all-negative groups produce zeros, not NaNs."""
        logps = torch.randn(4, 8)
        mask = torch.ones(4, 8)
        pos_mask = torch.zeros(4, dtype=torch.bool)

        weights = compute_popo_weights(logps, mask, pos_mask, num_generations=4)
        assert not torch.isnan(weights).any()

    def test_uniform_mode(self):
        """Test uniform weights give equal weight to all positives in a group."""
        logps = torch.randn(4, 8)
        mask = torch.ones(4, 8)
        pos_mask = torch.tensor([True, False, True, True])

        weights = compute_popo_weights(
            logps, mask, pos_mask, num_generations=4, weight_mode="uniform",
        )
        assert weights[1].item() == 0.0
        pos_weights = weights[pos_mask]
        assert torch.allclose(pos_weights, pos_weights[0].expand_as(pos_weights)), (
            "Uniform mode should give equal weight to all positives"
        )
        assert abs(pos_weights.sum().item() - 1.0) < 1e-5

    def test_uniform_mode_multi_prompt(self):
        """Test uniform mode normalizes independently per prompt group."""
        logps = torch.randn(8, 4)
        mask = torch.ones(8, 4)
        pos_mask = torch.tensor([True, True, False, False, True, False, True, True])

        weights = compute_popo_weights(
            logps, mask, pos_mask, num_generations=4, weight_mode="uniform",
        )
        # Prompt 0: 2 positives -> each gets 0.5
        assert abs(weights[0].item() - 0.5) < 1e-5
        assert abs(weights[1].item() - 0.5) < 1e-5
        assert weights[2].item() == 0.0
        assert weights[3].item() == 0.0
        # Prompt 1: 3 positives -> each gets 1/3
        expected = 1.0 / 3.0
        assert abs(weights[4].item() - expected) < 1e-5
        assert weights[5].item() == 0.0
        assert abs(weights[6].item() - expected) < 1e-5
        assert abs(weights[7].item() - expected) < 1e-5

    def test_length_normalize(self):
        """Test length normalization penalizes longer responses."""
        logps = torch.tensor([
            [-0.5, -0.3, 0.0, 0.0],
            [-0.8, -0.9, -0.7, -0.6],
        ])
        # Sample 0: 2 tokens, sample 1: 4 tokens
        mask = torch.tensor([
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ])
        pos_mask = torch.tensor([True, True])

        weights = compute_popo_weights(
            logps, mask, pos_mask, num_generations=2,
            weight_mode="uniform", length_normalize=True,
        )
        # Shorter response (sample 0) should get higher weight
        assert weights[0].item() > weights[1].item(), (
            "Shorter response should get higher weight with length normalization"
        )
        assert abs(weights.sum().item() - 1.0) < 1e-5

    def test_length_normalize_with_uniform(self):
        """Test combined uniform + length normalization."""
        logps = torch.randn(4, 8)
        mask = torch.tensor([
            [1.0] * 2 + [0.0] * 6,
            [1.0] * 8,
            [1.0] * 4 + [0.0] * 4,
            [1.0] * 6 + [0.0] * 2,
        ])
        pos_mask = torch.tensor([True, True, True, False])

        weights = compute_popo_weights(
            logps, mask, pos_mask, num_generations=4,
            weight_mode="uniform", length_normalize=True,
        )
        assert weights[3].item() == 0.0
        assert abs(weights[pos_mask].sum().item() - 1.0) < 1e-5
        # Shortest response (sample 0, 2 tokens) should get highest weight
        assert weights[0].item() > weights[1].item()
        assert weights[0].item() > weights[2].item()

    def test_invalid_weight_mode(self):
        """Test that an unknown weight_mode raises ValueError."""
        logps = torch.randn(4, 8)
        mask = torch.ones(4, 8)
        pos_mask = torch.ones(4, dtype=torch.bool)

        with pytest.raises(ValueError, match="weight_mode must be"):
            compute_popo_weights(
                logps, mask, pos_mask, num_generations=4,
                weight_mode="invalid",
            )
