"""Unit tests for POPOLoss."""

import pytest
import torch

from src.popo.loss import POPOLoss


class TestPOPOLoss:
    """Tests for the POPO loss components."""

    @pytest.fixture
    def loss_fn(self):
        return POPOLoss(alpha=0.1, beta_entropy=0.01, feature_noise_std=0.0)

    @pytest.fixture
    def batch_data(self):
        """Create a simple batch of test data."""
        B, T, D = 4, 16, 64
        return {
            "per_token_logps": torch.randn(B, T, requires_grad=True),
            "completion_mask": torch.ones(B, T),
            "weights": torch.tensor([0.3, 0.0, 0.5, 0.2]),
            "entropies": torch.rand(B, T, requires_grad=True),
            "online_features": torch.randn(B, D, requires_grad=True),
            "target_features": torch.randn(B, D),
        }

    def test_nll_loss_shape(self, loss_fn, batch_data):
        """Test NLL loss returns a scalar."""
        result = loss_fn.nll_loss(
            batch_data["per_token_logps"],
            batch_data["completion_mask"],
            batch_data["weights"],
        )
        assert result.shape == ()

    def test_nll_loss_zero_for_negatives(self, loss_fn):
        """Test NLL is zero when all weights are zero (no positives)."""
        logps = torch.randn(4, 8)
        mask = torch.ones(4, 8)
        weights = torch.zeros(4)
        result = loss_fn.nll_loss(logps, mask, weights)
        assert result.item() == 0.0

    def test_nll_normalization_positive_only(self, loss_fn):
        """Test NLL normalizes by positive token count, not all tokens."""
        # 2 samples, 4 tokens each
        logps = -torch.ones(2, 4)  # all -1.0 (NLL = +1.0)
        mask = torch.ones(2, 4)
        # Only first sample is positive
        weights = torch.tensor([1.0, 0.0])

        result = loss_fn.nll_loss(logps, mask, weights)
        # Expected: weight[0] * sum(1.0 for 4 tokens) / 4 positive tokens = 1.0
        assert abs(result.item() - 1.0) < 1e-5

    def test_similarity_loss_perfect_alignment(self, loss_fn):
        """Test similarity loss is negative when features are aligned."""
        features = torch.randn(4, 64)
        weights = torch.tensor([0.5, 0.0, 0.3, 0.2])
        result = loss_fn.similarity_loss(features, features.clone(), weights)
        assert result.item() < 0, "Aligned features should give negative sim loss"

    def test_similarity_loss_weighted(self, loss_fn):
        """Test that zero-weight samples don't contribute to sim loss."""
        torch.manual_seed(42)
        online = torch.randn(4, 64)
        target = torch.randn(4, 64)
        weights_all = torch.tensor([0.25, 0.25, 0.25, 0.25])
        weights_zero = torch.tensor([0.5, 0.0, 0.5, 0.0])

        loss_all = loss_fn.similarity_loss(online, target, weights_all)
        loss_zero = loss_fn.similarity_loss(online, target, weights_zero)

        assert not torch.allclose(loss_all, loss_zero), (
            f"Weighted and unweighted sim loss should differ: {loss_all} vs {loss_zero}"
        )

    def test_entropy_loss_sign(self, loss_fn):
        """Test entropy loss is negative (we maximize entropy by minimizing loss)."""
        entropies = torch.ones(4, 8)  # constant entropy
        mask = torch.ones(4, 8)
        result = loss_fn.entropy_loss(entropies, mask)
        assert result.item() < 0, "Entropy loss should be negative (we maximize entropy)"

    def test_kl_divergence_zero_for_same(self):
        """Test KL is approximately zero when distributions are identical."""
        logps = torch.randn(4, 16)
        mask = torch.ones(4, 16)
        kl = POPOLoss.compute_kl_divergence(logps, logps, mask)
        assert abs(kl.item()) < 1e-5

    def test_full_loss_gradient_flow(self, batch_data):
        """Test that gradients flow through the full loss computation."""
        loss_fn = POPOLoss(alpha=0.1, beta_entropy=0.01, feature_noise_std=0.0)
        result = loss_fn(
            per_token_logps=batch_data["per_token_logps"],
            completion_mask=batch_data["completion_mask"],
            weights=batch_data["weights"],
            entropies=batch_data["entropies"],
            online_features=batch_data["online_features"],
            target_features=batch_data["target_features"],
            training=True,
        )

        result["loss"].backward()

        assert batch_data["per_token_logps"].grad is not None, "logps gradient missing"
        assert batch_data["online_features"].grad is not None, "online features gradient missing"
        assert batch_data["entropies"].grad is not None, "entropies gradient missing"

    def test_full_loss_output_keys(self, loss_fn, batch_data):
        """Test that __call__ returns all expected keys."""
        result = loss_fn(
            per_token_logps=batch_data["per_token_logps"],
            completion_mask=batch_data["completion_mask"],
            weights=batch_data["weights"],
            entropies=batch_data["entropies"],
            online_features=batch_data["online_features"],
            target_features=batch_data["target_features"],
            training=True,
        )
        expected_keys = {"loss", "nll_loss", "sim_loss", "ent_loss", "entropy", "kl_divergence"}
        assert set(result.keys()) == expected_keys

    def test_full_loss_detached_metrics(self, loss_fn, batch_data):
        """Test that monitoring metrics are detached (no gradient)."""
        result = loss_fn(
            per_token_logps=batch_data["per_token_logps"],
            completion_mask=batch_data["completion_mask"],
            weights=batch_data["weights"],
            entropies=batch_data["entropies"],
            online_features=batch_data["online_features"],
            target_features=batch_data["target_features"],
            training=True,
        )
        for key in ["nll_loss", "sim_loss", "ent_loss", "entropy", "kl_divergence"]:
            assert not result[key].requires_grad, f"{key} should be detached"

    def test_similarity_loss_normalization_by_weight_sum(self):
        """Test sim_loss is normalized by weight sum (stable across batch sizes)."""
        loss_fn = POPOLoss(alpha=0.1, beta_entropy=0.01, feature_noise_std=0.0)
        torch.manual_seed(0)
        D = 64

        online = torch.randn(4, D)
        target = torch.randn(4, D)
        weights = torch.tensor([0.5, 0.0, 0.3, 0.2])
        loss = loss_fn.similarity_loss(online, target, weights)

        assert loss.shape == ()
        assert not torch.isnan(loss)

        # Zero-weight samples should not contribute
        weights_zero = torch.zeros(4)
        loss_zero = loss_fn.similarity_loss(online, target, weights_zero)
        assert abs(loss_zero.item()) < 1e-6
