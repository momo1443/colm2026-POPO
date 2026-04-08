"""Unit tests for EMATargetPolicy."""

import pytest
import torch
import torch.nn as nn

from src.popo.ema import EMATargetPolicy


class SimpleModel(nn.Module):
    """Simple model for testing EMA."""

    def __init__(self, dim=16):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class TestEMATargetPolicy:
    """Tests for the EMA target policy manager."""

    def test_initialization(self):
        """Test that EMA target is a deep copy with no gradients."""
        model = SimpleModel(16)
        ema = EMATargetPolicy(model, tau=0.99)

        # Target should have no gradients
        for p in ema.target_model.parameters():
            assert not p.requires_grad

        # Target should be in eval mode
        assert not ema.target_model.training

    def test_deep_copy(self):
        """Test that target is independent from online model."""
        model = SimpleModel(16)
        ema = EMATargetPolicy(model, tau=0.99)

        # Modify online model
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(0.0)

        # Target should not be affected
        target_sum = sum(p.sum().item() for p in ema.target_model.parameters())
        assert target_sum != 0.0, "Target should be independent from online model"

    def test_ema_update(self):
        """Test that EMA update follows: xi <- tau * xi + (1-tau) * theta."""
        model = SimpleModel(16)
        tau = 0.9
        ema = EMATargetPolicy(model, tau=tau)

        # Store initial target params
        old_target = {
            name: p.clone() for name, p in ema.target_model.named_parameters()
        }

        # Modify online model
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(1.0)

        # Perform EMA update
        ema.update(model)

        # Verify: new_target = tau * old_target + (1-tau) * 1.0
        for name, p in ema.target_model.named_parameters():
            expected = tau * old_target[name] + (1 - tau) * 1.0
            assert torch.allclose(p, expected, atol=1e-6), (
                f"EMA update incorrect for {name}"
            )

    def test_tau_zero(self):
        """Test tau=0 (instant copy)."""
        model = SimpleModel(16)
        ema = EMATargetPolicy(model, tau=0.0)

        # Fill online with 1s
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(1.0)

        ema.update(model)

        # Target should be exactly equal to online
        for tp, op in zip(ema.target_model.parameters(), model.parameters()):
            assert torch.allclose(tp, op)

    def test_tau_one(self):
        """Test tau=1.0 (frozen target, no update)."""
        model = SimpleModel(16)
        ema = EMATargetPolicy(model, tau=1.0)

        old_target = {
            name: p.clone() for name, p in ema.target_model.named_parameters()
        }

        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)

        ema.update(model)

        for name, p in ema.target_model.named_parameters():
            assert torch.allclose(p, old_target[name])

    def test_invalid_tau(self):
        """Test that invalid tau raises ValueError."""
        model = SimpleModel(16)
        with pytest.raises(ValueError, match="tau must be in"):
            EMATargetPolicy(model, tau=-0.1)
        with pytest.raises(ValueError, match="tau must be in"):
            EMATargetPolicy(model, tau=1.5)

    def test_state_dict_roundtrip(self):
        """Test save/load state dict."""
        model = SimpleModel(16)
        ema = EMATargetPolicy(model, tau=0.99)

        state = ema.state_dict()
        ema2 = EMATargetPolicy(model, tau=0.99)
        ema2.load_state_dict(state)

        for p1, p2 in zip(
            ema.target_model.parameters(), ema2.target_model.parameters()
        ):
            assert torch.allclose(p1, p2)
