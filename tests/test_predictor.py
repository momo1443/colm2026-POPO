"""Unit tests for PredictorMLP."""

import pytest
import torch

from src.popo.predictor import PredictorMLP


class TestPredictorMLP:
    """Tests for the predictor MLP module."""

    def test_output_shape_2layer(self):
        """Test 2-layer predictor produces correct output shape."""
        pred = PredictorMLP(input_dim=768, hidden_dim=1024, output_dim=768, num_layers=2)
        x = torch.randn(8, 768)
        out = pred(x)
        assert out.shape == (8, 768)

    def test_output_shape_3layer(self):
        """Test 3-layer predictor produces correct output shape."""
        pred = PredictorMLP(input_dim=896, hidden_dim=1024, output_dim=896, num_layers=3)
        x = torch.randn(4, 896)
        out = pred(x)
        assert out.shape == (4, 896)

    def test_default_output_dim(self):
        """Test that output_dim defaults to input_dim."""
        pred = PredictorMLP(input_dim=512, hidden_dim=256)
        x = torch.randn(2, 512)
        out = pred(x)
        assert out.shape == (2, 512)

    def test_gradient_flow(self):
        """Test that gradients flow through the predictor."""
        pred = PredictorMLP(input_dim=64, hidden_dim=32, output_dim=64)
        x = torch.randn(4, 64, requires_grad=True)
        out = pred(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        for p in pred.parameters():
            assert p.grad is not None

    def test_output_shape_4layer(self):
        """Test 4-layer predictor produces correct output shape."""
        pred = PredictorMLP(input_dim=768, hidden_dim=512, output_dim=768, num_layers=4)
        x = torch.randn(4, 768)
        out = pred(x)
        assert out.shape == (4, 768)

    def test_output_shape_5layer(self):
        """Test 5-layer predictor produces correct output shape."""
        pred = PredictorMLP(input_dim=256, hidden_dim=128, output_dim=256, num_layers=5)
        x = torch.randn(2, 256)
        out = pred(x)
        assert out.shape == (2, 256)

    def test_invalid_num_layers(self):
        """Test that num_layers < 2 raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be >= 2"):
            PredictorMLP(input_dim=64, hidden_dim=32, num_layers=1)

    def test_invalid_num_layers_0(self):
        """Test that num_layers=0 raises ValueError."""
        with pytest.raises(ValueError, match="num_layers must be >= 2"):
            PredictorMLP(input_dim=64, hidden_dim=32, num_layers=0)

    def test_batch_size_one(self):
        """Test predictor works with batch_size=1 (LayerNorm, not BatchNorm)."""
        pred = PredictorMLP(input_dim=64, hidden_dim=32)
        x = torch.randn(1, 64)
        out = pred(x)
        assert out.shape == (1, 64)

    def test_uses_layernorm(self):
        """Verify predictor uses LayerNorm (not BatchNorm1d)."""
        pred = PredictorMLP(input_dim=64, hidden_dim=32)
        has_layernorm = any(
            isinstance(m, torch.nn.LayerNorm) for m in pred.modules()
        )
        has_batchnorm = any(
            isinstance(m, torch.nn.BatchNorm1d) for m in pred.modules()
        )
        assert has_layernorm, "Should use LayerNorm"
        assert not has_batchnorm, "Should not use BatchNorm1d"
