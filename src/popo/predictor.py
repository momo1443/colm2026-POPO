"""
Predictor MLP for POPO's asymmetric online-target architecture.

Follows BYOL/SimSiam design: a predictor head h_phi applied only to the
online branch, creating asymmetry that prevents representational collapse
in positive-only contrastive learning.

Architecture: [Linear -> LayerNorm -> ReLU] x (N-1) -> Linear
Uses LayerNorm (not BatchNorm) for compatibility with DDP/FSDP and
batch_size=1 edge cases.
"""

import torch
import torch.nn as nn


class PredictorMLP(nn.Module):
    """
    Predictor network h_phi for asymmetric online-target architecture.

    Maps online features to the target feature space via a bottleneck MLP.
    Applied only to the online branch (not the target), ensuring the online
    network must learn meaningful transformations to match the target.

    Args:
        input_dim: Dimension of input features (model's hidden_size).
        hidden_dim: Hidden layer dimension (default: 1024).
        output_dim: Output dimension (default: same as input_dim).
        num_layers: Number of linear layers, must be >= 2. Default: 2.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        output_dim: int = None,
        num_layers: int = 2,
        device: torch.device = None,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")
        output_dim = output_dim or input_dim

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        if device is not None:
            self.net = self.net.to(device)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform and zero biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the predictor MLP.

        Args:
            x: Input features of shape (batch_size, input_dim).

        Returns:
            Predicted features of shape (batch_size, output_dim).
        """
        return self.net(x)
