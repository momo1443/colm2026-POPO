"""
POPOConfig: configuration class for POPO training.

Extends TRL's GRPOConfig with POPO-specific parameters.
Key addition: forces beta=0.0 (disabling KL penalty,
since POPO uses similarity loss instead).
"""

import logging
from dataclasses import dataclass, field
from trl import GRPOConfig

logger = logging.getLogger(__name__)


@dataclass
class POPOConfig(GRPOConfig):
    """
    Configuration for POPO (Positive-Only Policy Optimization).

    Extends GRPOConfig with:
    - alpha: Similarity loss weight
    - beta_entropy: Entropy regularization coefficient
    - tau: EMA momentum coefficient
    - tau_r: Reward threshold for positive filtering
    - Predictor MLP architecture parameters
    - Feature noise for robustness

    Note: beta (inherited from GRPOConfig) is forced to 0.0 in
    __post_init__. POPO replaces KL with similarity loss.
    """

    # POPO-specific hyperparameters
    alpha: float = field(
        default=0.1,
        metadata={"help": "Similarity loss weight. Range: [0.05, 0.3]."},
    )
    beta_entropy: float = field(
        default=0.01,
        metadata={"help": "Entropy regularization coefficient. Range: [0.005, 0.05]."},
    )
    tau: float = field(
        default=0.995,
        metadata={"help": "EMA momentum coefficient. Range: [0.99, 0.999]."},
    )
    tau_r: float = field(
        default=0.0,
        metadata={"help": "Reward threshold for positive filtering (0 for binary RLVR)."},
    )

    # Predictor MLP
    predictor_hidden_dim: int = field(
        default=1024,
        metadata={"help": "Hidden dimension of the predictor MLP."},
    )
    predictor_layers: int = field(
        default=2,
        metadata={"help": "Number of layers in the predictor MLP (>= 2)."},
    )

    # Feature noise
    feature_noise_std: float = field(
        default=0.02,
        metadata={"help": "Std of Gaussian noise added to target features."},
    )

    # Weight computation options
    weight_mode: str = field(
        default="softmax",
        metadata={
            "help": (
                "Weight mode for positive-set importance weighting. "
                "'softmax': policy-probability-based (Eq. 8). "
                "'uniform': equal weights for ablation."
            )
        },
    )
    length_normalize_weights: bool = field(
        default=False,
        metadata={
            "help": (
                "Normalize POPO weights by response length to penalize "
                "verbose outputs. Divides each weight by 1/|y| then "
                "re-normalizes per prompt group."
            )
        },
    )

    # Override GRPOConfig default: disable KL penalty for POPO
    beta: float = field(
        default=0.0,
        metadata={
            "help": (
                "KL divergence coefficient. Forced to 0.0 for POPO "
                "(uses similarity loss instead of KL penalty)."
            )
        },
    )

    # KL monitoring frequency (compute KL every N steps for TensorBoard)
    kl_monitoring_steps: int = field(
        default=10,
        metadata={"help": "Compute KL divergence for monitoring every N steps (0 to disable)."},
    )

    def __post_init__(self):
        super().__post_init__()

        # Enforce beta=0 for POPO
        if self.beta != 0.0:
            logger.warning(
                "POPOConfig: beta was set to %.4f but POPO requires beta=0.0 "
                "(KL penalty disabled). Overriding to 0.0.",
                self.beta,
            )
            self.beta = 0.0
