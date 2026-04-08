"""
POPO loss components.

Implements the combined POPO loss function:
    L_POPO(theta, phi) = L_NLL(theta) + alpha * L_sim(theta, phi) + beta_entropy * L_ent(theta)

where:
    L_NLL: Weighted negative log-likelihood over positive set
    L_sim: Weighted cosine similarity loss between predictor output and EMA target features
    L_ent: Entropy regularization for exploration

Also computes KL divergence (online vs target) for TensorBoard monitoring
(not used in POPO gradient).

See manuscript Algorithm 1 and Equations 6-10.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class POPOLoss:
    """
    Computes POPO loss components and aggregates the total loss.

    Args:
        alpha: Weight for similarity loss (default: 0.1).
        beta_entropy: Weight for entropy regularization (default: 0.01).
        feature_noise_std: Std of Gaussian noise added to target features
            for robustness (default: 0.02).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta_entropy: float = 0.01,
        feature_noise_std: float = 0.02,
    ):
        self.alpha = alpha
        self.beta_entropy = beta_entropy
        self.feature_noise_std = feature_noise_std

    def nll_loss(
        self,
        per_token_logps: torch.Tensor,
        completion_mask: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weighted negative log-likelihood loss over the positive set.

        Computes token-level weighted NLL normalized by total positive
        token count (standard SFT-style normalization):

            L_NLL = sum_{y in S+} sum_t [w(y|x) * (-log pi(y_t | y_{<t}, x)) * mask_t]
                    / sum_{y in S+} |y|

        This is the standard token-level normalization used in SFT and
        TRL's GRPO. Longer positive sequences contribute proportionally
        more tokens to both numerator and denominator.

        Args:
            per_token_logps: Per-token log probs, shape (B, T).
            completion_mask: Valid token mask, shape (B, T).
            weights: POPO weights per sequence, shape (B,).
                Zero for negative responses.

        Returns:
            Scalar NLL loss.
        """
        # Negate log-probs to get per-token NLL
        per_token_nll = -per_token_logps

        # Weighted NLL: weight each sequence's token-level NLL
        # weights shape (B,) -> (B, 1) for broadcasting
        weighted_nll = weights.unsqueeze(1) * per_token_nll * completion_mask

        # Normalize by positive token count only (where weights > 0)
        positive_token_mask = (weights > 0).unsqueeze(1).float() * completion_mask
        positive_token_count = positive_token_mask.sum().clamp(min=1.0)
        loss = weighted_nll.sum() / positive_token_count

        return loss

    def similarity_loss(
        self,
        online_features: torch.Tensor,
        target_features: torch.Tensor,
        weights: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """
        Weighted cosine similarity loss between predictor output and target features.

        L_sim = -sum_{y in S+} w(y|x) * cos(h_phi(z_online(y)), sg(z_target(y) + noise))
                / sum(weights)

        Normalizing by the sum of weights makes the loss invariant to the
        number of active prompt groups. Since POPO weights sum to 1.0 per
        prompt group, weights.sum() equals the (possibly fractional) number
        of active groups in the sub-batch. This avoids requiring prompt-group
        structure after TRL's shuffle/split.

        Args:
            online_features: Predicted features from predictor MLP,
                shape (B, D). Already passed through h_phi.
            target_features: Features from EMA target model,
                shape (B, D). Stop-gradient applied externally.
            weights: POPO weights per sequence, shape (B,).
                Zero for negative responses.
            training: Whether in training mode (affects noise injection).

        Returns:
            Scalar similarity loss (weighted negated cosine similarity,
            normalized by sum of weights).
        """
        if self.feature_noise_std > 0 and training:
            noise = torch.randn_like(target_features) * self.feature_noise_std
            target_features = target_features + noise

        online_norm = F.normalize(online_features, dim=-1)
        target_norm = F.normalize(target_features, dim=-1)

        cos_sim = (online_norm * target_norm).sum(dim=-1)  # (B,)
        weighted_cos_sim = weights * cos_sim

        weight_sum = weights.sum().clamp(min=1e-8)
        loss = -weighted_cos_sim.sum() / weight_sum
        # loss = -weighted_cos_sim.sum()

        return loss

    def entropy_loss(
        self,
        entropies: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Entropy regularization loss.

        L_ent = -mean(entropy)

        Negated because we want to maximize entropy (encourage exploration).

        Args:
            entropies: Per-token entropies, shape (B, T).
            completion_mask: Valid token mask, shape (B, T).

        Returns:
            Scalar entropy loss (negated mean entropy).
        """
        token_count = completion_mask.sum().clamp(min=1.0)
        mean_entropy = (entropies * completion_mask).sum() / token_count

        # Negate: minimizing this loss = maximizing entropy
        loss = -mean_entropy

        return loss

    @staticmethod
    def compute_kl_divergence(
        online_log_probs: torch.Tensor,
        target_log_probs: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute approximate KL divergence between online and target policies.

        Used for monitoring/logging only (not part of POPO gradient).
        Uses the approximation consistent with TRL's GRPOTrainer:
            KL ≈ E[exp(log_target - log_online) - (log_target - log_online) - 1]

        Note: This corresponds to the reverse KL direction, matching
        TRL's convention.

        Args:
            online_log_probs: Per-token log probs from online model, shape (B, T).
            target_log_probs: Per-token log probs from target model, shape (B, T).
            completion_mask: Valid token mask, shape (B, T).

        Returns:
            Scalar mean KL divergence.
        """
        per_token_kl = (
            torch.exp(target_log_probs - online_log_probs)
            - (target_log_probs - online_log_probs)
            - 1
        )

        token_count = completion_mask.sum().clamp(min=1.0)
        mean_kl = (per_token_kl * completion_mask).sum() / token_count

        return mean_kl

    def __call__(
        self,
        per_token_logps: torch.Tensor,
        completion_mask: torch.Tensor,
        weights: torch.Tensor,
        entropies: torch.Tensor,
        online_features: torch.Tensor,
        target_features: torch.Tensor,
        target_log_probs: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the full POPO loss.

        L_POPO = L_NLL + alpha * L_sim + beta_entropy * L_ent

        Args:
            per_token_logps: Per-token log probs, shape (B, T).
            completion_mask: Valid token mask, shape (B, T).
            weights: POPO weights per sequence, shape (B,).
            entropies: Per-token entropies, shape (B, T).
            online_features: Predicted features from predictor, shape (B, D).
            target_features: EMA target features, shape (B, D).
            target_log_probs: Optional target log probs for KL monitoring,
                shape (B, T).
            training: Whether in training mode (affects noise).

        Returns:
            Dict with keys:
                'loss': Total POPO loss (scalar, for backward)
                'nll_loss': L_NLL component (detached)
                'sim_loss': L_sim component (detached)
                'ent_loss': L_ent component (detached)
                'entropy': Mean entropy value (for monitoring, detached)
                'kl_divergence': KL(online || target) (for monitoring, detached)
        """
        nll = self.nll_loss(per_token_logps, completion_mask, weights)
        sim = self.similarity_loss(
            online_features, target_features, weights, training=training,
        )
        ent = self.entropy_loss(entropies, completion_mask)

        # Total loss
        total_loss = nll + self.alpha * sim + self.beta_entropy * ent

        # Compute monitoring metrics
        token_count = completion_mask.sum().clamp(min=1.0)
        mean_entropy = (entropies.detach() * completion_mask).sum() / token_count

        # KL divergence (for logging only, not in gradient)
        kl_div = torch.tensor(0.0, device=total_loss.device)
        if target_log_probs is not None:
            kl_div = self.compute_kl_divergence(
                per_token_logps.detach(), target_log_probs, completion_mask
            )

        return {
            "loss": total_loss,
            "nll_loss": nll.detach(),
            "sim_loss": sim.detach(),
            "ent_loss": ent.detach(),
            "entropy": mean_entropy.detach(),
            "kl_divergence": kl_div.detach(),
        }
