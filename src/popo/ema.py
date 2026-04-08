"""
EMA (Exponential Moving Average) target policy manager for POPO.

Maintains a slowly-evolving copy of the online policy that serves as
a stable learning anchor. Inspired by BYOL/SimSiam's momentum encoder.

Update rule: xi <- tau * xi + (1 - tau) * theta
"""

import copy
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EMATargetPolicy:
    """
    Manages an EMA-updated target policy for POPO.

    The target model parameters are updated as an exponential moving
    average of the online model parameters, providing a stable anchor
    that tracks the policy's evolution without abrupt changes.

    Supports two construction modes:
        1. ``model`` provided: creates target via ``copy.deepcopy`` (DDP / single-GPU).
        2. ``target_model`` provided: uses a pre-built model directly (required
           for FSDP, where deep-copying sharded FlatParameters is incorrect).

    Args:
        model: The online model to deep-copy. Mutually exclusive with
            ``target_model``.
        tau: EMA momentum coefficient (higher = slower update).
            Typical range: [0.99, 0.999]. Also accepts 0.0 (instant copy)
            and 1.0 (frozen target) for ablations.
        device: Device for the target model when created via deep copy.
            Ignored when ``target_model`` is supplied (FSDP-prepared models
            manage their own device placement).
        target_model: A pre-built target model. When supplied, ``model``
            and ``device`` are ignored.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tau: float = 0.995,
        device: Optional[str] = None,
        target_model: Optional[nn.Module] = None,
    ):
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {tau}")

        self.tau = tau

        if target_model is not None:
            self.target_model = target_model
        elif model is not None:
            self.target_model = copy.deepcopy(model)
        else:
            raise ValueError("Either model or target_model must be provided")

        self.target_model.requires_grad_(False)
        self.target_model.eval()

        if device is not None and target_model is None:
            self.target_model = self.target_model.to(device)

        logger.info("EMA target policy initialized (tau=%.4f)", tau)

    @torch.no_grad()
    def update(self, online_model: nn.Module) -> None:
        """
        Perform EMA update of target parameters.

        xi <- tau * xi + (1 - tau) * theta

        When both models are FSDP-wrapped with matching sharding, the
        parameter shapes already align (both hold the same shard).  For
        the DDP / single-GPU path the unwrapped online model is passed
        directly.

        Args:
            online_model: The online model whose parameters are used
                for the update.  Under FSDP this may still carry FSDP
                per-layer wrappers (matching the target's sharding).
        """
        for target_param, online_param in zip(
            self.target_model.parameters(),
            online_model.parameters(),
        ):
            target_param.data.mul_(self.tau).add_(
                online_param.data.to(target_param.device),
                alpha=1.0 - self.tau,
            )

    @torch.no_grad()
    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get last-layer hidden states from the target model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).

        Returns:
            Last hidden states of shape (batch_size, seq_len, hidden_size).
        """
        outputs = self.target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return outputs.hidden_states[-1]

    @torch.no_grad()
    def get_per_token_logps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-token log probabilities from the target model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            logits_to_keep: Number of completion tokens.

        Returns:
            Tuple of:
                per_token_logps: shape (batch_size, logits_to_keep)
                hidden_states: Last-layer hidden states, shape (batch_size, seq_len, D)
        """
        outputs = self.target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]

        # Keep only the logits for completion tokens
        logits = logits[:, -logits_to_keep - 1 : -1, :]

        # Compute log-probs
        log_probs = logits.log_softmax(dim=-1)
        completion_ids = input_ids[:, -logits_to_keep:]
        per_token_logps = torch.gather(
            log_probs, dim=-1, index=completion_ids.unsqueeze(-1)
        ).squeeze(-1)

        return per_token_logps, hidden_states

    def state_dict(self):
        """Return target model state dict for checkpointing."""
        return self.target_model.state_dict()

    def load_state_dict(self, state_dict):
        """Load target model state dict from checkpoint."""
        self.target_model.load_state_dict(state_dict)
