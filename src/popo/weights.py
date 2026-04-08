"""
Bounded importance weight computation for POPO.

Computes normalized weights over the positive set:
    w_theta(y|x) = pi_theta(y|x) / Z+(x)
    where Z+(x) = sum_{y' in S+(x)} pi_theta(y'|x)

This creates self-competition among correct responses, reinforcing
confident correct answers more strongly while maintaining diversity.

Supports two weight modes:
    - "softmax": softmax-normalized weights over positive set (Eq. 8)
    - "uniform": equal weights 1/|S+(x)| per prompt group (ablation)

Optional length normalization divides weights by response length,
penalizing verbose correct answers.

Corresponds to Eq. 8 in the manuscript.
"""

import torch
import torch.nn.functional as F


def compute_popo_weights(
    per_token_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    positive_mask: torch.Tensor,
    num_generations: int,
    temperature: float = 1.0,
    weight_mode: str = "softmax",
    length_normalize: bool = False,
) -> torch.Tensor:
    """
    Compute normalized importance weights over the positive set.

    For each prompt, computes weights over positive responses only,
    giving zero weight to negatives. Weights sum to 1.0 within each
    prompt's positive set.

    Args:
        per_token_logps: Per-token log probabilities, shape (B, T)
            where B = num_prompts * num_generations.
        completion_mask: Mask for completion tokens, shape (B, T).
        positive_mask: Boolean mask for positive (correct) responses,
            shape (B,). True = correct, False = incorrect.
        num_generations: Number of generations per prompt (G).
        temperature: Temperature for softmax (default: 1.0).
            Only used when weight_mode="softmax".
        weight_mode: "softmax" for policy-probability-based weights,
            "uniform" for equal weights across positives (ablation).
        length_normalize: If True, divide weights by response length
            then re-normalize. Penalizes verbose correct answers.

    Returns:
        Normalized weights, shape (B,). Zero for non-positive responses.
        Sums to 1.0 within each prompt's positive set.
    """
    batch_size = per_token_logps.size(0)
    num_prompts = batch_size // num_generations
    positive_mask_grouped = positive_mask.view(num_prompts, num_generations)

    if weight_mode == "softmax":
        seq_logps = compute_sequence_log_probs(per_token_logps, completion_mask)
        seq_logps_grouped = seq_logps.view(num_prompts, num_generations)

        masked_logps = seq_logps_grouped.clone()
        masked_logps[~positive_mask_grouped] = float("-inf")

        weights_grouped = F.softmax(masked_logps / temperature, dim=-1)
        weights_grouped = torch.nan_to_num(weights_grouped, nan=0.0)

    elif weight_mode == "uniform":
        pos_counts = positive_mask_grouped.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        weights_grouped = positive_mask_grouped.float() / pos_counts

    else:
        raise ValueError(
            f"weight_mode must be 'softmax' or 'uniform', got '{weight_mode}'"
        )

    if length_normalize:
        seq_lengths = completion_mask.sum(dim=-1).clamp(min=1.0)
        seq_lengths_grouped = seq_lengths.view(num_prompts, num_generations)
        weights_grouped = weights_grouped / seq_lengths_grouped
        # Zero out negatives and re-normalize per prompt group
        weights_grouped = weights_grouped * positive_mask_grouped.float()
        group_sums = weights_grouped.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weights_grouped = weights_grouped / group_sums

    weights = weights_grouped.view(batch_size)
    return weights


def compute_sequence_log_probs(
    per_token_logps: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute length-normalized sequence-level log probabilities.

    Sums per-token log-probs over the completion, normalized by
    completion length for fair comparison across variable-length responses.

    Args:
        per_token_logps: Per-token log probabilities, shape (B, T).
        completion_mask: Mask for valid completion tokens, shape (B, T).

    Returns:
        Average log probability per sequence, shape (B,).
    """
    seq_logps = (per_token_logps * completion_mask).sum(dim=-1)
    seq_lengths = completion_mask.sum(dim=-1).clamp(min=1.0)
    seq_logps = seq_logps / seq_lengths
    return seq_logps
