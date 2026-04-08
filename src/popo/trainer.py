"""
POPOTrainer: Positive-Only Policy Optimization trainer.

Extends TRL's GRPOTrainer with three key modifications:
1. Positive-only filtering: only train on responses with reward > tau_r
2. EMA target policy: replaces frozen reference with slowly-evolving target
3. Similarity loss + entropy regularization: replaces KL penalty

Override points:
    - _generate_and_score_completions: Injects raw rewards AND pre-computes
        POPO weights for the full generation batch (before TRL's shuffle/split).
    - _compute_loss: Core POPO loss computation using pre-computed weights.
    - create_optimizer: Includes predictor MLP parameters.

Design note (batch alignment):
    TRL's _prepare_inputs shuffles and splits the generation batch into
    sub-batches for gradient accumulation. This breaks prompt grouping
    (all num_generations completions per prompt). Since POPO's weight
    computation requires per-prompt softmax normalization, we pre-compute
    weights in _generate_and_score_completions where prompt groups are
    intact, then store them in the output dict. The per-sample weights
    survive the shuffle/split and are used directly in _compute_loss.

See manuscript Algorithm 1 for the full training loop.
"""

import logging
import os
from typing import Optional, List, Union, Callable

import torch
from transformers import AutoModelForCausalLM
from trl import GRPOTrainer
from trl.trainer.utils import selective_log_softmax

from src.popo.config import POPOConfig
from src.popo.ema import EMATargetPolicy
from src.popo.predictor import PredictorMLP
from src.popo.loss import POPOLoss
from src.popo.weights import compute_popo_weights
from src.popo.callbacks import EMAUpdateCallback

logger = logging.getLogger(__name__)


class _RewardInterceptor:
    """
    Wraps a reward function to cache the last computed rewards.

    TRL's _generate_and_score_completions computes rewards internally
    but does not pass them to _compute_loss. This interceptor transparently
    caches rewards so that POPOTrainer can access raw reward values for
    positive-set filtering.
    """

    def __init__(self, fn: Callable):
        self._fn = fn
        self.last_rewards: Optional[torch.Tensor] = None
        # Preserve attributes TRL uses for metric logging
        self.__name__ = getattr(fn, "__name__", fn.__class__.__name__)
        self.__doc__ = getattr(fn, "__doc__", "")

    def __call__(self, *args, **kwargs):
        rewards = self._fn(*args, **kwargs)
        # Cache as tensor for later use
        if isinstance(rewards, list):
            self.last_rewards = torch.tensor(rewards, dtype=torch.float32)
        elif isinstance(rewards, torch.Tensor):
            self.last_rewards = rewards.clone().detach().float()
        else:
            self.last_rewards = torch.tensor(rewards, dtype=torch.float32)
        return rewards


class POPOTrainer(GRPOTrainer):
    """
    Trainer for POPO (Positive-Only Policy Optimization).

    Extends GRPOTrainer by overriding _compute_loss to implement:
    - Positive set filtering (reward > tau_r via intercepted rewards)
    - POPO weight computation (softmax over positive set)
    - NLL + similarity + entropy loss (single forward pass)
    - KL divergence monitoring (optional, every N steps)

    Automatically initializes:
    - EMATargetPolicy: target model updated via EMA callback
    - PredictorMLP: asymmetric predictor head (included in optimizer)
    - POPOLoss: loss aggregation
    - EMAUpdateCallback: registered as TrainerCallback

    Args:
        args: POPOConfig instance.
        reward_funcs: Reward function(s) for scoring completions.
        **kwargs: All other GRPOTrainer arguments (model, train_dataset, etc.).
    """

    def __init__(
        self,
        args: POPOConfig,
        reward_funcs: Union[Callable, List[Callable]] = None,
        **kwargs,
    ):
        if not isinstance(args, POPOConfig):
            raise TypeError(
                f"POPOTrainer requires POPOConfig, got {type(args).__name__}"
            )

        # Store POPO config before super().__init__
        self.popo_config = args

        # Intercept reward function(s) to cache raw rewards.
        # When multiple reward functions are provided, all interceptors'
        # rewards are averaged for positive filtering and weight computation.
        if reward_funcs is not None:
            if callable(reward_funcs) and not isinstance(reward_funcs, list):
                self._reward_interceptors = [_RewardInterceptor(reward_funcs)]
                reward_funcs = self._reward_interceptors[0]
            elif isinstance(reward_funcs, list):
                self._reward_interceptors = [
                    _RewardInterceptor(fn) for fn in reward_funcs
                ]
                reward_funcs = self._reward_interceptors

        # Initialize base GRPOTrainer
        super().__init__(args=args, reward_funcs=reward_funcs, **kwargs)

        # Initialize POPO-specific components after base init
        self._init_popo_components()

    def _init_popo_components(self):
        """Initialize EMA target, predictor MLP, and loss function."""
        config = self.popo_config

        # Get model's hidden size for predictor dimensions
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped_model, "config"):
            model_config = unwrapped_model.config
            hidden_size = getattr(model_config, "hidden_size", 1024)
        else:
            hidden_size = 1024
            logger.warning(
                "Could not detect model hidden_size, defaulting to %d", hidden_size
            )

        # ------------------------------------------------------------------
        # EMA target policy  (Bug C2/C3: FSDP vs DDP branching)
        # ------------------------------------------------------------------
        if self.is_fsdp_enabled:
            # Under FSDP, accelerator.unwrap_model() only strips the
            # outermost wrapper; per-layer FSDP wrapping remains, so
            # deepcopy would copy sharded FlatParameter tensors (each
            # rank holds only 1/N-th of parameters).  Instead: gather
            # full params, build a fresh model, then FSDP-wrap it.
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from trl.models import prepare_fsdp

            with FSDP.summon_full_params(self.model, writeback=False):
                full_state_dict = unwrapped_model.state_dict()

            target_model = AutoModelForCausalLM.from_config(
                unwrapped_model.config
            )
            target_model.load_state_dict(full_state_dict)
            del full_state_dict
            torch.cuda.empty_cache()
            target_model = prepare_fsdp(target_model, self.accelerator)

            self.ema_target = EMATargetPolicy(
                tau=config.tau,
                target_model=target_model,
            )
            logger.info("EMA target created via FSDP-aware path (prepare_fsdp)")
        else:
            self.ema_target = EMATargetPolicy(
                model=unwrapped_model,
                tau=config.tau,
                device=self.accelerator.device,
            )

        # ------------------------------------------------------------------
        # Predictor MLP  (Bug W1: avoid broken FSDP auto-wrap on tiny MLP)
        # ------------------------------------------------------------------
        self.predictor = PredictorMLP(
            input_dim=hidden_size,
            hidden_dim=config.predictor_hidden_dim,
            output_dim=hidden_size,
            num_layers=config.predictor_layers,
            device=self.accelerator.device,
        )

        if self.accelerator.num_processes > 1:
            import torch.distributed as dist

            if not dist.is_initialized():
                raise RuntimeError(
                    "Distributed process group not initialized when "
                    "trying to wrap predictor with DDP."
                )
            self.predictor = torch.nn.parallel.DistributedDataParallel(
                self.predictor,
                device_ids=[self.accelerator.local_process_index],
            )
            logger.info("Predictor MLP wrapped with DDP")
        # Single GPU: no wrapping needed

        # POPO Loss
        self.popo_loss_fn = POPOLoss(
            alpha=config.alpha,
            beta_entropy=config.beta_entropy,
            feature_noise_std=config.feature_noise_std,
        )

        # Register EMA callback (with accelerator for unwrapping)
        self.add_callback(
            EMAUpdateCallback(self.ema_target, accelerator=self.accelerator)
        )

        logger.info(
            "POPO components initialized: "
            "alpha=%.3f, beta_entropy=%.3f, tau=%.4f, tau_r=%.2f, "
            "predictor_dim=%d (hidden_size=%d), feature_noise=%.3f",
            config.alpha,
            config.beta_entropy,
            config.tau,
            config.tau_r,
            config.predictor_hidden_dim,
            hidden_size,
            config.feature_noise_std,
        )

    def create_optimizer(self):
        """
        Override to include predictor MLP parameters in the optimizer.

        The base Trainer.create_optimizer() only includes self.model.parameters().
        We add the predictor MLP parameters so they receive gradients from
        the similarity loss.  The predictor may be DDP-wrapped, so we
        use ``accelerator.unwrap_model`` for a consistent parameter set.
        """
        optimizer = super().create_optimizer()

        if hasattr(self, "predictor"):
            unwrapped_predictor = self.accelerator.unwrap_model(self.predictor)
            predictor_params = list(unwrapped_predictor.parameters())
            if predictor_params:
                param_group = {
                    "params": predictor_params,
                    "lr": self.args.learning_rate,
                    "weight_decay": self.args.weight_decay,
                }
                self.optimizer.add_param_group(param_group)
                logger.info(
                    "Added predictor MLP to optimizer (%d parameters)",
                    sum(p.numel() for p in predictor_params),
                )

        return self.optimizer

    # ------------------------------------------------------------------
    # Override: _generate_and_score_completions
    # ------------------------------------------------------------------
    def _generate_and_score_completions(self, inputs):
        """
        Override to inject raw rewards AND pre-compute POPO weights.

        TRL's _prepare_inputs shuffles and splits the generation batch into
        sub-batches for gradient accumulation. This breaks prompt grouping.
        Since compute_popo_weights requires per-prompt softmax normalization,
        we compute weights HERE where the full prompt groups are intact.
        The per-sample weights survive the shuffle/split and are used
        directly in _compute_loss.

        Added keys to result dict:
            raw_rewards: Raw reward values per completion (B,)
            popo_weights: Pre-computed POPO weights per completion (B,)
            positive_mask_float: 1.0 for positive, 0.0 for negative (B,)
        """
        result = super()._generate_and_score_completions(inputs)

        # Step 1: Inject raw rewards from interceptor(s)
        if hasattr(self, "_reward_interceptors"):
            device = result["completion_ids"].device
            available = [
                ri.last_rewards for ri in self._reward_interceptors
                if ri.last_rewards is not None
            ]
            if available:
                stacked = torch.stack([r.to(device) for r in available])
                result["raw_rewards"] = stacked.mean(dim=0)

        # Step 2: Pre-compute POPO weights for the full generation batch
        if "raw_rewards" in result:
            self._precompute_popo_weights(result)

        return result

    def _precompute_popo_weights(self, result):
        """
        Pre-compute POPO weights for the full generation batch.

        This method is called from _generate_and_score_completions where
        the batch still has proper prompt grouping (each prompt's
        num_generations completions are contiguous). After this, the
        result dict will have per-sample weights that can be shuffled/split.

        The model forward pass runs unconditionally because under FSDP it
        is a collective operation — all ranks must participate even when
        the local batch has no positive examples.

        Args:
            result: The output dict from _generate_and_score_completions.
                Modified in-place to add 'popo_weights' and
                'positive_mask_float'.
        """
        raw_rewards = result["raw_rewards"]
        positive_mask = raw_rewards > self.popo_config.tau_r

        num_positive = positive_mask.sum().item()
        batch_size = raw_rewards.size(0)

        # Always run the model forward — FSDP all-gather is collective.
        prompt_ids = result["prompt_ids"]
        prompt_mask = result["prompt_mask"]
        completion_ids = result["completion_ids"]
        completion_mask = result["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size=self.args.per_device_train_batch_size
                * self.num_generations,
            )

        if num_positive == 0:
            result["popo_weights"] = torch.zeros_like(raw_rewards)
            result["positive_mask_float"] = torch.zeros_like(raw_rewards)
            logger.warning(
                "No positive examples in generation batch (%d samples, "
                "tau_r=%.2f). All POPO weights set to zero.",
                batch_size,
                self.popo_config.tau_r,
            )
            return

        weights = compute_popo_weights(
            per_token_logps=per_token_logps,
            completion_mask=completion_mask,
            positive_mask=positive_mask,
            num_generations=self.num_generations,
            weight_mode=self.popo_config.weight_mode,
            length_normalize=self.popo_config.length_normalize_weights,
        )

        result["popo_weights"] = weights
        result["positive_mask_float"] = positive_mask.float()

        logger.debug(
            "Pre-computed POPO weights: %d/%d positive (%.1f%%), "
            "weight range [%.4f, %.4f]",
            num_positive,
            batch_size,
            100.0 * num_positive / batch_size,
            weights[weights > 0].min().item() if num_positive > 0 else 0.0,
            weights.max().item(),
        )

    # ------------------------------------------------------------------
    # Override: _compute_loss
    # ------------------------------------------------------------------
    def _compute_loss(self, model, inputs):
        """
        Compute the POPO loss.

        Uses pre-computed POPO weights from _generate_and_score_completions.
        This avoids the need for prompt-group alignment in mini-batches,
        which is broken by TRL's shuffle/split in _prepare_inputs.

        Key design decisions:
        - Single forward pass through online model (logits + hidden states)
        - Entropy computed WITH gradients (for L_ent to backprop)
        - Pre-computed weights from generation-time model (analogous to
          how GRPO uses generation-time advantages)
        - Similarity loss weighted by POPO weights (Eq. 9 in manuscript)

        Memory optimizations vs. naive approach:
        - logits_to_keep: only compute logits for completion positions
        - selective_log_softmax: avoids materializing full [B,T,V] log_softmax
        - Chunked entropy: processes 128 rows at a time instead of full tensor
        - Forward hook: captures only last-layer hidden states (not all 28)

        Args:
            model: The training model (may be wrapped by accelerate).
            inputs: Dict from _generate_and_score_completions, post-split.

        Returns:
            Scalar loss tensor.
        """
        # ==================================================================
        # Step 1: Single forward pass through online model
        # ==================================================================
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        completion_start = prompt_ids.size(1)

        # Hook on the final RMSNorm to capture post-norm hidden states.
        # Hooking here (instead of the last decoder layer) gives us post-norm
        # states directly and avoids calling norm() after the forward pass,
        # which is fragile under FSDP (norm weights may be resharded by then).
        captured = {}
        unwrapped = self.accelerator.unwrap_model(model)
        handle = unwrapped.model.norm.register_forward_hook(
            lambda m, i, o: captured.__setitem__("hidden", o)
        )

        # Forward with logits_to_keep: lm_head only processes the last N+1
        # positions, but the backbone still runs on all positions (so the
        # hook captures full-sequence hidden states).
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if "logits_to_keep" in self.model_kwarg_keys:
            model_kwargs["logits_to_keep"] = logits_to_keep + 1
        try:
            logits = model(**model_kwargs).logits
        finally:
            handle.remove()

        # Slice logits the same way TRL does: drop the last position,
        # then keep only the last logits_to_keep positions (no-op when
        # the model already applied logits_to_keep internally).
        logits = logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :]
        logits = logits / self.temperature

        # Per-token log-probs via TRL's memory-efficient selective_log_softmax
        # (avoids materializing full [B, T, V] log_softmax tensor)
        per_token_logps = selective_log_softmax(logits, completion_ids)

        # Entropy WITH gradients (needed for L_ent backprop), chunked to
        # avoid materializing full [B*T, V] softmax + log_softmax at once
        entropies = _chunked_entropy_from_logits(logits)

        del logits

        # ==================================================================
        # Step 2: Retrieve pre-computed weights and positive mask
        # ==================================================================
        if "popo_weights" in inputs:
            # Pre-computed weights from _generate_and_score_completions
            weights = inputs["popo_weights"]
            positive_mask = inputs.get(
                "positive_mask_float", (weights > 0).float()
            ).bool()
        elif "raw_rewards" in inputs:
            # Fallback: uniform weights over positive set (no prompt grouping)
            raw_rewards = inputs["raw_rewards"]
            positive_mask = raw_rewards > self.popo_config.tau_r
            num_pos = positive_mask.sum().clamp(min=1.0)
            weights = positive_mask.float() / num_pos
            logger.warning(
                "popo_weights not found in inputs; using uniform weights "
                "over positive set."
            )
        else:
            # Last resort: use advantages as proxy
            advantages = inputs["advantages"]
            positive_mask = advantages > 0
            num_pos = positive_mask.sum().clamp(min=1.0)
            weights = positive_mask.float() / num_pos
            logger.warning(
                "No raw_rewards or popo_weights; using advantages for "
                "positive filtering."
            )

        num_positive = positive_mask.sum().item()
        batch_size = positive_mask.size(0)

        # ==================================================================
        # Step 3: Extract features — runs unconditionally.
        # Under FSDP, forward passes are collective (all-gather per layer).
        # Every rank must execute them regardless of local positive count;
        # skipping would cause a rank-divergent collective deadlock.
        # ==================================================================
        online_hidden = captured["hidden"]
        online_completion_hidden = online_hidden[:, completion_start:, :]
        online_features = _masked_mean_pool(
            online_completion_hidden, completion_mask
        )
        del online_hidden

        # Under FSDP mixed precision, the hook captures bf16 hidden states
        # but the predictor (DDP-wrapped, not FSDP) has float32 weights.
        online_features = online_features.to(
            next(self.accelerator.unwrap_model(self.predictor).parameters()).dtype
        )

        predicted_features = self.predictor(online_features)

        with torch.no_grad():
            target_hidden = self.ema_target.get_hidden_states(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            target_completion_hidden = target_hidden[:, completion_start:, :]
            target_features = _masked_mean_pool(
                target_completion_hidden, completion_mask
            )

        # ==================================================================
        # Step 4: Optional KL monitoring (every N steps)
        # ==================================================================
        target_log_probs = None
        kl_steps = self.popo_config.kl_monitoring_steps
        should_compute_kl = (
            kl_steps > 0
            and self.state.global_step > 0
            and self.state.global_step % kl_steps == 0
        )
        if should_compute_kl:
            with torch.no_grad():
                target_logps, _ = self.ema_target.get_per_token_logps(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits_to_keep=logits_to_keep,
                )
                target_log_probs = target_logps

        # ==================================================================
        # Step 5: Compute loss (or graph-connected zero when no positives)
        # ==================================================================
        if num_positive == 0:
            # All forward passes above already ran (required for FSDP).
            # Return a zero loss connected to both model and predictor
            # so DDP/FSDP backward hooks fire with zero gradients.
            dummy_loss = 0.0 * per_token_logps.sum()
            dummy_loss = dummy_loss + 0.0 * predicted_features.sum()

            zero = dummy_loss.detach()
            self._log_popo_metrics(
                nll_loss=zero,
                sim_loss=zero,
                ent_loss=zero,
                entropy=zero,
                kl_divergence=None,
                positive_ratio=0.0,
            )
            return dummy_loss / self.current_gradient_accumulation_steps

        loss_dict = self.popo_loss_fn(
            per_token_logps=per_token_logps,
            completion_mask=completion_mask,
            weights=weights,
            entropies=entropies,
            online_features=predicted_features,
            target_features=target_features,
            target_log_probs=target_log_probs,
            training=self.model.training,
        )

        loss = loss_dict["loss"]

        # Scale by gradient accumulation (consistent with GRPO)
        loss = loss / self.current_gradient_accumulation_steps

        # ==================================================================
        # Step 6: Log metrics
        # ==================================================================
        self._log_popo_metrics(
            nll_loss=loss_dict["nll_loss"],
            sim_loss=loss_dict["sim_loss"],
            ent_loss=loss_dict["ent_loss"],
            entropy=loss_dict["entropy"],
            kl_divergence=(
                loss_dict["kl_divergence"] if target_log_probs is not None
                else None
            ),
            positive_ratio=num_positive / batch_size,
        )

        entropy_val = loss_dict["entropy"].item()
        if (
            entropy_val < 0.5
            and getattr(self, "_entropy_warned_step", -1) != self.state.global_step
        ):
            logger.warning(
                "Low entropy detected (%.3f < 0.5). Risk of policy collapse. "
                "Consider increasing beta_entropy.",
                entropy_val,
            )
            self._entropy_warned_step = self.state.global_step

        return loss

    # ------------------------------------------------------------------
    # Checkpoint save/load (persist EMA + predictor state)
    # ------------------------------------------------------------------
    def _save_checkpoint(self, model, trial):
        """Save EMA target and predictor alongside the model checkpoint.

        Under FSDP, ``FSDP.optim_state_dict(model, optimizer)`` maps
        every optimizer parameter to an FSDP fully-qualified name via
        the wrapped *model*.  The predictor lives in a separate DDP
        module and is not part of the FSDP model, so its parameters
        cause a ``KeyError``.  We temporarily remove the predictor
        param group before the parent save and restore it afterwards.
        The predictor's optimizer state is saved separately.
        """
        predictor_group = None
        predictor_opt_state = {}
        if self.is_fsdp_enabled and hasattr(self, "predictor"):
            unwrapped_pred = self.accelerator.unwrap_model(self.predictor)
            pred_param_ids = {id(p) for p in unwrapped_pred.parameters()}
            for idx, group in enumerate(self.optimizer.param_groups):
                if any(id(p) in pred_param_ids for p in group["params"]):
                    predictor_group = self.optimizer.param_groups.pop(idx)
                    for p in predictor_group["params"]:
                        if p in self.optimizer.state:
                            predictor_opt_state[id(p)] = self.optimizer.state.pop(p)
                    break

        super()._save_checkpoint(model, trial)

        if predictor_group is not None:
            for p in predictor_group["params"]:
                if id(p) in predictor_opt_state:
                    self.optimizer.state[p] = predictor_opt_state[id(p)]
            self.optimizer.add_param_group(predictor_group)

        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        run_dir = self._get_output_dir(trial)
        checkpoint_dir = os.path.join(run_dir, checkpoint_folder)

        # Under FSDP, state_dict() is a collective (all-gather).
        # All ranks must call it; only the saving rank writes to disk.
        ema_sd = self.ema_target.state_dict()
        predictor_to_save = self.accelerator.unwrap_model(self.predictor)
        pred_sd = predictor_to_save.state_dict()

        if self.args.should_save:
            torch.save(ema_sd, os.path.join(checkpoint_dir, "ema_target.pt"))
            torch.save(pred_sd, os.path.join(checkpoint_dir, "predictor.pt"))
            if predictor_opt_state:
                torch.save(
                    {id(p): predictor_opt_state[id(p)] for p in predictor_group["params"]
                     if id(p) in predictor_opt_state},
                    os.path.join(checkpoint_dir, "predictor_optim.pt"),
                )
            logger.info("Saved EMA target and predictor to %s", checkpoint_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Restore EMA target and predictor from a checkpoint directory."""
        if self.is_fsdp_enabled and hasattr(self, "predictor"):
            unwrapped_pred = self.accelerator.unwrap_model(self.predictor)
            pred_param_ids = {id(p) for p in unwrapped_pred.parameters()}
            predictor_group = None
            predictor_opt_state = {}
            for idx, group in enumerate(self.optimizer.param_groups):
                if any(id(p) in pred_param_ids for p in group["params"]):
                    predictor_group = self.optimizer.param_groups.pop(idx)
                    for p in predictor_group["params"]:
                        if p in self.optimizer.state:
                            predictor_opt_state[id(p)] = self.optimizer.state.pop(p)
                    break

            super()._load_from_checkpoint(resume_from_checkpoint, model=model)

            if predictor_group is not None:
                for p in predictor_group["params"]:
                    if id(p) in predictor_opt_state:
                        self.optimizer.state[p] = predictor_opt_state[id(p)]
                self.optimizer.add_param_group(predictor_group)
        else:
            super()._load_from_checkpoint(resume_from_checkpoint, model=model)

        ema_path = os.path.join(resume_from_checkpoint, "ema_target.pt")
        pred_path = os.path.join(resume_from_checkpoint, "predictor.pt")

        if os.path.isfile(ema_path):
            self.ema_target.load_state_dict(
                torch.load(ema_path, map_location=self.accelerator.device, weights_only=True)
            )
            logger.info("Restored EMA target from %s", ema_path)
        else:
            logger.warning("EMA target checkpoint not found at %s", ema_path)

        if os.path.isfile(pred_path):
            predictor_to_load = self.accelerator.unwrap_model(self.predictor)
            predictor_to_load.load_state_dict(
                torch.load(pred_path, map_location=self.accelerator.device, weights_only=True)
            )
            logger.info("Restored predictor from %s", pred_path)
        else:
            logger.warning("Predictor checkpoint not found at %s", pred_path)

    # ------------------------------------------------------------------
    # Metric logging helper
    # ------------------------------------------------------------------
    def _log_popo_metrics(
        self,
        nll_loss,
        sim_loss,
        ent_loss,
        entropy,
        kl_divergence,
        positive_ratio,
    ):
        """
        Log POPO-specific metrics to the trainer's metric dict.

        Appends **local** scalar values only.  ``accelerator.gather()``
        is a collective all-gather that requires all ranks to
        participate; calling it inside ``_compute_loss`` (which runs
        per micro-batch during gradient accumulation) can cause
        deadlocks if gather fires on some code paths but not others.
        The HuggingFace Trainer already aggregates ``_metrics`` across
        ranks at logging time.
        """
        mode = "train" if self.model.training else "eval"

        def _to_scalar(v):
            return v.item() if torch.is_tensor(v) else float(v)

        self._metrics[mode]["popo/nll_loss"].append(_to_scalar(nll_loss))
        self._metrics[mode]["popo/sim_loss"].append(_to_scalar(sim_loss))
        self._metrics[mode]["popo/ent_loss"].append(_to_scalar(ent_loss))
        self._metrics[mode]["entropy"].append(_to_scalar(entropy))

        if kl_divergence is not None:
            self._metrics[mode]["popo/kl_divergence"].append(
                _to_scalar(kl_divergence)
            )

        self._metrics[mode]["popo/positive_ratio"].append(
            float(positive_ratio)
        )


def _chunked_entropy_from_logits(
    logits: torch.Tensor,
    chunk_size: int = 128,
) -> torch.Tensor:
    """
    Compute per-token entropy from logits using chunked row processing.

    Avoids materializing full [B*T, V] softmax + log_softmax tensors
    simultaneously. Processes ``chunk_size`` rows at a time, reducing
    peak activation memory from O(B*T*V) to O(chunk_size*V).

    Entropy is computed in fp32 for numerical stability even when
    logits are in bf16/fp16.

    Args:
        logits: Logit tensor of shape (B, T, V).
        chunk_size: Number of (B*T) rows to process per chunk.

    Returns:
        Per-token entropies of shape (B, T).
    """
    B, T, V = logits.shape
    logits_2d = logits.reshape(-1, V)
    entropy_chunks = []
    for i in range(0, logits_2d.size(0), chunk_size):
        chunk = logits_2d[i : i + chunk_size].float()
        log_probs = chunk.log_softmax(dim=-1)
        probs = log_probs.exp()
        entropy_chunks.append(-(probs * log_probs).sum(dim=-1))
    return torch.cat(entropy_chunks).reshape(B, T)


def _masked_mean_pool(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean pooling over masked positions.

    Args:
        hidden_states: Shape (B, T, D).
        mask: Shape (B, T). 1 for valid tokens, 0 for padding.

    Returns:
        Pooled features of shape (B, D).
    """
    mask_expanded = mask.unsqueeze(-1).to(hidden_states.dtype)  # (B, T, 1)
    sum_hidden = (hidden_states * mask_expanded).sum(dim=1)  # (B, D)
    count = mask_expanded.sum(dim=1).clamp(min=1.0)  # (B, 1)
    return sum_hidden / count
