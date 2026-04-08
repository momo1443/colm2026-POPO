"""
Training callbacks for POPO.

Provides EMAUpdateCallback that performs EMA parameter updates
after each optimizer step, using HuggingFace's TrainerCallback API.

This approach mirrors TRL's own SyncRefModelCallback pattern:
both the online model and the EMA target share the same FSDP sharding
(or are both unwrapped under DDP), so a direct parameter zip works.
"""

import logging

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class EMAUpdateCallback(TrainerCallback):
    """
    Callback to update EMA target parameters after each training step.

    Invokes ema_target.update(model) on every on_step_end event,
    which applies the EMA update rule:
        xi <- tau * xi + (1 - tau) * theta

    Under FSDP the EMA target is wrapped with matching sharding
    (via ``prepare_fsdp``), so we unwrap only the outermost accelerate
    wrapper and pass directly — no ``summon_full_params`` needed.
    Under DDP we similarly unwrap and pass the bare model.

    Args:
        ema_target: Instance of EMATargetPolicy to update.
        accelerator: HuggingFace Accelerator instance (for unwrapping).
    """

    def __init__(self, ema_target, accelerator=None):
        super().__init__()
        self.ema_target = ema_target
        self.accelerator = accelerator

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """
        Perform EMA update after each optimizer step.

        Args:
            args: Training arguments.
            state: Training state.
            control: Training control.
            model: The online model (may be DDP/FSDP wrapped).
        """
        if model is None:
            return

        if self.accelerator is not None:
            unwrapped = self.accelerator.unwrap_model(model)
        else:
            unwrapped = model
        self.ema_target.update(unwrapped)
