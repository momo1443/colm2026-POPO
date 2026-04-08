#!/usr/bin/env python3
"""
POPO Training Script.

Training script for Positive-Only Policy Optimization (POPO)
using POPOTrainer (extends GRPOTrainer). Supports argparse CLI
with optional YAML config defaults.

Usage:
    # With YAML config + CLI overrides:
    python scripts/train_popo.py --config configs/popo/default.yaml \
        --model_name Qwen/Qwen2.5-Math-1.5B --max_steps 1000

    # With accelerate (FSDP + vLLM server):
    CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
        --config_file=configs/accelerate/fsdp.yaml \
        scripts/train_popo.py \
        --config configs/popo/default.yaml \
        --model_name Qwen/Qwen2.5-Math-7B \
        --use_vllm --vllm_mode server
"""

import os
import sys
import argparse
import logging

import yaml
import torch
from peft import LoraConfig, TaskType

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.popo.config import POPOConfig
from src.popo.trainer import POPOTrainer
from src.reward.math_verify_reward import math_verify_reward
from src.data.templates import TEMPLATE_CONFIGS
from src.data.datasets import load_and_prepare_dataset
from src.utils import (
    apply_yaml_defaults,
    setup_experiment_dir,
    setup_logging,
    print_env_info,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Argument Parser
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all configuration groups."""
    parser = argparse.ArgumentParser(
        description="POPO Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (CLI args override YAML values)",
    )

    # --- Model ---
    model = parser.add_argument_group("Model Configuration")
    model.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-7B")
    model.add_argument("--attn_implementation", type=str, default="sdpa",
                        choices=["sdpa", "eager"])
    model.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    model.add_argument("--resume_from_checkpoint", type=str, default=None)

    # --- Dataset ---
    data = parser.add_argument_group("Dataset Configuration")
    data.add_argument("--dataset_name", type=str, default="agentica-org/DeepScaleR-Preview-Dataset")
    data.add_argument("--dataset_config", type=str, default=None)
    data.add_argument("--train_split", type=str, default="train")
    data.add_argument("--test_split", type=str, default=None)
    data.add_argument("--max_train_samples", type=int, default=None)
    data.add_argument("--max_test_samples", type=int, default=None)

    # --- Template ---
    tmpl = parser.add_argument_group("Template Configuration")
    tmpl.add_argument("--prompt_template", type=str, default="qwen_math",
                       choices=list(TEMPLATE_CONFIGS.keys()))

    # --- Training ---
    train = parser.add_argument_group("Training Configuration")
    train.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: auto-generated under logs/)")
    train.add_argument("--max_steps", type=int, default=1000)
    train.add_argument("--per_device_train_batch_size", type=int, default=1)
    train.add_argument("--gradient_accumulation_steps", type=int, default=8)
    train.add_argument("--learning_rate", type=float, default=1e-6)
    train.add_argument("--warmup_ratio", type=float, default=0.1)
    train.add_argument("--lr_scheduler_type", type=str, default="cosine")
    train.add_argument("--weight_decay", type=float, default=0.01)
    train.add_argument("--max_grad_norm", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=42)

    # --- POPO-specific ---
    popo = parser.add_argument_group("POPO Configuration")
    popo.add_argument("--alpha", type=float, default=0.1,
                       help="Similarity loss weight [0.05, 0.3]")
    popo.add_argument("--beta", type=float, default=0.0,
                       help="kl regularization coefficient [0,1]")
    popo.add_argument("--beta_entropy", type=float, default=0.01,
                       help="Entropy regularization coefficient [0.005, 0.05]")
    popo.add_argument("--tau", type=float, default=0.995,
                       help="EMA momentum coefficient [0.99, 0.999]")
    popo.add_argument("--tau_r", type=float, default=0.0,
                       help="Reward threshold for positive filtering")
    popo.add_argument("--predictor_hidden_dim", type=int, default=1024,
                       help="Hidden dimension of predictor MLP")
    popo.add_argument("--predictor_layers", type=int, default=2,
                       help="Number of predictor MLP layers (>= 2)")
    popo.add_argument("--feature_noise_std", type=float, default=0.02,
                       help="Gaussian noise std on target features")
    popo.add_argument("--kl_monitoring_steps", type=int, default=10,
                       help="Compute KL every N steps (0 to disable)")
    popo.add_argument("--weight_mode", type=str, default="softmax",
                       choices=["softmax", "uniform"],
                       help="Weight mode: 'softmax' (Eq.8) or 'uniform' (ablation)")
    popo.add_argument("--length_normalize_weights", action="store_true",
                       help="Normalize weights by response length to penalize verbose outputs")

    # --- Generation (shared with GRPO) ---
    gen = parser.add_argument_group("Generation Configuration")
    gen.add_argument("--num_generations", type=int, default=8)
    gen.add_argument("--max_completion_length", type=int, default=512)
    gen.add_argument("--max_prompt_length", type=int, default=512)
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--num_iterations", type=int, default=1)

    # --- LoRA ---
    lora = parser.add_argument_group("LoRA Configuration")
    lora.add_argument("--use_lora", action="store_true")
    lora.add_argument("--lora_r", type=int, default=16)
    lora.add_argument("--lora_alpha", type=int, default=32)
    lora.add_argument("--lora_dropout", type=float, default=0.05)
    lora.add_argument("--lora_target_modules", type=str, nargs="+",
                       default=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"])

    # --- vLLM ---
    vllm = parser.add_argument_group("vLLM Configuration")
    vllm.add_argument("--use_vllm", action="store_true")
    vllm.add_argument("--vllm_mode", type=str, default="server",
                       choices=["server", "colocate"])
    vllm.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    vllm.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    vllm.add_argument("--vllm_server_host", type=str, default="localhost")
    vllm.add_argument("--vllm_server_port", type=int, default=8000)

    # --- Logging ---
    log = parser.add_argument_group("Logging & Checkpointing")
    log.add_argument("--logging_steps", type=int, default=1)
    log.add_argument("--save_steps", type=int, default=1000)
    log.add_argument("--save_total_limit", type=int, default=1)
    log.add_argument("--report_to", type=str, default="tensorboard",
                      choices=["tensorboard", "wandb", "none"])
    log.add_argument("--run_name", type=str, default=None)

    # --- Evaluation ---
    ev = parser.add_argument_group("Evaluation Configuration")
    ev.add_argument("--do_eval", action="store_true")
    ev.add_argument("--eval_steps", type=int, default=100)
    ev.add_argument("--eval_strategy", type=str, default="steps",
                     choices=["steps", "epoch", "no"])
    ev.add_argument("--per_device_eval_batch_size", type=int, default=1)

    # --- Memory ---
    mem = parser.add_argument_group("Memory Optimization")
    mem.add_argument("--gradient_checkpointing", action="store_true")
    mem.add_argument("--optim", type=str, default="adamw_torch_fused",
                      choices=["adamw_torch", "adamw_torch_fused",
                               "adamw_8bit", "paged_adamw_8bit"])

    return parser


def parse_args() -> argparse.Namespace:
    """Parse arguments with optional YAML config defaults."""
    parser = build_parser()

    # First pass: check if --config is provided
    args, _ = parser.parse_known_args()

    # If config file is provided, load YAML defaults
    if args.config:
        apply_yaml_defaults(parser, args.config)

    # Final parse with YAML defaults applied
    args = parser.parse_args()
    return args


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Setup experiment directory
    if args.output_dir is None:
        run_name = args.run_name or (
            f"popo_{args.model_name.split('/')[-1]}_{args.dataset_name.split('/')[-1]}"
        )
        args.output_dir = os.path.join("logs", run_name)

    experiment_dir = setup_experiment_dir(
        base_dir=os.path.dirname(args.output_dir),
        experiment_name=os.path.basename(args.output_dir),
        args=args,
    )

    # Setup logging
    setup_logging(
        log_dir=os.path.join(experiment_dir, "train_logs"),
        rank=local_rank,
    )

    # Print environment info
    print_env_info(args)

    # Load datasets
    train_dataset, test_dataset = load_and_prepare_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        train_split=args.train_split,
        test_split=args.test_split,
        prompt_template=args.prompt_template,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        do_eval=args.do_eval,
    )

    # Build POPOConfig
    config_kwargs = {
        # Directories
        "output_dir": os.path.join(experiment_dir, "checkpoints"),
        "logging_dir": os.path.join(experiment_dir, "runs"),
        # Training
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optim": args.optim,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "bf16": args.torch_dtype == "bfloat16",
        "fp16": args.torch_dtype == "float16",
        "gradient_checkpointing": args.gradient_checkpointing,
        "seed": args.seed,
        # Generation
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "max_prompt_length": args.max_prompt_length,
        "temperature": args.temperature,
        "num_iterations": args.num_iterations,
        # POPO-specific
        "beta": args.beta,  # We suggest to keep 0.0 for POPO
        "alpha": args.alpha,
        "beta_entropy": args.beta_entropy,
        "tau": args.tau,
        "tau_r": args.tau_r,
        "predictor_hidden_dim": args.predictor_hidden_dim,
        "predictor_layers": args.predictor_layers,
        "feature_noise_std": args.feature_noise_std,
        "kl_monitoring_steps": args.kl_monitoring_steps,
        "weight_mode": args.weight_mode,
        "length_normalize_weights": args.length_normalize_weights,
        # Logging
        "logging_steps": args.logging_steps,
        "report_to": args.report_to if args.report_to != "none" else [],
        # Checkpointing
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "resume_from_checkpoint": args.resume_from_checkpoint,
    }

    if args.run_name:
        config_kwargs["run_name"] = args.run_name

    if args.do_eval and test_dataset is not None:
        config_kwargs["eval_strategy"] = args.eval_strategy
        config_kwargs["eval_steps"] = args.eval_steps
        config_kwargs["per_device_eval_batch_size"] = args.per_device_eval_batch_size

    # vLLM configuration
    if args.use_vllm:
        config_kwargs["use_vllm"] = True
        config_kwargs["vllm_mode"] = args.vllm_mode
        if args.vllm_mode == "colocate":
            config_kwargs["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
            config_kwargs["vllm_tensor_parallel_size"] = args.vllm_tensor_parallel_size
        elif args.vllm_mode == "server":
            config_kwargs["vllm_server_host"] = args.vllm_server_host
            config_kwargs["vllm_server_port"] = args.vllm_server_port

    config_kwargs["model_init_kwargs"] = {
        "attn_implementation": args.attn_implementation,
        # Keep torch_dtype as string; HF's from_pretrained converts it
        # internally. Using the torch.dtype object breaks JSON serialization
        # when TensorBoard logs the training config.
        "torch_dtype": args.torch_dtype,
    }

    config = POPOConfig(**config_kwargs)

    if local_rank == 0:
        logger.info(
            "POPOConfig: alpha=%.3f, beta_entropy=%.3f, tau=%.4f, "
            "tau_r=%.2f, predictor_dim=%d, lr=%s",
            config.alpha,
            config.beta_entropy,
            config.tau,
            config.tau_r,
            config.predictor_hidden_dim,
            config.learning_rate,
        )

    # LoRA config
    lora_config = None
    if args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
        )
        if local_rank == 0:
            logger.info(
                "LoRA: r=%d, alpha=%d, dropout=%.2f",
                lora_config.r,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
            )

    # Create POPO trainer
    if local_rank == 0:
        logger.info("Creating POPO trainer for model: %s", args.model_name)

    trainer = POPOTrainer(
        model=args.model_name,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if args.do_eval else None,
        reward_funcs=math_verify_reward,
        peft_config=lora_config,
    )

    if local_rank == 0:
        logger.info("POPO Trainer created successfully!")
        total = sum(p.numel() for p in trainer.model.parameters())
        trainable = sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        )
        predictor_params = sum(p.numel() for p in trainer.predictor.parameters())
        logger.info(
            "Model params: %s total, %s trainable (%.2f%%)",
            f"{total:,}",
            f"{trainable:,}",
            100 * trainable / total,
        )
        logger.info("Predictor MLP params: %s", f"{predictor_params:,}")

        # Save model summary
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        hidden_size = getattr(
            getattr(unwrapped, "config", None), "hidden_size", "unknown"
        )
        model_summary = {
            "base_model": args.model_name,
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": round(100 * trainable / total, 2),
            "predictor": {
                "input_dim": hidden_size,
                "hidden_dim": args.predictor_hidden_dim,
                "output_dim": hidden_size,
                "num_layers": args.predictor_layers,
                "total_params": predictor_params,
            },
            "ema_tau": args.tau,
            "use_lora": args.use_lora,
            "weight_mode": args.weight_mode,
            "length_normalize_weights": args.length_normalize_weights,
        }
        summary_path = os.path.join(
            experiment_dir, "model_summary", "model_summary.yaml"
        )
        with open(summary_path, "w") as f:
            yaml.dump(model_summary, f, default_flow_style=False, sort_keys=False)
        logger.info("Model summary saved to %s", summary_path)

    # Train
    if local_rank == 0:
        logger.info("Starting POPO training...")

    trainer.train()

    if local_rank == 0:
        logger.info("Training complete!")

    # Save final model
    final_dir = os.path.join(experiment_dir, "checkpoints", "final")
    if local_rank == 0:
        logger.info("Saving final model to %s", final_dir)

    trainer.save_model(final_dir)

    # Under FSDP, state_dict() is a collective — all ranks must call it.
    ema_sd = trainer.ema_target.state_dict()
    predictor_sd = trainer.accelerator.unwrap_model(trainer.predictor).state_dict()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if local_rank == 0:
        tokenizer = trainer.tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(final_dir)

        ema_dir = os.path.join(experiment_dir, "checkpoints", "ema_target")
        os.makedirs(ema_dir, exist_ok=True)
        torch.save(ema_sd, os.path.join(ema_dir, "ema_state.pt"))
        logger.info("EMA target saved to %s", ema_dir)

        predictor_path = os.path.join(experiment_dir, "checkpoints", "predictor.pt")
        torch.save(predictor_sd, predictor_path)
        logger.info("Predictor saved to %s", predictor_path)

        logger.info("Done!")


if __name__ == "__main__":
    main()
