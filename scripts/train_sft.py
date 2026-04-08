#!/usr/bin/env python3
"""
SFT Baseline Training Script.

Professional training script for Supervised Fine-Tuning (SFT)
using TRL's SFTTrainer. Supports argparse CLI with optional YAML config
defaults.

Usage:
    # With YAML config + CLI overrides:
    python scripts/train_sft.py --config configs/sft/default.yaml \
        --model_name Qwen/Qwen2.5-Math-7B --num_train_epochs 3

    # With accelerate (FSDP):
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
        --config_file=configs/accelerate/fsdp.yaml \
        scripts/train_sft.py \
        --config configs/sft/default.yaml \
        --model_name Qwen/Qwen2.5-Math-7B \
        --use_lora --do_eval
"""

import os
import sys
import argparse
import logging

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.templates import TEMPLATE_CONFIGS
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
        description="SFT Training Script",
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

    # --- SFT-specific dataset fields ---
    sft_data = parser.add_argument_group("SFT Dataset Field Configuration")
    sft_data.add_argument("--sft_question_field", type=str, default="problem",
                           help="Field name for the question/problem in the dataset")
    sft_data.add_argument("--sft_answer_field", type=str, default="answer",
                           help="Field name for the answer in the dataset")

    # --- Template ---
    tmpl = parser.add_argument_group("Template Configuration")
    tmpl.add_argument("--prompt_template", type=str, default="qwen_math",
                       choices=list(TEMPLATE_CONFIGS.keys()))

    # --- Training ---
    train = parser.add_argument_group("Training Configuration")
    train.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: auto-generated under logs/)")
    train.add_argument("--max_steps", type=int, default=-1,
                        help="Max training steps (-1 = use num_train_epochs)")
    train.add_argument("--num_train_epochs", type=float, default=3.0)
    train.add_argument("--per_device_train_batch_size", type=int, default=2)
    train.add_argument("--per_device_eval_batch_size", type=int, default=2)
    train.add_argument("--gradient_accumulation_steps", type=int, default=8)
    train.add_argument("--learning_rate", type=float, default=2e-5)
    train.add_argument("--warmup_ratio", type=float, default=0.1)
    train.add_argument("--lr_scheduler_type", type=str, default="cosine")
    train.add_argument("--weight_decay", type=float, default=0.01)
    train.add_argument("--max_grad_norm", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length for SFT")
    train.add_argument("--packing", action="store_true",
                        help="Enable sequence packing for efficiency")

    # --- LoRA ---
    lora = parser.add_argument_group("LoRA Configuration")
    lora.add_argument("--use_lora", action="store_true")
    lora.add_argument("--lora_r", type=int, default=16)
    lora.add_argument("--lora_alpha", type=int, default=32)
    lora.add_argument("--lora_dropout", type=float, default=0.05)
    lora.add_argument("--lora_target_modules", type=str, nargs="+",
                       default=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"])

    # --- Logging ---
    log = parser.add_argument_group("Logging & Checkpointing")
    log.add_argument("--logging_steps", type=int, default=10)
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
            f"sft_{args.model_name.split('/')[-1]}_{args.dataset_name.split('/')[-1]}"
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

    # =========================================================================
    # Load raw dataset directly (bypass load_and_prepare_dataset for SFT)
    #
    # Why: load_and_prepare_dataset uses DATASET_FIELD_CONFIGS which maps
    # deepscaler's answer_field to "answer" (the short answer). For SFT we
    # need the "solution" field (long-form reasoning), so we load the raw
    # dataset and build prompt+completion ourselves.
    # =========================================================================
    if local_rank == 0:
        logger.info("Loading dataset: %s (%s)", args.dataset_name, args.dataset_config)
        logger.info("Using template: %s (%s)",
                     args.prompt_template, TEMPLATE_CONFIGS[args.prompt_template].name)
        logger.info("SFT fields: question='%s', answer='%s'",
                     args.sft_question_field, args.sft_answer_field)

    # Load raw training dataset
    if args.dataset_config:
        raw_train = load_dataset(args.dataset_name, args.dataset_config, split=args.train_split)
    else:
        raw_train = load_dataset(args.dataset_name, split=args.train_split)

    # Load raw test dataset if requested
    raw_test = None
    if args.do_eval and args.test_split:
        try:
            if args.dataset_config:
                raw_test = load_dataset(args.dataset_name, args.dataset_config, split=args.test_split)
            else:
                raw_test = load_dataset(args.dataset_name, split=args.test_split)
        except Exception as e:
            if local_rank == 0:
                logger.warning("Could not load test split '%s': %s", args.test_split, e)

    # Limit samples if specified
    if args.max_train_samples is not None:
        n = min(args.max_train_samples, len(raw_train))
        raw_train = raw_train.select(range(n))
    if raw_test is not None and args.max_test_samples is not None:
        n = min(args.max_test_samples, len(raw_test))
        raw_test = raw_test.select(range(n))

    # Build prompt+completion using template config
    template_config = TEMPLATE_CONFIGS[args.prompt_template]
    q_field = args.sft_question_field
    a_field = args.sft_answer_field

    def format_for_sft(example):
        question = example[q_field]
        solution = example[a_field]
        if not isinstance(solution, str):
            solution = str(solution)

        # Build prompt using the same logic as format_with_template in templates.py
        parts = []
        if template_config.system_prompt:
            if args.prompt_template == "qwen_math":
                parts.append(f"<|im_start|>system\n{template_config.system_prompt}<|im_end|>\n")
            elif args.prompt_template == "llama":
                parts.append(
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                    f"\n\n{template_config.system_prompt}<|eot_id|>"
                )
            elif args.prompt_template == "r1":
                parts.append(f"{template_config.system_prompt}\n")
            elif args.prompt_template == "deepseek":
                parts.append(f"{template_config.system_prompt}\n\n")
        parts.append(f"{template_config.user_prefix}{question}{template_config.user_suffix}")
        parts.append(template_config.assistant_prefix)
        prompt = "".join(parts)

        # Completion = long-form solution + assistant suffix token
        completion = f"{solution}{template_config.assistant_suffix}"

        return {"prompt": prompt, "completion": completion}

    # Apply formatting, remove all original columns, keep only prompt + completion
    train_dataset = raw_train.map(format_for_sft, remove_columns=raw_train.column_names)
    test_dataset = None
    if raw_test is not None:
        test_dataset = raw_test.map(format_for_sft, remove_columns=raw_test.column_names)

    if local_rank == 0:
        logger.info("Train dataset: %d samples", len(train_dataset))
        if test_dataset is not None:
            logger.info("Test dataset: %d samples", len(test_dataset))
        logger.info("Example prompt (first 500 chars):\n%s", train_dataset[0]["prompt"][:500])
        logger.info("Example completion (first 300 chars):\n%s", train_dataset[0]["completion"][:300])

    # =========================================================================
    # Build SFTConfig
    # =========================================================================
    config_kwargs = {
        "output_dir": os.path.join(experiment_dir, "checkpoints"),
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_seq_length,
        "packing": args.packing,
        "optim": args.optim,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "bf16": args.torch_dtype == "bfloat16",
        "fp16": args.torch_dtype == "float16",
        "gradient_checkpointing": args.gradient_checkpointing,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": args.report_to if args.report_to != "none" else [],
        "logging_dir": os.path.join(experiment_dir, "runs"),
        "logging_steps": args.logging_steps,
        "seed": args.seed,
    }

    if args.run_name:
        config_kwargs["run_name"] = args.run_name

    if args.do_eval and test_dataset is not None:
        config_kwargs["eval_strategy"] = args.eval_strategy
        config_kwargs["eval_steps"] = args.eval_steps

    config_kwargs["model_init_kwargs"] = {
        "torch_dtype": args.torch_dtype,
        "attn_implementation": args.attn_implementation,
    }

    config = SFTConfig(**config_kwargs)

    if local_rank == 0:
        logger.info("SFTConfig: epochs=%s, max_steps=%s, lr=%s, max_length=%d, packing=%s",
                     config.num_train_epochs, config.max_steps,
                     config.learning_rate, config.max_length, config.packing)

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
            logger.info("LoRA: r=%d, alpha=%d, dropout=%.2f",
                         lora_config.r, lora_config.lora_alpha,
                         lora_config.lora_dropout)

    # Create trainer
    if local_rank == 0:
        logger.info("Creating SFT trainer for model: %s", args.model_name)

    trainer = SFTTrainer(
        model=args.model_name,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if args.do_eval else None,
        peft_config=lora_config,
    )

    if local_rank == 0:
        logger.info("Trainer created successfully!")
        if hasattr(trainer.model, "print_trainable_parameters"):
            trainer.model.print_trainable_parameters()
        else:
            total = sum(p.numel() for p in trainer.model.parameters())
            trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            logger.info("Trainable params: %s / %s (%.2f%%)",
                         f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    # Train
    if local_rank == 0:
        logger.info("Starting SFT training...")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if local_rank == 0:
        logger.info("Training complete!")

    # Save final model
    final_dir = os.path.join(experiment_dir, "checkpoints", "final")
    if local_rank == 0:
        logger.info("Saving final model to %s", final_dir)

    trainer.save_model(final_dir)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if local_rank == 0:
        tokenizer = trainer.processing_class
        if tokenizer is not None:
            tokenizer.save_pretrained(final_dir)
        logger.info("Done!")


if __name__ == "__main__":
    main()