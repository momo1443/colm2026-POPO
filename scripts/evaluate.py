#!/usr/bin/env python3
"""
Pass@k Evaluation Script.

Evaluates trained models on math reasoning benchmarks using vLLM
for efficient batched generation.

Usage:
    python scripts/evaluate.py \
        --model_path logs/grpo_run/checkpoints/final \
        --dataset_name openai/gsm8k \
        --template qwen_math \
        --n_samples 10 --k_values 1 5 10 \
        --temperature 0.7 --eval_greedy
"""

import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.templates import TEMPLATE_CONFIGS
from src.evaluation.evaluator import run_pass_k_evaluation
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pass@k Evaluation for Math Reasoning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model or HuggingFace model name")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Optional separate tokenizer path")

    # Dataset
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset configuration (e.g., 'main' for gsm8k, omit for AIME)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--template", type=str, default="qwen_math",
                        choices=list(TEMPLATE_CONFIGS.keys()))

    # Evaluation
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--eval_greedy", action="store_true")

    # Inspection
    parser.add_argument("--inspect_samples", type=int, default=10,
                        help="Number of random problems to dump full generation "
                             "details for manual inspection (0 to disable)")

    # vLLM
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=4096)

    # Output
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_results", action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()

    results = run_pass_k_evaluation(
        model_path=args.model_path,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        template=args.template,
        n_samples=args.n_samples,
        k_values=args.k_values,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tokenizer_path=args.tokenizer_path,
        save_results=args.save_results,
        output_dir=args.output_dir,
        eval_greedy=args.eval_greedy,
        inspect_samples=args.inspect_samples,
    )

    return results


if __name__ == "__main__":
    main()
