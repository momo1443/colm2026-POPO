# """
# vLLM-based Pass@k evaluator for math reasoning tasks.

# Provides efficient batched generation and evaluation using vLLM,
# with the PassKEvaluatorVLLM class and a convenience wrapper.

# Canonical source: test_environment/passk_evaluator.py lines 224-672.
# Imports shared utilities from src.reward and src.data (no local copies).
# """

# import os
# import json
# import logging
# from typing import List, Dict, Any, Callable, Optional

# import numpy as np
# from tqdm import tqdm

# from src.evaluation.passk import compute_pass_at_k_batch
# from src.evaluation.avek import compute_ave_at_k_batch
# from src.reward.math_verify_reward import math_verify_reward
# from src.data.templates import TEMPLATE_CONFIGS, get_format_function, detect_dataset_type

# logger = logging.getLogger(__name__)


# class PassKEvaluatorVLLM:
#     """
#     Pass@k evaluator using vLLM for efficient batched generation.

#     Example::

#         evaluator = PassKEvaluatorVLLM(
#             model_name_or_path="Qwen/Qwen2.5-Math-7B",
#             reward_fn=math_verify_reward,
#         )
#         results = evaluator.evaluate(
#             dataset=test_dataset,
#             n_samples=10,
#             k_values=[1, 5, 10],
#             temperature=0.7,
#         )
#     """

#     def __init__(
#         self,
#         model_name_or_path: str,
#         reward_fn: Callable,
#         tokenizer_name_or_path: Optional[str] = None,
#         tensor_parallel_size: int = 1,
#         gpu_memory_utilization: float = 0.9,
#         max_model_len: int = 4096,
#         trust_remote_code: bool = True,
#     ):
#         """
#         Initialize the vLLM-based evaluator.

#         Args:
#             model_name_or_path: HuggingFace model name or local path.
#             reward_fn: Reward function with signature
#                 (completions: List[str], solutions: List[str]) -> List[float].
#             tokenizer_name_or_path: Optional separate tokenizer path.
#             tensor_parallel_size: Number of GPUs for tensor parallelism.
#             gpu_memory_utilization: Fraction of GPU memory to use.
#             max_model_len: Maximum sequence length.
#             trust_remote_code: Whether to trust remote code.
#         """
#         try:
#             from vllm import LLM, SamplingParams
#         except ImportError:
#             raise ImportError(
#                 "vLLM is required for Pass@k evaluation but is not installed. "
#                 "Install with: pip install vllm"
#             )

#         self._SamplingParams = SamplingParams

#         tokenizer_path = tokenizer_name_or_path or model_name_or_path

#         logger.info("Loading model: %s", model_name_or_path)
#         logger.info("Tokenizer: %s", tokenizer_path)
#         logger.info("Tensor parallel size: %d", tensor_parallel_size)

#         self.model = LLM(
#             model=model_name_or_path,
#             tokenizer=tokenizer_path,
#             tensor_parallel_size=tensor_parallel_size,
#             gpu_memory_utilization=gpu_memory_utilization,
#             max_model_len=max_model_len,
#             trust_remote_code=trust_remote_code,
#         )
#         self.reward_fn = reward_fn
#         self.model_name = model_name_or_path
#         logger.info("Model loaded successfully!")

#     def evaluate(
#         self,
#         dataset,
#         n_samples: int = 10,
#         k_values: Optional[List[int]] = None,
#         temperature: float = 0.7,
#         top_p: float = 0.95,
#         max_tokens: int = 4096,
#         batch_size: int = 256,
#         show_progress: bool = True,
#         save_generations: bool = False,
#         output_path: Optional[str] = None,
#     ) -> Dict[str, Any]:
#         """
#         Evaluate Pass@k using vLLM batched generation.

#         Args:
#             dataset: HuggingFace dataset with 'prompt' and 'solution' columns.
#             n_samples: Number of samples to generate per problem.
#             k_values: List of k values for Pass@k (default: [1, 5, 10]).
#             temperature: Sampling temperature.
#             top_p: Top-p sampling parameter.
#             max_tokens: Maximum tokens per completion.
#             batch_size: Prompts per vLLM batch.
#             show_progress: Whether to show progress bar.
#             save_generations: Whether to save all generations to file.
#             output_path: Path to save generations (if save_generations=True).

#         Returns:
#             Dict with Pass@k and Ave@k metrics and detailed statistics.
#         """
#         if k_values is None:
#             k_values = [1, 5, 10]

#         # Filter k_values to those <= n_samples
#         k_values = [k for k in k_values if k <= n_samples]
#         if not k_values:
#             raise ValueError(f"All k_values must be <= n_samples ({n_samples})")

#         prompts = [ex["prompt"] for ex in dataset]
#         solutions = [ex["solution"] for ex in dataset]
#         num_problems = len(prompts)

#         logger.info(
#             "Pass@k evaluation: %d problems, %d samples/problem, k=%s",
#             num_problems,
#             n_samples,
#             k_values,
#         )

#         # Expand prompts: each prompt repeated n_samples times
#         all_prompts = []
#         for prompt in prompts:
#             all_prompts.extend([prompt] * n_samples)

#         total_generations = len(all_prompts)

#         # Configure sampling
#         sampling_params = self._SamplingParams(
#             temperature=temperature,
#             top_p=top_p,
#             max_tokens=max_tokens,
#             n=1,
#         )

#         # Generate in batches
#         all_completions = []
#         for batch_start in tqdm(
#             range(0, total_generations, batch_size),
#             desc="Generating",
#             disable=not show_progress,
#         ):
#             batch_end = min(batch_start + batch_size, total_generations)
#             batch_prompts = all_prompts[batch_start:batch_end]

#             outputs = self.model.generate(batch_prompts, sampling_params)

#             for output in outputs:
#                 all_completions.append(output.outputs[0].text)

#         # Evaluate rewards per problem
#         results = []
#         all_rewards = []
#         per_problem_stats = []
#         all_generations_data = []

#         for prob_idx in tqdm(
#             range(num_problems), desc="Evaluating", disable=not show_progress
#         ):
#             start_idx = prob_idx * n_samples
#             end_idx = start_idx + n_samples

#             problem_completions = all_completions[start_idx:end_idx]
#             problem_solution = solutions[prob_idx]
#             problem_solutions = [problem_solution] * n_samples

#             rewards = self.reward_fn(problem_completions, problem_solutions)
#             c = sum(1 for r in rewards if r > 0.5)

#             results.append({"n": n_samples, "c": c})
#             all_rewards.extend(rewards)

#             per_problem_stats.append(
#                 {
#                     "problem_idx": prob_idx,
#                     "n_correct": c,
#                     "n_samples": n_samples,
#                     "accuracy": c / n_samples,
#                 }
#             )

#             if save_generations:
#                 all_generations_data.append(
#                     {
#                         "problem_idx": prob_idx,
#                         "prompt": prompts[prob_idx],
#                         "solution": problem_solution,
#                         "completions": problem_completions,
#                         "rewards": rewards,
#                         "n_correct": c,
#                     }
#                 )

#         # Compute Pass@k and Ave@k
#         pass_k_metrics = compute_pass_at_k_batch(results, k_values)
#         ave_k_metrics = compute_ave_at_k_batch(results, k_values)

#         # Aggregate statistics
#         accuracies = [stat["accuracy"] for stat in per_problem_stats]
#         final_results = {
#             **pass_k_metrics,
#             **ave_k_metrics,
#             "mean_reward": float(np.mean(all_rewards)),
#             "std_reward": float(np.std(all_rewards)),
#             "mean_accuracy": float(np.mean(accuracies)),
#             "std_accuracy": float(np.std(accuracies)),
#             "total_problems": num_problems,
#             "n_samples_per_problem": n_samples,
#             "temperature": temperature,
#             "total_correct": sum(r["c"] for r in results),
#             "total_samples": num_problems * n_samples,
#         }

#         # Save generations if requested
#         if save_generations and output_path:
#             gen_path = (
#                 output_path
#                 if output_path.endswith(".json")
#                 else f"{output_path}_generations.json"
#             )
#             with open(gen_path, "w") as f:
#                 json.dump(all_generations_data, f, indent=2)
#             logger.info("Generations saved to: %s", gen_path)

#         # Log results
#         logger.info("Pass@k Results:")
#         for k in k_values:
#             pct = pass_k_metrics[f"pass@{k}"] * 100
#             logger.info("  Pass@%d: %.4f (%.2f%%)", k, pass_k_metrics[f"pass@{k}"], pct)
#         logger.info("Ave@k Results:")
#         for k in k_values:
#             pct = ave_k_metrics[f"ave@{k}"] * 100
#             logger.info("  Ave@%d: %.4f (%.2f%%)", k, ave_k_metrics[f"ave@{k}"], pct)
#         logger.info(
#             "  Mean Accuracy: %.4f (%.2f%%)",
#             final_results["mean_accuracy"],
#             final_results["mean_accuracy"] * 100,
#         )

#         return final_results

#     def evaluate_greedy(
#         self,
#         dataset,
#         max_tokens: int = 4096,
#         batch_size: int = 256,
#         show_progress: bool = True,
#     ) -> Dict[str, Any]:
#         """
#         Evaluate with greedy decoding (temperature=0) for Pass@1.

#         Equivalent to standard accuracy evaluation.
#         """
#         return self.evaluate(
#             dataset=dataset,
#             n_samples=1,
#             k_values=[1],
#             temperature=0.0,
#             max_tokens=max_tokens,
#             batch_size=batch_size,
#             show_progress=show_progress,
#         )


# def run_pass_k_evaluation(
#     model_path: str,
#     dataset_name: str = "openai/gsm8k",
#     dataset_config: str = "main",
#     split: str = "test",
#     template: str = "qwen_math",
#     n_samples: int = 10,
#     k_values: Optional[List[int]] = None,
#     temperature: float = 0.7,
#     max_tokens: int = 4096,
#     tensor_parallel_size: int = 1,
#     gpu_memory_utilization: float = 0.9,
#     max_model_len: int = 4096,
#     tokenizer_path: Optional[str] = None,
#     reward_fn: Optional[Callable] = None,
#     save_results: bool = True,
#     output_dir: Optional[str] = None,
#     eval_greedy: bool = True,
# ) -> Dict[str, Any]:
#     """
#     Convenience function to run Pass@k evaluation end-to-end.

#     Loads dataset, creates evaluator, runs evaluation, and saves results.

#     Args:
#         model_path: Path to model or HuggingFace model name.
#         dataset_name: Dataset name on HuggingFace.
#         dataset_config: Dataset configuration.
#         split: Dataset split to evaluate.
#         template: Prompt template to use.
#         n_samples: Number of samples per problem.
#         k_values: List of k values for Pass@k (default: [1, 5, 10]).
#         temperature: Sampling temperature.
#         max_tokens: Maximum tokens to generate.
#         tensor_parallel_size: Number of GPUs for tensor parallelism.
#         gpu_memory_utilization: GPU memory fraction to use.
#         max_model_len: Maximum sequence length.
#         tokenizer_path: Optional separate tokenizer path.
#         reward_fn: Reward function (defaults to math_verify_reward).
#         save_results: Whether to save results to JSON file.
#         output_dir: Directory to save results.
#         eval_greedy: Whether to also run greedy evaluation.

#     Returns:
#         Dict with evaluation results.
#     """
#     from datasets import load_dataset

#     if k_values is None:
#         k_values = [1, 5, 10]

#     # Load and format dataset
#     # logger.info("Loading dataset: %s (%s), split: %s", dataset_name, dataset_config, split)
#     # dataset = load_dataset(dataset_name, dataset_config, split=split)

#     # dataset_type = "gsm8k" if "gsm8k" in dataset_name.lower() else "math"
#     # format_fn = get_format_function(template, dataset_type)
#     # dataset = dataset.map(format_fn, remove_columns=dataset.column_names)

#     logger.info("Loading dataset: %s (config=%s), split: %s", dataset_name, dataset_config, split)
#     if dataset_config and dataset_config.lower() not in ("none", "null", ""):
#         dataset = load_dataset(dataset_name, dataset_config, split=split)
#     else:
#         dataset = load_dataset(dataset_name, split=split)
#     dataset_type = detect_dataset_type(dataset_name)
#     logger.info("Detected dataset type: %s (columns: %s)", dataset_type, dataset.column_names)

#     format_fn = get_format_function(template, dataset_type)
#     dataset = dataset.map(format_fn, remove_columns=dataset.column_names)


#     logger.info("Dataset size: %d problems", len(dataset))

#     # Use default reward function if not provided
#     if reward_fn is None:
#         reward_fn = math_verify_reward

#     # Create evaluator
#     evaluator = PassKEvaluatorVLLM(
#         model_name_or_path=model_path,
#         reward_fn=reward_fn,
#         tokenizer_name_or_path=tokenizer_path,
#         tensor_parallel_size=tensor_parallel_size,
#         gpu_memory_utilization=gpu_memory_utilization,
#         max_model_len=max_model_len,
#     )

#     results = {}

#     # Greedy evaluation
#     if eval_greedy:
#         logger.info("Running greedy evaluation (Pass@1)")
#         greedy_results = evaluator.evaluate_greedy(
#             dataset=dataset, max_tokens=max_tokens
#         )
#         results["greedy"] = greedy_results

#     # Sampling-based evaluation
#     if n_samples > 1:
#         logger.info("Running sampling-based Pass@k (n=%d)", n_samples)
#         sampling_results = evaluator.evaluate(
#             dataset=dataset,
#             n_samples=n_samples,
#             k_values=k_values,
#             temperature=temperature,
#             max_tokens=max_tokens,
#         )
#         results["sampling"] = sampling_results

#     # Save results
#     if save_results:
#         output_dir = output_dir or os.path.dirname(model_path) or "."
#         results_path = os.path.join(output_dir, "pass_k_results.json")
#         with open(results_path, "w") as f:
#             json.dump(results, f, indent=2)
#         logger.info("Results saved to: %s", results_path)

#     return results


"""
vLLM-based Pass@k evaluator for math reasoning tasks.

Provides efficient batched generation and evaluation using vLLM,
with the PassKEvaluatorVLLM class and a convenience wrapper.

Canonical source: test_environment/passk_evaluator.py lines 224-672.
Imports shared utilities from src.reward and src.data (no local copies).
"""

import os
import json
import random
import logging
from typing import List, Dict, Any, Callable, Optional, Set

import numpy as np
from tqdm import tqdm

from src.evaluation.passk import compute_pass_at_k_batch
from src.evaluation.avek import compute_ave_at_k_batch
from src.reward.math_verify_reward import math_verify_reward
from src.data.templates import TEMPLATE_CONFIGS, get_format_function, detect_dataset_type

logger = logging.getLogger(__name__)


class PassKEvaluatorVLLM:
    """
    Pass@k evaluator using vLLM for efficient batched generation.

    Example::

        evaluator = PassKEvaluatorVLLM(
            model_name_or_path="Qwen/Qwen2.5-Math-7B",
            reward_fn=math_verify_reward,
        )
        results = evaluator.evaluate(
            dataset=test_dataset,
            n_samples=10,
            k_values=[1, 5, 10],
            temperature=0.7,
        )
    """

    def __init__(
        self,
        model_name_or_path: str,
        reward_fn: Callable,
        tokenizer_name_or_path: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        trust_remote_code: bool = True,
    ):
        """
        Initialize the vLLM-based evaluator.

        Args:
            model_name_or_path: HuggingFace model name or local path.
            reward_fn: Reward function with signature
                (completions: List[str], solutions: List[str]) -> List[float].
            tokenizer_name_or_path: Optional separate tokenizer path.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use.
            max_model_len: Maximum sequence length.
            trust_remote_code: Whether to trust remote code.
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is required for Pass@k evaluation but is not installed. "
                "Install with: pip install vllm"
            )

        self._SamplingParams = SamplingParams

        tokenizer_path = tokenizer_name_or_path or model_name_or_path

        logger.info("Loading model: %s", model_name_or_path)
        logger.info("Tokenizer: %s", tokenizer_path)
        logger.info("Tensor parallel size: %d", tensor_parallel_size)

        self.model = LLM(
            model=model_name_or_path,
            tokenizer=tokenizer_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
        )
        self.reward_fn = reward_fn
        self.model_name = model_name_or_path
        logger.info("Model loaded successfully!")

    def evaluate(
        self,
        dataset,
        n_samples: int = 10,
        k_values: Optional[List[int]] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        batch_size: int = 256,
        show_progress: bool = True,
        inspect_indices: Optional[Set[int]] = None,
        raw_questions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate Pass@k using vLLM batched generation.

        Runs on ALL problems for metrics. Optionally collects detailed
        generation data for a subset of problems (inspect_indices) for
        qualitative inspection.

        Args:
            dataset: HuggingFace dataset with 'prompt' and 'solution' columns.
            n_samples: Number of samples to generate per problem.
            k_values: List of k values for Pass@k (default: [1, 5, 10]).
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            max_tokens: Maximum tokens per completion.
            batch_size: Prompts per vLLM batch.
            show_progress: Whether to show progress bar.
            inspect_indices: Set of problem indices to collect full
                generation details for. None or empty to skip.
            raw_questions: Original question texts (before template
                formatting), aligned with dataset order. Used to populate
                a human-readable 'question' field in inspection output.

        Returns:
            Dict with Pass@k / Ave@k metrics. When inspect_indices is
            provided, also contains an 'inspections' key with detailed
            per-problem generation data.
        """
        if k_values is None:
            k_values = [1, 5, 10]

        k_values = [k for k in k_values if k <= n_samples]
        if not k_values:
            raise ValueError(f"All k_values must be <= n_samples ({n_samples})")

        prompts = [ex["prompt"] for ex in dataset]
        solutions = [ex["solution"] for ex in dataset]
        num_problems = len(prompts)
        inspect_indices = inspect_indices or set()

        logger.info(
            "Pass@k evaluation: %d problems, %d samples/problem, k=%s",
            num_problems,
            n_samples,
            k_values,
        )

        # Expand prompts: each prompt repeated n_samples times
        all_prompts = []
        for prompt in prompts:
            all_prompts.extend([prompt] * n_samples)

        total_generations = len(all_prompts)

        sampling_params = self._SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=1,
        )

        # Generate in batches
        all_completions = []
        for batch_start in tqdm(
            range(0, total_generations, batch_size),
            desc="Generating",
            disable=not show_progress,
        ):
            batch_end = min(batch_start + batch_size, total_generations)
            batch_prompts = all_prompts[batch_start:batch_end]

            outputs = self.model.generate(batch_prompts, sampling_params)

            for output in outputs:
                all_completions.append(output.outputs[0].text)

        # Evaluate rewards per problem
        results = []
        all_rewards = []
        per_problem_stats = []
        inspection_data = []

        for prob_idx in tqdm(
            range(num_problems), desc="Evaluating", disable=not show_progress
        ):
            start_idx = prob_idx * n_samples
            end_idx = start_idx + n_samples

            problem_completions = all_completions[start_idx:end_idx]
            problem_solution = solutions[prob_idx]
            problem_solutions = [problem_solution] * n_samples

            rewards = self.reward_fn(problem_completions, problem_solutions)
            c = sum(1 for r in rewards if r > 0.5)

            results.append({"n": n_samples, "c": c})
            all_rewards.extend(rewards)

            per_problem_stats.append(
                {
                    "problem_idx": prob_idx,
                    "n_correct": c,
                    "n_samples": n_samples,
                    "accuracy": c / n_samples,
                }
            )

            if prob_idx in inspect_indices:
                entry = {
                    "problem_idx": prob_idx,
                    "prompt": prompts[prob_idx],
                    "ground_truth": problem_solution,
                    "n_correct": c,
                    "n_samples": n_samples,
                    "accuracy": c / n_samples,
                    "completions": [
                        {"text": comp, "correct": rew > 0.5}
                        for comp, rew in zip(problem_completions, rewards)
                    ],
                }
                if raw_questions is not None:
                    entry["question"] = raw_questions[prob_idx]
                inspection_data.append(entry)

        # Compute Pass@k and Ave@k
        pass_k_metrics = compute_pass_at_k_batch(results, k_values)
        ave_k_metrics = compute_ave_at_k_batch(results, k_values)

        # Aggregate statistics
        accuracies = [stat["accuracy"] for stat in per_problem_stats]
        final_results = {
            **pass_k_metrics,
            **ave_k_metrics,
            "mean_reward": float(np.mean(all_rewards)),
            "std_reward": float(np.std(all_rewards)),
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "total_problems": num_problems,
            "n_samples_per_problem": n_samples,
            "temperature": temperature,
            "total_correct": sum(r["c"] for r in results),
            "total_samples": num_problems * n_samples,
        }

        if inspection_data:
            final_results["inspections"] = inspection_data

        # Log results
        logger.info("Pass@k Results:")
        for k in k_values:
            pct = pass_k_metrics[f"pass@{k}"] * 100
            logger.info("  Pass@%d: %.4f (%.2f%%)", k, pass_k_metrics[f"pass@{k}"], pct)
        logger.info("Ave@k Results:")
        for k in k_values:
            pct = ave_k_metrics[f"ave@{k}"] * 100
            logger.info("  Ave@%d: %.4f (%.2f%%)", k, ave_k_metrics[f"ave@{k}"], pct)
        logger.info(
            "  Mean Accuracy: %.4f (%.2f%%)",
            final_results["mean_accuracy"],
            final_results["mean_accuracy"] * 100,
        )

        return final_results

    def evaluate_greedy(
        self,
        dataset,
        max_tokens: int = 4096,
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate with greedy decoding (temperature=0) for Pass@1.

        Equivalent to standard accuracy evaluation.
        """
        return self.evaluate(
            dataset=dataset,
            n_samples=1,
            k_values=[1],
            temperature=0.0,
            max_tokens=max_tokens,
            batch_size=batch_size,
            show_progress=show_progress,
        )


def run_pass_k_evaluation(
    model_path: str,
    dataset_name: str = "openai/gsm8k",
    dataset_config: str = "main",
    split: str = "test",
    template: str = "qwen_math",
    n_samples: int = 10,
    k_values: Optional[List[int]] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
    tokenizer_path: Optional[str] = None,
    reward_fn: Optional[Callable] = None,
    save_results: bool = True,
    output_dir: Optional[str] = None,
    eval_greedy: bool = True,
    inspect_samples: int = 5,
) -> Dict[str, Any]:
    """
    Convenience function to run Pass@k evaluation end-to-end.

    Evaluates on the FULL dataset for metrics. Optionally saves a
    human-readable JSON with detailed generation outputs for a random
    subset of problems (controlled by inspect_samples).

    Args:
        model_path: Path to model or HuggingFace model name.
        dataset_name: Dataset name on HuggingFace.
        dataset_config: Dataset configuration.
        split: Dataset split to evaluate.
        template: Prompt template to use.
        n_samples: Number of samples per problem.
        k_values: List of k values for Pass@k (default: [1, 5, 10]).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: GPU memory fraction to use.
        max_model_len: Maximum sequence length.
        tokenizer_path: Optional separate tokenizer path.
        reward_fn: Reward function (defaults to math_verify_reward).
        save_results: Whether to save results to JSON file.
        output_dir: Directory to save results.
        eval_greedy: Whether to also run greedy evaluation.
        inspect_samples: Number of random problems to dump full
            generation details for manual inspection. 0 to disable.
    """
    from datasets import load_dataset
    from src.data.templates import DATASET_FIELD_CONFIGS

    if k_values is None:
        k_values = [1, 5, 10]

    logger.info("Loading dataset: %s (config=%s), split: %s", dataset_name, dataset_config, split)
    if dataset_config and dataset_config.lower() not in ("none", "null", ""):
        raw_dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        raw_dataset = load_dataset(dataset_name, split=split)

    dataset_type = detect_dataset_type(dataset_name)
    logger.info("Detected dataset type: %s (columns: %s)", dataset_type, raw_dataset.column_names)

    # Extract raw question texts before template formatting (for readable inspection output)
    raw_questions = None
    if inspect_samples > 0:
        field_cfg = DATASET_FIELD_CONFIGS.get(dataset_type, DATASET_FIELD_CONFIGS["math"])
        q_field = field_cfg.question_field
        # Case-insensitive lookup
        actual_q_field = None
        for col in raw_dataset.column_names:
            if col.lower() == q_field.lower():
                actual_q_field = col
                break
        if actual_q_field:
            raw_questions = [str(ex[actual_q_field]) for ex in raw_dataset]

    format_fn = get_format_function(template, dataset_type)
    dataset = raw_dataset.map(format_fn, remove_columns=raw_dataset.column_names)

    logger.info("Dataset size: %d problems", len(dataset))

    if reward_fn is None:
        reward_fn = math_verify_reward

    evaluator = PassKEvaluatorVLLM(
        model_name_or_path=model_path,
        reward_fn=reward_fn,
        tokenizer_name_or_path=tokenizer_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

    # Pick random problem indices for inspection
    inspect_indices: Set[int] = set()
    if inspect_samples > 0:
        n_inspect = min(inspect_samples, len(dataset))
        inspect_indices = set(random.sample(range(len(dataset)), n_inspect))

    results = {}

    # Greedy evaluation (full dataset)
    if eval_greedy:
        logger.info("Running greedy evaluation (Pass@1)")
        greedy_results = evaluator.evaluate_greedy(
            dataset=dataset, max_tokens=max_tokens
        )
        results["greedy"] = greedy_results

    # Sampling-based evaluation (full dataset)
    if n_samples > 1:
        logger.info("Running sampling-based Pass@k (n=%d)", n_samples)
        sampling_results = evaluator.evaluate(
            dataset=dataset,
            n_samples=n_samples,
            k_values=k_values,
            temperature=temperature,
            max_tokens=max_tokens,
            inspect_indices=inspect_indices,
            raw_questions=raw_questions,
        )
        results["sampling"] = sampling_results

    output_dir = output_dir or os.path.dirname(model_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Save inspection JSON (sampled problem details for manual review)
    inspections = results.get("sampling", {}).pop("inspections", None)
    if inspections:
        inspect_path = os.path.join(output_dir, "inspect_samples.json")
        with open(inspect_path, "w", encoding="utf-8") as f:
            json.dump(inspections, f, indent=2, ensure_ascii=False)
        logger.info(
            "Inspection samples (%d problems) saved to: %s",
            len(inspections), inspect_path,
        )

    # Save metrics (no bulky generation data)
    if save_results:
        results_path = os.path.join(output_dir, "pass_k_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to: %s", results_path)

    return results
