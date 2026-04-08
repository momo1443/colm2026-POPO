"""
Dataset loading and preparation for math reasoning tasks.

Supports:
- openai/gsm8k (grade school math)
- MATH (competition math)
- Other HuggingFace datasets with question/answer fields

Canonical source: test_environment/train_grpo_v2.py lines 680-736.
"""

import os
import logging
from typing import Optional, Tuple

from datasets import load_dataset, Dataset

# from src.data.templates import TEMPLATE_CONFIGS, get_format_function
from src.data.templates import TEMPLATE_CONFIGS, get_format_function, detect_dataset_type

logger = logging.getLogger(__name__)


def load_and_prepare_dataset(
    dataset_name: str = "openai/gsm8k",
    dataset_config: str = "main",
    train_split: str = "train",
    test_split: Optional[str] = "test",
    prompt_template: str = "qwen_math",
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    do_eval: bool = False,
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load and format training (and optionally test) datasets.

    Applies the specified prompt template to each example, producing
    'prompt' and 'solution' columns suitable for TRL's GRPOTrainer.

    Args:
        dataset_name: HuggingFace dataset name or local path.
        dataset_config: Dataset configuration (e.g., "main").
        train_split: Name of the training split.
        test_split: Name of the test split (None to skip).
        prompt_template: Template key (e.g., "qwen_math", "r1").
        max_train_samples: Limit training samples (None for all).
        max_test_samples: Limit test samples (None for all).
        do_eval: Whether to load the test split.

    Returns:
        Tuple of (train_dataset, test_dataset). test_dataset may be None.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Validate template name early
    if prompt_template not in TEMPLATE_CONFIGS:
        raise ValueError(
            f"Unknown template: {prompt_template}. "
            f"Available: {list(TEMPLATE_CONFIGS.keys())}"
        )

    if local_rank == 0:
        logger.info("Loading dataset: %s (%s)", dataset_name, dataset_config)
        logger.info(
            "Using template: %s (%s)",
            prompt_template,
            TEMPLATE_CONFIGS[prompt_template].name,
        )

    # Load training dataset
    # train_dataset = load_dataset(
    #     dataset_name,
    #     dataset_config,
    #     split=train_split,
    # )
    if dataset_config:
        train_dataset = load_dataset(dataset_name, dataset_config, split=train_split)
    else:
        train_dataset = load_dataset(dataset_name, split=train_split)

    # Load test dataset if requested
    test_dataset = None
    if do_eval and test_split:
        try:
            # test_dataset = load_dataset(
            #     dataset_name,
            #     dataset_config,
            #     split=test_split,
            # )
            if dataset_config:
                test_dataset = load_dataset(dataset_name, dataset_config, split=test_split)
            else:
                test_dataset = load_dataset(dataset_name, split=test_split)
        except Exception as e:
            if local_rank == 0:
                logger.warning("Could not load test split '%s': %s", test_split, e)

    # Limit samples if specified
    if max_train_samples is not None:
        n = min(max_train_samples, len(train_dataset))
        train_dataset = train_dataset.select(range(n))

    if test_dataset is not None and max_test_samples is not None:
        n = min(max_test_samples, len(test_dataset))
        test_dataset = test_dataset.select(range(n))

    # Determine dataset type from name
    # dataset_type = "gsm8k" if "gsm8k" in dataset_name.lower() else "math"
    dataset_type = detect_dataset_type(dataset_name)

    # Get format function for the specified template
    format_fn = get_format_function(prompt_template, dataset_type)

    # Apply formatting
    train_dataset = train_dataset.map(
        format_fn, remove_columns=train_dataset.column_names
    )

    if test_dataset is not None:
        test_dataset = test_dataset.map(
            format_fn, remove_columns=test_dataset.column_names
        )

    if local_rank == 0:
        logger.info("Train dataset: %d samples", len(train_dataset))
        if test_dataset is not None:
            logger.info("Test dataset: %d samples", len(test_dataset))
        logger.info(
            "Example prompt (first 500 chars):\n%s",
            train_dataset[0]["prompt"][:500],
        )

    return train_dataset, test_dataset
