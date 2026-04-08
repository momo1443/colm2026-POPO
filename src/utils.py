"""
Shared utilities for POPO project.

Provides:
    - YAML config loading with CLI override support
    - Experiment directory setup (logs/{experiment_name}/...)
    - Logging helpers
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


# =============================================================================
# YAML Config Loading with CLI Override
# =============================================================================

def load_yaml_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary of configuration key-value pairs.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}

    return config


def apply_yaml_defaults(parser: argparse.ArgumentParser, config_path: str) -> None:
    """
    Load YAML config and set as parser defaults.

    This allows CLI arguments to override YAML values. The pattern is:
        1. Define parser with hardcoded defaults
        2. Load YAML and call parser.set_defaults(**yaml_config)
        3. Re-parse args -- explicit CLI args override YAML, YAML overrides hardcoded

    Unknown YAML keys (not matching any argparse argument) raise ValueError
    to catch typos early.

    Args:
        parser: The argparse parser (before final parse_args call).
        config_path: Path to the YAML config file.

    Raises:
        ValueError: If YAML contains keys not matching any parser argument.
    """
    yaml_config = load_yaml_config(config_path)

    # Validate keys against known parser arguments
    known_args = {action.dest for action in parser._actions}
    unknown_keys = set(yaml_config.keys()) - known_args
    if unknown_keys:
        raise ValueError(
            f"Unknown keys in YAML config '{config_path}': {unknown_keys}. "
            f"Check for typos. Known arguments: {sorted(known_args)}"
        )

    parser.set_defaults(**yaml_config)


# =============================================================================
# Experiment Directory Setup
# =============================================================================

EXPERIMENT_SUBDIRS = [
    "checkpoints",
    "runs",           # TensorBoard event files
    "model_summary",
    "train_config",
    "train_logs",
]


def setup_experiment_dir(
    base_dir: str = "logs",
    experiment_name: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
) -> str:
    """
    Create the experiment directory structure.

    Creates:
        {base_dir}/{experiment_name}/
            ├── checkpoints/
            ├── runs/              (TensorBoard)
            ├── model_summary/
            ├── train_config/
            └── train_logs/

    Args:
        base_dir: Root directory for all experiments (default: "logs").
        experiment_name: Name for this experiment. If None, auto-generated
            from timestamp.
        args: If provided, saves the full config to train_config/config.yaml.

    Returns:
        Path to the experiment directory.
    """
    if experiment_name is None:
        experiment_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    experiment_dir = os.path.join(base_dir, experiment_name)

    for subdir in EXPERIMENT_SUBDIRS:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)

    # Save config snapshot for reproducibility
    if args is not None:
        config_path = os.path.join(experiment_dir, "train_config", "config.yaml")
        config_dict = vars(args) if isinstance(args, argparse.Namespace) else args
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)

    return experiment_dir


def get_tensorboard_dir(experiment_dir: str) -> str:
    """Return the TensorBoard logging directory for an experiment."""
    return os.path.join(experiment_dir, "runs")


def get_checkpoint_dir(experiment_dir: str) -> str:
    """Return the checkpoint directory for an experiment."""
    return os.path.join(experiment_dir, "checkpoints")


def get_train_logs_dir(experiment_dir: str) -> str:
    """Return the training logs directory for an experiment."""
    return os.path.join(experiment_dir, "train_logs")


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    rank: int = 0,
) -> None:
    """
    Configure logging for training scripts.

    Args:
        log_dir: Directory to save log files. If None, logs to stdout only.
        level: Logging level.
        rank: Process rank. Only rank 0 logs to file.
    """
    handlers = []

    # Console handler (all ranks)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    handlers.append(console_handler)

    # File handler (rank 0 only)
    if log_dir is not None and rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "training.log"), mode="a"
        )
        file_handler.setLevel(level)
        file_fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)


# =============================================================================
# Environment Info
# =============================================================================

def print_env_info(args: argparse.Namespace) -> None:
    """Print environment and configuration summary (rank 0 only)."""
    import torch

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        return

    print("=" * 70)
    print("POPO Training Environment")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024**3
            print(f"  GPU {i}: {props.name} ({mem_gb:.1f} GB)")

    print(f"\nModel: {getattr(args, 'model_name', 'N/A')}")
    print(f"Dataset: {getattr(args, 'dataset_name', 'N/A')}")
    print(f"Template: {getattr(args, 'prompt_template', 'N/A')}")
    print(f"Use vLLM: {getattr(args, 'use_vllm', False)}")
    if getattr(args, "use_vllm", False):
        print(f"  vLLM Mode: {getattr(args, 'vllm_mode', 'N/A')}")
    print("=" * 70)
    print()
