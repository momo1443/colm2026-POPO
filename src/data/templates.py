"""
Prompt template management for various LLM model families.

Supports:
- R1 (DeepSeek-R1 style with <think>/<answer>)
- Qwen-Math (im_start/im_end)
- Llama-3 Instruct
- DeepSeek-Math
- Simple instruction format
- No template (raw question)

Canonical source: test_environment/train_grpo_v2.py lines 60-227.
"""

from typing import Optional, Callable, Dict
from dataclasses import dataclass


@dataclass
class TemplateConfig:
    """Configuration for a prompt template."""
    name: str
    system_prompt: Optional[str]
    user_prefix: str
    user_suffix: str
    assistant_prefix: str
    assistant_suffix: str = ""
    supports_thinking: bool = False
    answer_format: str = "boxed"  # "boxed", "####", or "plain"


# Template configurations for different model families
TEMPLATE_CONFIGS: Dict[str, TemplateConfig] = {
    "r1": TemplateConfig(
        name="R1 (DeepSeek-R1)",
        system_prompt=(
            "A conversation between User and Assistant. The User asks a question, "
            "and the Assistant solves it. The Assistant first thinks about the reasoning "
            "process in the mind and then provides the User with the answer. "
            "The reasoning process is enclosed within <think> </think> and answer is "
            "enclosed within <answer> </answer> tags, respectively, i.e., "
            "<think> reasoning process here </think> <answer> answer here </answer>."
        ),
        user_prefix="User: ",
        user_suffix="\n",
        assistant_prefix="Assistant: <think>",
        supports_thinking=True,
        answer_format="boxed",
    ),
    "qwen_math": TemplateConfig(
        name="Qwen-Math",
        system_prompt="Please reason step by step, and put your final answer within \\boxed{}.",
        user_prefix="<|im_start|>user\n",
        user_suffix="<|im_end|>\n",
        assistant_prefix="<|im_start|>assistant\n",
        assistant_suffix="<|im_end|>",
        supports_thinking=False,
        answer_format="boxed",
    ),
    "llama": TemplateConfig(
        name="Llama-3 Instruct",
        system_prompt=(
            "You are a helpful math assistant. Solve the problem step by step "
            "and put your final answer in \\boxed{}."
        ),
        user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
        user_suffix="<|eot_id|>",
        assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
        assistant_suffix="<|eot_id|>",
        supports_thinking=False,
        answer_format="boxed",
    ),
    "deepseek": TemplateConfig(
        name="DeepSeek-Math",
        system_prompt=(
            "You are a helpful assistant. Please solve the following math problem "
            "step by step and put your final answer within \\boxed{}."
        ),
        user_prefix="User: ",
        user_suffix="\n\n",
        assistant_prefix="Assistant: ",
        supports_thinking=False,
        answer_format="boxed",
    ),
    "none": TemplateConfig(
        name="No Template",
        system_prompt=None,
        user_prefix="",
        user_suffix="",
        assistant_prefix="",
        supports_thinking=False,
        answer_format="plain",
    ),
    "simple": TemplateConfig(
        name="Simple Instruction",
        system_prompt=None,
        user_prefix="Solve this math problem step by step.\n\nProblem: ",
        user_suffix="\n\nSolution:",
        assistant_prefix="",
        supports_thinking=False,
        answer_format="boxed",
    ),
}

@dataclass
class DatasetFieldConfig:
    """Maps dataset fields to canonical names."""
    question_field: str = "question"
    answer_field: str = "answer"


DATASET_FIELD_CONFIGS: Dict[str, DatasetFieldConfig] = {
    "gsm8k":       DatasetFieldConfig(question_field="question", answer_field="answer"),
    "math":        DatasetFieldConfig(question_field="problem",  answer_field="solution"),
    "aime":        DatasetFieldConfig(question_field="problem",  answer_field="answer"),
    "deepscaler":  DatasetFieldConfig(question_field="problem",  answer_field="answer"),
    "amc23":  DatasetFieldConfig(question_field="question",  answer_field="answer"),
    "math500":       DatasetFieldConfig(question_field="problem",  answer_field="solution"),
    "olympiadbench": DatasetFieldConfig(question_field="question", answer_field="final_answer"),
}

def detect_dataset_type(dataset_name: str) -> str:
    name_lower = dataset_name.lower()
    if "gsm8k" in name_lower:
        return "gsm8k"
    elif "aime" in name_lower:
        return "aime"
    elif "deepscale" in name_lower:
        return "deepscaler"
    elif "amc23" in name_lower:
        return "amc23"
    elif "math-500" in name_lower or "math_500" in name_lower:
        return "math500"
    elif "olympiad" in name_lower:
        return "olympiadbench"
    else:
        return "math"

def format_with_template(
    example: dict,
    template_name: str,
    dataset_type: str = "gsm8k",
) -> dict:

    field_config = DATASET_FIELD_CONFIGS.get(dataset_type, DATASET_FIELD_CONFIGS["math"])

    def _get_field(ex: dict, field_name: str):
        """Try exact match first, then case-insensitive."""
        if field_name in ex:
            return ex[field_name]
        for key in ex:
            if key.lower() == field_name.lower():
                return ex[key]
        return ""

    question = _get_field(example, field_config.question_field)
    solution = _get_field(example, field_config.answer_field)

    if not question:
        for fallback in ["problem", "question"]:
            question = _get_field(example, fallback)
            if question:
                break

    if not solution:
        for fallback in ["answer", "solution"]:
            solution = _get_field(example, fallback)
            if solution:
                break

    if isinstance(solution, list):
        solution = solution[0] if len(solution) == 1 else ", ".join(str(s) for s in solution)
    if not isinstance(solution, str):
        solution = str(solution)

    config = TEMPLATE_CONFIGS.get(template_name)
    if config is None:
        raise ValueError(
            f"Unknown template: {template_name}. "
            f"Available: {list(TEMPLATE_CONFIGS.keys())}"
        )

    # Build the prompt
    parts = []

    # Add system prompt with template-specific formatting
    if config.system_prompt:
        if template_name == "qwen_math":
            parts.append(
                f"<|im_start|>system\n{config.system_prompt}<|im_end|>\n"
            )
        elif template_name == "llama":
            parts.append(
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                f"\n\n{config.system_prompt}<|eot_id|>"
            )
        elif template_name == "r1":
            parts.append(f"{config.system_prompt}\n")
        elif template_name == "deepseek":
            parts.append(f"{config.system_prompt}\n\n")

    # Add user message
    parts.append(f"{config.user_prefix}{question}{config.user_suffix}")

    # Add assistant prefix
    parts.append(config.assistant_prefix)

    prompt = "".join(parts)

    return {"prompt": prompt, "solution": solution}


def get_format_function(
    template_name: str,
    dataset_type: str,
) -> Callable:
    """
    Get a formatting callable for the specified template and dataset type.

    Returns a function suitable for use with dataset.map().

    Args:
        template_name: Key into TEMPLATE_CONFIGS.
        dataset_type: Type of dataset ("gsm8k" or "math").

    Returns:
        Callable that takes a dataset example dict and returns
        a dict with 'prompt' and 'solution' keys.
    """

    def format_fn(example: dict) -> dict:
        return format_with_template(example, template_name, dataset_type)

    return format_fn
