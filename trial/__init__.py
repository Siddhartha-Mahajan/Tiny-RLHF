"""Modular pipeline components for the MedMCQA project."""

from .config import CONFIG, DEVICE
from .datasets import download_medmcqa, format_medmcqa_for_chat
from .chat import (
    build_chat_template,
    apply_chat_template_to_messages,
    extract_solution_from_text,
    extract_pred_with_fallback,
)
from .baseline import baseline_evaluate_model
from .sft import run_sft_pretrain
from .grpo import run_grpo_training
from .inference import run_inference_with_lora
from .pipeline import pipeline_cli, main as pipeline_main
from .reporting import load_eval_summary, print_eval_summary
from .rewards import (
    match_format_exactly,
    match_format_approximately,
    check_answer,
    check_numbers,
)

__all__ = [
    "CONFIG",
    "DEVICE",
    "download_medmcqa",
    "format_medmcqa_for_chat",
    "build_chat_template",
    "apply_chat_template_to_messages",
    "extract_solution_from_text",
    "extract_pred_with_fallback",
    "baseline_evaluate_model",
    "run_sft_pretrain",
    "run_grpo_training",
    "run_inference_with_lora",
    "pipeline_cli",
    "pipeline_main",
    "load_eval_summary",
    "print_eval_summary",
    "match_format_exactly",
    "match_format_approximately",
    "check_answer",
    "check_numbers",
]
