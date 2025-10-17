"""Global configuration shared across pipeline modules."""
from __future__ import annotations

import os

import torch

# Ensure a default GPU device is visible when none is set externally.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "model_name": "unsloth/Qwen3-4B-Base",
    "max_seq_length": 2048,
    "lora_rank": 32,
    "gpu_memory_utilization": 0.9,
    "seed": 3407,
    "data_dir": "data",
    "checkpoints_dir": "checkpoints",
    "experiments_dir": "experiments",
    "preformat_lora_name": "preformat_lora",
    "grpo_lora_name": "grpo_lora",
    "device": DEVICE,
    "sft": {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 5,
        "num_train_epochs": 5,
        "learning_rate": 2e-4,
        "logging_steps": 5,
    },
    "grpo": {
        "temperature": 1.0,
        "learning_rate": 5e-6,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "num_generations": 4,
        "max_steps": 2000,
        "save_steps": 100,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
    },
    "eval_batch_size": 16,
}

__all__ = ["CONFIG", "DEVICE"]
