"""Inference helpers for running LoRA adapters."""
from __future__ import annotations

from typing import List, Optional

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from peft import PeftModel  # type: ignore

from .config import CONFIG


def run_inference_with_lora(
    model_name: str,
    tokenizer_name: str,
    lora_path: str,
    prompts: List[str],
    max_new_tokens: int = 256,
    batch_size: Optional[int] = None,
) -> List[str]:
    """Generate completions using a (possibly) LoRA-augmented model."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        print("Warning: device_map/float16 shortcut failed, falling back:", exc)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(CONFIG["device"])

    if lora_path and os.path.isdir(lora_path):
        try:
            model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
            print("Loaded LoRA adapter from", lora_path)
        except Exception as exc:
            print("Failed to load LoRA adapter via PeftModel:", exc)
            print("Continuing with base model (no adapter).")
    else:
        print("No LoRA adapter found at", lora_path, "- continuing with base model.")

    model.eval()
    if batch_size is None:
        batch_size = CONFIG.get("eval_batch_size", 8)

    outputs: List[str] = []
    device = next(model.parameters()).device

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CONFIG["max_seq_length"],
        )
        enc = {key: value.to(device) for key, value in enc.items()}

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
            )

        for idx in range(len(batch_prompts)):
            decoded = tokenizer.decode(gen[idx], skip_special_tokens=False)
            prompt_decoded = tokenizer.decode(enc["input_ids"][idx], skip_special_tokens=False)
            if decoded.startswith(prompt_decoded):
                outputs.append(decoded[len(prompt_decoded):].strip())
            else:
                outputs.append(decoded.strip())

    return outputs


__all__ = ["run_inference_with_lora"]
