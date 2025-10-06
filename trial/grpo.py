"""GRPO training utilities."""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np  # type: ignore
from datasets import Dataset  # type: ignore
from transformers import AutoTokenizer  # type: ignore

from .chat import apply_chat_template_to_messages, build_chat_template
from .config import CONFIG
from .deps import FastLanguageModel, GRPOConfig, GRPOTrainer, SamplingParams
from .rewards import (
    check_answer,
    check_numbers,
    match_format_approximately,
    match_format_exactly,
)
from .utils import makedirs


def _resolve_gold_from_row(row: Dict[str, object]) -> Tuple[Optional[str], Optional[str]]:
    gold_label = row.get("answer_label")
    gold_text = row.get("answer_text")

    raw = row.get("raw") or {}
    if isinstance(raw, dict):
        if not gold_label and raw.get("answer_label"):
            gold_label = raw.get("answer_label")
        if not gold_text and raw.get("answer_text"):
            gold_text = raw.get("answer_text")
        if not gold_text and raw.get("answer"):
            gold_text = raw.get("answer")

        cop_value = raw.get("cop")
        options = raw.get("options") or row.get("options") or []
        if cop_value is not None and cop_value not in ("", None) and not gold_label and not gold_text:
            try:
                idx = int(cop_value)
                if idx == -1:
                    pass
                elif 0 <= idx < len(options):
                    gold_label = chr(ord("A") + idx)
                    gold_text = options[idx]
                elif 1 <= idx <= len(options):
                    gold_label = chr(ord("A") + (idx - 1))
                    gold_text = options[idx - 1]
            except Exception:
                cop_string = str(cop_value).strip()
                if cop_string:
                    gold_text = cop_string

    def clean(value: Optional[str]) -> Optional[str]:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value  # type: ignore

    return clean(gold_label), clean(gold_text)


def _build_reward_wrappers(rows: Sequence[Dict[str, object]], markers: Dict[str, str]) -> List[Callable]:
    raw_reward_functions = []
    if match_format_exactly:
        raw_reward_functions.append(match_format_exactly)
    if match_format_approximately:
        raw_reward_functions.append(match_format_approximately)
    if check_answer:
        raw_reward_functions.append(check_answer)
    if check_numbers:
        raw_reward_functions.append(check_numbers)

    if not raw_reward_functions:
        raise RuntimeError(
            "No reward functions available. Ensure rewards.py is present and imports succeed."
        )

    first_call_debug = {"seen": False, "printed_sample": False}

    def make_batch_wrapper(fn: Callable):
        name = getattr(fn, "__name__", "reward_fn")

        def wrapped(completions, prompts_arg=None, completion_ids=None, **kwargs):  # type: ignore
            if not first_call_debug["seen"]:
                print("=== REWARD WRAPPER FIRST CALL DEBUG ===")
                print("wrapped fn:", name)
                print("completions type:", type(completions))
                if isinstance(completions, (list, tuple)):
                    print("len(completions):", len(completions))
                    if completions:
                        preview = completions[0]
                        print(
                            "first completion preview (truncated):",
                            (str(preview)[:500] + "...") if preview else None,
                        )
                print("prompts_arg type:", type(prompts_arg))
                print(
                    "completion_ids type:",
                    type(completion_ids),
                    "sample:",
                    (completion_ids[:4] if isinstance(completion_ids, (list, tuple)) else completion_ids),
                )
                first_call_debug["seen"] = True

            flat_completions: List[str] = []
            comp_to_idx: List[int] = []

            if isinstance(completion_ids, (list, tuple)):
                if isinstance(completions, (list, tuple)) and len(completions) == len(completion_ids):
                    for idx, comp_entry in enumerate(completions):
                        target_idx = completion_ids[idx]
                        if isinstance(target_idx, (list, tuple)):
                            mapped_idx = idx
                        else:
                            try:
                                mapped_idx = int(target_idx)
                            except Exception:
                                mapped_idx = idx
                        if isinstance(comp_entry, (list, tuple)):
                            for c in comp_entry:
                                flat_completions.append(c)
                                comp_to_idx.append(mapped_idx)
                        else:
                            flat_completions.append(comp_entry)
                            comp_to_idx.append(mapped_idx)
                else:
                    if isinstance(completions, (list, tuple)):
                        for idx, comp_entry in enumerate(completions):
                            if idx < len(completion_ids):
                                candidate = completion_ids[idx]
                                if isinstance(candidate, (list, tuple)):
                                    mapped_idx = idx
                                else:
                                    try:
                                        mapped_idx = int(candidate)
                                    except Exception:
                                        mapped_idx = idx
                            else:
                                mapped_idx = idx
                            if isinstance(comp_entry, (list, tuple)):
                                for c in comp_entry:
                                    flat_completions.append(c)
                                    comp_to_idx.append(mapped_idx)
                            else:
                                flat_completions.append(comp_entry)
                                comp_to_idx.append(mapped_idx)
                    else:
                        flat_completions = [completions]
                        comp_to_idx = [0]
            else:
                if isinstance(completions, (list, tuple)):
                    if completions and isinstance(completions[0], (list, tuple)) and isinstance(prompts_arg, (list, tuple)):
                        for pi, sub in enumerate(completions):
                            for c in (sub if isinstance(sub, (list, tuple)) else [sub]):
                                flat_completions.append(c)
                                comp_to_idx.append(pi if pi < len(prompts_arg) else 0)
                    else:
                        for idx, c in enumerate(completions):
                            flat_completions.append(c)
                            comp_to_idx.append(idx if isinstance(prompts_arg, (list, tuple)) and idx < len(prompts_arg) else 0)
                else:
                    flat_completions = [completions]
                    comp_to_idx = [0]

            rewards: List[float] = []
            for completion, ex_idx in zip(flat_completions, comp_to_idx):
                example = rows[ex_idx] if isinstance(ex_idx, int) and 0 <= ex_idx < len(rows) else None
                prompt_for_call = None
                if isinstance(prompts_arg, (list, tuple)) and isinstance(ex_idx, int) and ex_idx < len(prompts_arg):
                    prompt_for_call = prompts_arg[ex_idx]
                elif isinstance(prompts_arg, str):
                    prompt_for_call = prompts_arg

                value = None
                try:
                    if example is not None:
                        try:
                            value = fn(completion, prompt_for_call, example, markers)
                        except TypeError:
                            try:
                                value = fn(completion, prompt_for_call, example)
                            except TypeError:
                                value = None
                    if value is None:
                        try:
                            value = fn(completion, prompt_for_call)
                        except TypeError:
                            try:
                                value = fn(completion)
                            except Exception:
                                value = None
                except Exception as exc:
                    print(f"Reward function {name} raised for ex_idx={ex_idx}: {exc}")
                    value = None

                if isinstance(value, (int, float)):
                    rewards.append(float(value))
                elif isinstance(value, (list, tuple)):
                    try:
                        rewards.append(float(value[0]) if value else 0.0)
                    except Exception:
                        rewards.append(0.0)
                else:
                    rewards.append(0.0)

            if first_call_debug.get("seen") and not first_call_debug.get("printed_sample"):
                print("=== REWARD WRAPPER SAMPLE OUTPUT ===")
                print("flat_completions_count:", len(flat_completions))
                print("comp_to_idx sample:", comp_to_idx[:10])
                print("rewards sample:", rewards[:10])
                first_call_debug["printed_sample"] = True

            return rewards

        return wrapped

    return [make_batch_wrapper(fn) for fn in raw_reward_functions]


def run_grpo_training(
    formatted_jsonl: str,
    model_name: str,
    preformat_lora_path: Optional[str] = None,
    out_lora_dir: Optional[str] = None,
    require_label: bool = True,
) -> str:
    """Execute the GRPO training loop and save the resulting LoRA adapter."""
    if GRPOTrainer is None or SamplingParams is None or FastLanguageModel is None:
        raise RuntimeError("trl.GRPOTrainer, vllm.SamplingParams and unsloth.FastLanguageModel are required.")

    print("Loading formatted dataset for GRPO:", formatted_jsonl)
    with open(formatted_jsonl, "r", encoding="utf-8") as fh:
        rows_all = [json.loads(line) for line in fh]
    if not rows_all:
        raise RuntimeError(f"No examples found in {formatted_jsonl}")
    print(f"Total examples in file: {len(rows_all)}")

    rows = rows_all
    if require_label:
        filtered: List[Dict[str, object]] = []
        for row in rows_all:
            lbl, txt = _resolve_gold_from_row(row)
            if lbl is not None or txt is not None:
                filtered.append(row)
        print(f"Filtered training rows: kept {len(filtered)}/{len(rows_all)} examples with gold label/text.")
        rows = filtered
        if not rows:
            raise RuntimeError(
                "No labeled examples after filtering. Set require_label=False or fix processed dataset (cop != -1 or answer_label present)."
            )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    markers = build_chat_template(tokenizer)

    prompts: List[str] = []
    answers: List[str] = []
    for row in rows:
        prompts.append(
            apply_chat_template_to_messages(
                tokenizer,
                row["messages"],
                add_generation_prompt=True,
                tokenize=False,
            )
        )
        lbl, txt = _resolve_gold_from_row(row)
        answers.append(txt or lbl or "")

    print(f"Prepared {len(prompts)} prompts for GRPO (after filtering).")

    sampling = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=CONFIG.get("seed", 3407),
        stop=[tokenizer.eos_token] if hasattr(tokenizer, "eos_token") else None,
        include_stop_str_in_output=True,
    )

    print("Loading model (FastLanguageModel)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=CONFIG["max_seq_length"],
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=CONFIG["lora_rank"],
        gpu_memory_utilization=CONFIG["gpu_memory_utilization"],
    )

    print("Applying PEFT/LoRA wrappers to model...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora_rank"],
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=CONFIG["lora_rank"] * 2,
        use_gradient_checkpointing="unsloth",
        random_state=CONFIG.get("seed", 3407),
    )

    if preformat_lora_path:
        try:
            print("Loading preformat LoRA:", preformat_lora_path)
            model.load_lora(preformat_lora_path)
            print("Preformat LoRA loaded.")
        except Exception as exc:
            print("Could not load preformat LoRA (continuing):", exc)

    grpo_args = GRPOConfig(
        vllm_sampling_params=sampling,
        temperature=CONFIG["grpo"]["temperature"],
        learning_rate=CONFIG["grpo"]["learning_rate"],
        weight_decay=CONFIG["grpo"]["weight_decay"],
        warmup_ratio=CONFIG["grpo"]["warmup_ratio"],
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=CONFIG["grpo"]["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["grpo"]["gradient_accumulation_steps"],
        num_generations=CONFIG["grpo"]["num_generations"],
        max_prompt_length=512,
        max_completion_length=CONFIG["max_seq_length"] - 512,
        max_steps=CONFIG["grpo"]["max_steps"],
        save_steps=CONFIG["grpo"]["save_steps"],
        report_to="none",
        output_dir=CONFIG["checkpoints_dir"],
    )

    print("Tokenizing prompts to compute prompt length quantile...")
    token_lengths: List[int] = []
    for prompt in prompts:
        try:
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
            token_lengths.append(int(ids.shape[0]))
        except Exception:
            token_lengths.append(min(len(prompt.split()), CONFIG["max_seq_length"] // 4))

    try:
        max_prompt_length = int(np.quantile(np.array(token_lengths), 0.9))
    except Exception:
        max_prompt_length = min(max(token_lengths) if token_lengths else 512, 512)
    grpo_args.max_prompt_length = max_prompt_length + 1
    grpo_args.max_completion_length = CONFIG["max_seq_length"] - grpo_args.max_prompt_length
    print(
        f"Set max_prompt_length={grpo_args.max_prompt_length}, "
        f"max_completion_length={grpo_args.max_completion_length}"
    )

    train_dataset = Dataset.from_dict({"prompt": prompts, "answer": answers})

    reward_wrappers = _build_reward_wrappers(rows, markers)
    print(f"Wrapping {len(reward_wrappers)} reward functions for GRPO trainer (batch-aware).")

    print("Creating GRPOTrainer with args:", grpo_args)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_wrappers,
        args=grpo_args,
        train_dataset=train_dataset,
    )

    setattr(trainer, "image_token_id", None)
    setattr(trainer, "vision_start_token_id", None)
    setattr(trainer, "vision_end_token_id", None)

    print("Starting GRPO training (this may take a while)...")
    trainer.train()

    makedirs(CONFIG["checkpoints_dir"])
    out_dir = os.path.join(CONFIG["checkpoints_dir"], out_lora_dir or CONFIG["grpo_lora_name"])
    try:
        model.save_lora(out_dir)
        print("Saved GRPO LoRA to", out_dir)
    except Exception as exc:
        print("Failed to save GRPO LoRA via model.save_lora():", exc)
        try:
            trainer.save_model(out_dir)
            print("Trainer saved model to", out_dir)
        except Exception as exc2:
            print("Also failed to save via trainer.save_model():", exc2)
            raise

    return out_dir


__all__ = ["run_grpo_training"]
