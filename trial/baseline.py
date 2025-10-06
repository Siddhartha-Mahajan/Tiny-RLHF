"""Baseline evaluation helpers."""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

from .chat import (
    apply_chat_template_to_messages,
    build_chat_template,
    extract_pred_with_fallback,
)
from .config import CONFIG
from .deps import FastLanguageModel, SamplingParams


def _resolve_gold_label_and_text(example: Dict[str, object]) -> Tuple[Optional[str], Optional[str]]:
    gold_label = example.get("answer_label")
    gold_text = example.get("answer_text")

    raw = example.get("raw") or {}
    if isinstance(raw, dict):
        if not gold_label and raw.get("answer_label"):
            gold_label = raw.get("answer_label")
        if not gold_text and raw.get("answer_text"):
            gold_text = raw.get("answer_text")
        if not gold_text and raw.get("answer"):
            gold_text = raw.get("answer")

        cop_value = raw.get("cop")
        options = raw.get("options") or example.get("options") or []
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


def baseline_evaluate_model(
    model_name: str,
    formatted_jsonl: str,
    tokenizer_name: Optional[str] = None,
    max_new_tokens: int = 64,
    limit: Optional[int] = 100,
    batch_size: Optional[int] = None,
) -> None:
    """Evaluate a pretrained model on the formatted validation split."""
    print("Starting baseline evaluation (robust label detection, batched)...")

    rows: List[Dict[str, object]] = []
    with open(formatted_jsonl, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if limit and limit > 0 and idx >= limit:
                break
            rows.append(json.loads(line))

    if not rows:
        print("No examples loaded from", formatted_jsonl)
        return

    labeled: List[Dict[str, object]] = []
    unlabeled: List[Dict[str, object]] = []
    for example in rows:
        gold_label, gold_text = _resolve_gold_label_and_text(example)
        if gold_label is not None or gold_text is not None:
            example["_gold_label"] = gold_label
            example["_gold_text"] = gold_text
            labeled.append(example)
        else:
            unlabeled.append(example)

    print(
        f"Loaded {len(rows)} examples -> {len(labeled)} labeled, {len(unlabeled)} unlabeled "
        "(unlabeled will be skipped)."
    )
    if not labeled:
        print(
            "No labeled examples found — cannot compute baseline accuracy. "
            "Check processed files or use a different split (train/validation)."
        )
        if rows:
            from pprint import pprint

            print("Sample processed row (first):")
            pprint(rows[0])
        return

    if tokenizer_name is None:
        tokenizer_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    markers = build_chat_template(tokenizer)

    model = None
    use_unsloth = False
    if FastLanguageModel is not None:
        try:
            print("Attempting to load model via Unsloth FastLanguageModel ...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name,
                max_seq_length=CONFIG["max_seq_length"],
                load_in_4bit=False,
                fast_inference=True,
                max_lora_rank=CONFIG["lora_rank"],
                gpu_memory_utilization=CONFIG["gpu_memory_utilization"],
            )
            use_unsloth = True
        except Exception as exc:
            print("Unsloth load failed, falling back to transformers:", exc)
            model = None

    if model is None:
        print("Loading base model via transformers (may be slower)...")
        model = AutoModelForCausalLM.from_pretrained(model_name).to(CONFIG["device"])

    if batch_size is None:
        batch_size = CONFIG.get("eval_batch_size", 8)

    prompts: List[str] = []
    ordered_examples: List[Dict[str, object]] = []
    for example in labeled:
        prompt_text = apply_chat_template_to_messages(
            tokenizer,
            example["messages"],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompts.append(prompt_text)
        ordered_examples.append(example)

    correct = 0
    total = 0

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_examples = ordered_examples[start : start + batch_size]
        outputs = [""] * len(batch_prompts)

        try:
            if use_unsloth and hasattr(model, "fast_generate"):
                if SamplingParams is None:
                    raise RuntimeError("vllm.SamplingParams not available for fast_generate")
                sampling_params = SamplingParams(temperature=1.0, max_tokens=max_new_tokens)
                results = model.fast_generate(batch_prompts, sampling_params=sampling_params, lora_request=None)
                for idx, result in enumerate(results):
                    try:
                        outputs[idx] = result.outputs[0].text
                    except Exception:
                        outputs[idx] = getattr(result, "text", "")
            else:
                enc = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=CONFIG["max_seq_length"],
                )
                enc = {key: value.to(CONFIG["device"]) for key, value in enc.items()}

                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=getattr(tokenizer, "eos_token_id", None),
                )

                for idx in range(len(batch_prompts)):
                    decoded = tokenizer.decode(gen[idx], skip_special_tokens=False)
                    prompt_decoded = tokenizer.decode(enc["input_ids"][idx], skip_special_tokens=False)
                    outputs[idx] = (
                        decoded[len(prompt_decoded):].strip()
                        if decoded.startswith(prompt_decoded)
                        else decoded
                    )
        except Exception as batch_exc:
            print("Batch generation failed at start", start, "— falling back to per-example.", batch_exc)
            for idx, prompt in enumerate(batch_prompts):
                try:
                    enc_single = tokenizer(prompt, return_tensors="pt")
                    enc_single = {key: tensor.to(CONFIG["device"]) for key, tensor in enc_single.items()}
                    gen_single = model.generate(
                        **enc_single,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        eos_token_id=getattr(tokenizer, "eos_token_id", None),
                    )
                    decoded = tokenizer.decode(gen_single[0], skip_special_tokens=False)
                    prompt_decoded = tokenizer.decode(enc_single["input_ids"][0], skip_special_tokens=False)
                    outputs[idx] = (
                        decoded[len(prompt_decoded):].strip()
                        if decoded.startswith(prompt_decoded)
                        else decoded
                    )
                except Exception as single_exc:
                    print("Per-example fallback failed:", single_exc)
                    outputs[idx] = ""

        for idx, generated in enumerate(outputs):
            example = batch_examples[idx]
            raw = example.get("raw") if isinstance(example.get("raw"), dict) else {}
            options = None
            if isinstance(raw, dict) and raw.get("options"):
                options = raw.get("options")
            elif example.get("options"):
                options = example.get("options")

            prediction = extract_pred_with_fallback(
                generated,
                markers["solution_start"],
                markers["solution_end"],
                options=options,
            )
            gold_label = example.get("_gold_label")
            gold_text = example.get("_gold_text")

            match = False
            if gold_label:
                if prediction and prediction.strip().upper() == str(gold_label).strip().upper():
                    match = True
                elif prediction and str(gold_label).strip().upper() in prediction.strip().upper():
                    match = True
            elif gold_text:
                if prediction and prediction.strip().lower() in str(gold_text).strip().lower():
                    match = True

            print(
                f"[{total + 1}] gold_label={gold_label} gold_text={gold_text} "
                f"| pred={prediction} | match={match}"
            )
            if match:
                correct += 1
            total += 1

    print(f"Baseline: {correct}/{total} = {correct / total if total else 0:.4f}")


__all__ = ["baseline_evaluate_model"]
