"""Command-line orchestrator for the modular MedMCQA pipeline."""
from __future__ import annotations

import argparse
import json
import os
from typing import List

from transformers import AutoTokenizer  # type: ignore

from .baseline import baseline_evaluate_model
from .chat import apply_chat_template_to_messages, build_chat_template, extract_pred_with_fallback
from .config import CONFIG
from .datasets import download_medmcqa, format_medmcqa_for_chat
from .grpo import run_grpo_training
from .inference import run_inference_with_lora
from .reporting import print_eval_summary
from .sft import run_sft_pretrain
from .utils import makedirs


def _resolve_limit(limit: int | None) -> int | None:
    return limit if limit and limit > 0 else None


def _ensure_directories() -> None:
    makedirs(CONFIG["data_dir"], CONFIG["checkpoints_dir"], CONFIG["experiments_dir"])


def _load_formatted_rows(path: str, limit: int | None = None) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if limit is not None and idx >= limit:
                break
            rows.append(json.loads(line))
    return rows


def _quick_eval(args, limit: int | None) -> None:
    val_formatted = os.path.join(args.formatted_dir, "medmcqa_formatted.validation.jsonl")
    if not os.path.exists(val_formatted):
        print("Validation formatted file not found:", val_formatted)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    markers = build_chat_template(tokenizer)

    rows = _load_formatted_rows(val_formatted, limit=min(limit, 200) if limit else 200)
    prompts = [
        apply_chat_template_to_messages(
            tokenizer,
            row["messages"],
            add_generation_prompt=True,
            tokenize=False,
        )
        for row in rows
    ]
    outputs = run_inference_with_lora(
        args.model_name,
        args.model_name,
        args.grpo_lora,
        prompts,
        max_new_tokens=128,
        batch_size=args.batch_size,
    )

    correct = 0
    total = 0
    for idx, output in enumerate(outputs):
        row = rows[idx]
        sol = extract_pred_with_fallback(
            output,
            markers["solution_start"],
            markers["solution_end"],
            options=(row.get("raw", {}) or {}).get("options"),
        )
        gold_label = row.get("answer_label") or (row.get("raw", {}) or {}).get("answer_label")
        gold_text = row.get("answer_text") or (row.get("raw", {}) or {}).get("answer_text")
        match = False
        if gold_label:
            if sol and sol.strip().upper() == str(gold_label).strip().upper():
                match = True
            elif sol and str(gold_label).strip().upper() in sol.strip().upper():
                match = True
        elif gold_text:
            if sol and sol.strip().lower() in str(gold_text).strip().lower():
                match = True
        print(f"[{idx + 1}] gold_label={gold_label} gold_text={gold_text} | pred={sol} | match={match}")
        if match:
            correct += 1
        total += 1
    print(f"GRPO validation eval: {correct}/{total} = {correct / total if total else 0:.4f}")


def pipeline_cli(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MedMCQA modular pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download", help="Download and process MedMCQA dataset (writes per-split files)")
    sub.add_parser("prep", help="Build chat template and format dataset for SFT/GRPO (train + validation)")
    sub.add_parser("baseline", help="Run baseline evaluation on validation split (batched)")
    sub.add_parser("sft", help="Run SFT pre-finetune on train split to teach formatting")
    sub.add_parser("grpo", help="Run GRPO training on train split (then evaluate on validation)")
    sub.add_parser("eval", help="Run inference on validation using saved LoRA")
    sub.add_parser("summary", help="Print the stored evaluation summary from experiments_v2")
    sub.add_parser("all", help="Run entire pipeline in sequence (download->prep->baseline->sft->grpo->eval)")

    parser.add_argument("--limit", type=int, default=5000, help="limit number of examples to process (0 or negative = no limit)")
    parser.add_argument("--model_name", type=str, default=CONFIG["model_name"])
    parser.add_argument("--processed_dir", type=str, default=CONFIG["data_dir"])
    parser.add_argument("--formatted_dir", type=str, default=CONFIG["data_dir"])
    parser.add_argument("--preformat_lora", type=str, default=os.path.join(CONFIG["checkpoints_dir"], CONFIG["preformat_lora_name"]))
    parser.add_argument("--grpo_lora", type=str, default=os.path.join(CONFIG["checkpoints_dir"], CONFIG["grpo_lora_name"]))
    parser.add_argument("--batch_size", type=int, default=CONFIG["eval_batch_size"], help="batch size for eval/inference")

    args = parser.parse_args(argv)

    _ensure_directories()
    limit = _resolve_limit(args.limit)

    if args.cmd == "download":
        download_medmcqa(args.processed_dir)

    elif args.cmd == "prep":
        print("Loading tokenizer to build chat template...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        markers = build_chat_template(tokenizer)
        for split in ["train", "validation"]:
            in_path = os.path.join(args.processed_dir, f"medmcqa_processed.{split}.jsonl")
            out_path = os.path.join(args.formatted_dir, f"medmcqa_formatted.{split}.jsonl")
            if os.path.exists(in_path):
                format_medmcqa_for_chat(in_path, out_path, tokenizer, markers, subset_limit=limit)
            else:
                print(f"Skipping split {split} (not found): {in_path}")

    elif args.cmd == "baseline":
        val_formatted = os.path.join(args.formatted_dir, "medmcqa_formatted.validation.jsonl")
        if not os.path.exists(val_formatted):
            print("Validation formatted file not found:", val_formatted)
            return
        baseline_evaluate_model(args.model_name, val_formatted, tokenizer_name=args.model_name, limit=limit, batch_size=args.batch_size)

    elif args.cmd == "sft":
        train_formatted = os.path.join(args.formatted_dir, "medmcqa_formatted.train.jsonl")
        if not os.path.exists(train_formatted):
            print("Train formatted file not found:", train_formatted)
            return
        out = run_sft_pretrain(train_formatted, args.model_name, CONFIG["preformat_lora_name"])
        print("SFT pretrain done, saved:", out)

    elif args.cmd == "grpo":
        train_formatted = os.path.join(args.formatted_dir, "medmcqa_formatted.train.jsonl")
        if not os.path.exists(train_formatted):
            print("Train formatted file not found:", train_formatted)
            return
        out = run_grpo_training(train_formatted, args.model_name, preformat_lora_path=args.preformat_lora, out_lora_dir=CONFIG["grpo_lora_name"])
        print("GRPO done, saved:", out)

        val_formatted = os.path.join(args.formatted_dir, "medmcqa_formatted.validation.jsonl")
        if os.path.exists(val_formatted):
            print("Running quick validation-set eval with GRPO LoRA...")
            _quick_eval(args, limit)

    elif args.cmd == "eval":
        val_formatted = os.path.join(args.formatted_dir, "medmcqa_formatted.validation.jsonl")
        if not os.path.exists(val_formatted):
            print("Validation formatted file not found:", val_formatted)
            return
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        markers = build_chat_template(tokenizer)
        rows = _load_formatted_rows(val_formatted, limit=min(limit, 20) if limit else 20)
        prompts = [
            apply_chat_template_to_messages(
                tokenizer,
                row["messages"],
                add_generation_prompt=True,
                tokenize=False,
            )
            for row in rows
        ]
        outputs = run_inference_with_lora(
            args.model_name,
            args.model_name,
            args.grpo_lora,
            prompts,
            max_new_tokens=256,
            batch_size=args.batch_size,
        )
        for idx, output in enumerate(outputs):
            sol = extract_pred_with_fallback(
                output,
                markers["solution_start"],
                markers["solution_end"],
                options=(rows[idx].get("raw", {}) or {}).get("options"),
            )
            print(f"=== Example {idx} ===\nGen:\n{output}\nExtracted solution: {sol}\n")

    elif args.cmd == "summary":
        print_eval_summary()

    elif args.cmd == "all":
        download_medmcqa(args.processed_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        markers = build_chat_template(tokenizer)
        for split in ["train", "validation"]:
            in_path = os.path.join(args.processed_dir, f"medmcqa_processed.{split}.jsonl")
            out_path = os.path.join(args.formatted_dir, f"medmcqa_formatted.{split}.jsonl")
            if os.path.exists(in_path):
                format_medmcqa_for_chat(in_path, out_path, tokenizer, markers, subset_limit=limit)
        val_formatted = os.path.join(args.formatted_dir, "medmcqa_formatted.validation.jsonl")
        if os.path.exists(val_formatted):
            baseline_evaluate_model(args.model_name, val_formatted, tokenizer_name=args.model_name, limit=50, batch_size=args.batch_size)
        train_formatted = os.path.join(args.formatted_dir, "medmcqa_formatted.train.jsonl")
        if os.path.exists(train_formatted):
            run_sft_pretrain(train_formatted, args.model_name, CONFIG["preformat_lora_name"])
            run_grpo_training(
                train_formatted,
                args.model_name,
                preformat_lora_path=os.path.join(CONFIG["checkpoints_dir"], CONFIG["preformat_lora_name"]),
            )
        print("Pipeline complete. Validation split used for all testing/evaluation.")

    else:
        parser.print_help()


def main() -> None:
    pipeline_cli()


if __name__ == "__main__":
    main()
