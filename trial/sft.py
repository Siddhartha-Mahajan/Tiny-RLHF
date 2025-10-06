"""Supervised fine-tuning (SFT) utilities."""
from __future__ import annotations

import json
import os
from typing import Dict, List

import torch
from datasets import Dataset  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

from .chat import apply_chat_template_to_messages, build_chat_template
from .config import CONFIG
from .utils import makedirs


def run_sft_pretrain(formatted_jsonl: str, model_name: str, out_lora_dir: str) -> str:
    """Run the SFT warm-up stage and persist the resulting LoRA adapter."""
    print("Preparing SFT dataset from:", formatted_jsonl)
    rows: List[Dict[str, object]] = []
    with open(formatted_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    markers = build_chat_template(tokenizer)

    texts: List[Dict[str, str]] = []
    for row in rows:
        formatted = apply_chat_template_to_messages(
            tokenizer,
            row["messages"],
            add_generation_prompt=False,
            tokenize=False,
        )
        texts.append({"text": formatted})

    hf_dataset = Dataset.from_list(texts)

    def tokenize_fn(batch: Dict[str, List[str]]):
        out = tokenizer(batch["text"], truncation=True, max_length=CONFIG["max_seq_length"])
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = hf_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print("Loading base model via transformers:", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    try:
        model = prepare_model_for_kbit_training(model)
    except Exception:
        pass

    lora_rank = CONFIG["lora_rank"]
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=os.path.join(CONFIG["checkpoints_dir"], out_lora_dir),
        per_device_train_batch_size=CONFIG["sft"]["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["sft"]["gradient_accumulation_steps"],
        num_train_epochs=CONFIG["sft"]["num_train_epochs"],
        learning_rate=CONFIG["sft"]["learning_rate"],
        logging_steps=CONFIG["sft"]["logging_steps"],
        fp16=True if torch.cuda.is_available() else False,
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("Starting SFT pre-finetune...")
    trainer.train()

    makedirs(CONFIG["checkpoints_dir"])
    out_path = os.path.join(CONFIG["checkpoints_dir"], out_lora_dir)
    print("Saving LoRA to", out_path)
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    print("Saved preformat LoRA to", out_path)
    return out_path


__all__ = ["run_sft_pretrain"]
