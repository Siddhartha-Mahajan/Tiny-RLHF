"""Supervised fine-tuning trainer."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset
from trl import SFTConfig, SFTTrainer as TRLSFTTrainer

from tiny_rlhf.data import DatasetSplits
from tiny_rlhf.models import ModelHandle
from tiny_rlhf.utils.chat import render_chat_messages

from .base import BaseTrainer, TrainResult

logger = logging.getLogger(__name__)


class SFTTrainer(BaseTrainer):
    def __init__(self, model: ModelHandle, config):
        super().__init__(model, config)
        self._trl_trainer: Optional[TRLSFTTrainer] = None
        self._train_dataset_hf: Optional[Dataset] = None
        self._eval_dataset_hf: Optional[Dataset] = None

    def _build_text_records(self, split: List[Dict], include_assistant: bool = True) -> List[Dict[str, str]]:
        tokenizer = self.model.tokenizer
        records: List[Dict[str, str]] = []
        for example in split:
            messages = example.get("messages", [])
            if include_assistant:
                has_assistant = any(msg.get("role") == "assistant" for msg in messages)
                if not has_assistant:
                    continue
            text = render_chat_messages(tokenizer, messages, add_generation_prompt=False)
            records.append({"text": text, "id": example.get("id")})
        return records

    def train(self, dataset: DatasetSplits) -> TrainResult:
        logger.info("Starting SFT training with %d raw examples", len(dataset.train))

        train_records = self._build_text_records(dataset.train, include_assistant=True)
        if not train_records:
            raise RuntimeError("No supervised records with assistant responses available for SFT training.")

        eval_records = self._build_text_records(dataset.validation, include_assistant=True)

        self._train_dataset_hf = Dataset.from_list(train_records)
        self._eval_dataset_hf = Dataset.from_list(eval_records) if eval_records else None

        max_seq_length = 2048
        if self.dataset_metadata and self.dataset_metadata.get("max_seq_length"):
            max_seq_length = int(self.dataset_metadata["max_seq_length"])

        output_dir = Path(self.experiment.run.output_dir) / "sft"
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = SFTConfig(
            output_dir=str(output_dir),
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            packing=False,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_train_epochs or 1,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio or 0.0,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy="steps" if self._eval_dataset_hf and self.config.eval_steps else "no",
            eval_steps=self.config.eval_steps,
            max_grad_norm=None,
            report_to="wandb" if self.experiment.run.logging.use_wandb else "none",
        )

        self._trl_trainer = TRLSFTTrainer(
            model=self.model.model,
            args=training_args,
            tokenizer=self.model.tokenizer,
            train_dataset=self._train_dataset_hf,
            eval_dataset=self._eval_dataset_hf,
        )

        train_output = self._trl_trainer.train()
        metrics = {"train_loss": train_output.training_loss}
        if self._eval_dataset_hf and self.config.eval_steps:
            eval_metrics = self._trl_trainer.evaluate()
            metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})

        artifacts = {"output_dir": str(output_dir)}
        return TrainResult(metrics=metrics, artifacts=artifacts)

    def evaluate(self, dataset: DatasetSplits) -> Dict[str, float]:
        if self._trl_trainer is None:
            logger.warning("SFT trainer evaluate called before train; building temporary dataset.")
            temp_records = self._build_text_records(dataset.validation, include_assistant=True)
            if not temp_records:
                return {}
            eval_dataset = Dataset.from_list(temp_records)
            eval_output_dir = Path(self.experiment.run.output_dir) / "sft-eval"
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            temp_trainer = TRLSFTTrainer(
                model=self.model.model,
                args=SFTConfig(
                    output_dir=str(eval_output_dir),
                    dataset_text_field="text",
                    max_seq_length=self.dataset_metadata.get("max_seq_length", 2048) if self.dataset_metadata else 2048,
                    per_device_train_batch_size=self.config.per_device_train_batch_size,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    logging_steps=self.config.logging_steps,
                    report_to="none",
                ),
                tokenizer=self.model.tokenizer,
                train_dataset=eval_dataset,
                eval_dataset=eval_dataset,
            )
            return temp_trainer.evaluate()

        if not self._eval_dataset_hf:
            logger.info("No evaluation dataset available for SFT trainer.")
            return {}

        return self._trl_trainer.evaluate(self._eval_dataset_hf)
