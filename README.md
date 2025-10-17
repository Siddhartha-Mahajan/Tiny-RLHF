## TinyRLHF (minimal)

This repository now contains a compact set of utilities for training small language models with three essential algorithms:

- **SFT** – supervised fine-tuning on chat or instruction data.
- **DPO** – direct preference optimisation on pairwise preferences.
- **GRPO** – reward-based optimisation using the implementation available in `trl`.

Everything else has been stripped away: no Hydra, no CLI launcher, no heavy registries. You load a YAML file, get a typed configuration object, and call the training function you need.

### Installation

```bash
pip install -e .
```

PyTorch, `transformers`, `trl`, `datasets`, and `peft` are required and listed in `pyproject.toml`. Enable optional Weights & Biases logging with `pip install -e .[wandb]`.

### Quick start

```python
from tiny_rlhf import load_experiment, run_experiment

config = load_experiment("examples/sft.yaml")
metrics = run_experiment(config)
print(metrics)
```

The helper will select the correct trainer based on `trainer.algorithm` inside the YAML.

### Minimal configuration schema

```yaml
dataset:
	format: chat            # chat | instruction | preference | grpo
	train_file: data/train.jsonl
	validation_file: data/val.jsonl
model:
	pretrained: meta-llama/Llama-3.1-8B-Instruct
	use_peft: true
	target_modules: [q_proj, v_proj]
trainer:
	algorithm: sft          # sft | dpo | grpo
	per_device_train_batch_size: 2
	gradient_accumulation_steps: 8
run:
	output_dir: outputs/run1
	use_wandb: false
reward:                   # optional, only used for GRPO
	type: exact_match
```

Data expectations by algorithm:

- **SFT** (`chat`): each JSONL row needs a `messages` list of `{role, content}` objects.
- **SFT** (`instruction`): rows contain `prompt` and `completion` strings.
- **DPO**: rows contain `prompt`, `chosen`, and `rejected` strings.
- **GRPO**: rows contain `prompt` (or `messages`) and optional `answer` strings used by the default exact-match reward.

### What remains

- `tiny_rlhf.config` – tiny dataclasses + `load_experiment` helper.
- `tiny_rlhf.data` – JSONL readers that map onto Hugging Face datasets.
- `tiny_rlhf.modeling` – AutoModel loader with optional PEFT LoRA wrapping.
- `tiny_rlhf.training` – thin wrappers around `trl`'s `SFTTrainer`, `DPOTrainer`, and `GRPOTrainer`.
- `tiny_rlhf.rewards` – minimal reward builders (exact match, keyword, or Python script hook).

That's it. Extend the dataclasses or swap in your own reward scripts as needed.