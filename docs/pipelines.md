# Pipelines

Each pipeline coordinates dataset preparation, model loading, adapter configuration, training, evaluation, and logging.

## SFT (`tiny_rlhf.pipelines.sft`)

1. Load dataset via the configured adapter.
2. Initialize the base model backend (Transformers or Unsloth).
3. Attach LoRA adapters if specified.
4. Train using the `SFTTrainer` wrapper.
5. Evaluate on the validation split and persist artifacts.

## GRPO (`tiny_rlhf.pipelines.grpo`)

1. Load dataset and prepare prompts.
2. Generate responses with the current policy.
3. Score responses using composite rewards (format checks + judge or learned reward model).
4. Update policy via GRPO trainer.
5. Log metrics and intermediate checkpoints.

## DPO (`tiny_rlhf.pipelines.dpo`)

1. Load preference pairs.
2. Initialize reference and policy models (sharing weights when possible).
3. Optimize the DPO objective with configurable beta and LoRA adapters.
4. Export adapters and evaluation summaries.

Additional pipelines (e.g., PPO, reward modeling) can be added by implementing a `run(config)` function and registering it in `configs/registry.yaml`.
