# Reward Models & Judges

Rewards can combine deterministic format checks, learned models, and external judges.

- `tiny_rlhf.rewards.formatting` implements answer-shape validators (e.g., label-only responses for multiple-choice tasks).
- `tiny_rlhf.rewards.learned_reward` wraps a trainable reward model that can be fine-tuned separately.
- `tiny_rlhf.rewards.judge_client` routes prompts/responses to an LLM judge backend.

Judge implementations live in `tiny_rlhf.judges`. The default placeholder returns neutral scores so that pipelines run without external dependencies. Replace it with integrations for OpenAI, local VLLM deployments, or custom APIs.
