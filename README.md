## TinyRLHF

TinyRLHF is a lightweight, extensible toolkit for steering small language models (SLMs) with supervised fine-tuning (SFT), reinforcement learning from human feedback (RLHF), and direct preference optimization (DPO). The goal is to provide a single open-source home for experimenting with adapters, reward strategies, and training paradigms without locking you into a single backend such as Unsloth.

### Features

- Configurable pipelines for SFT, GRPO-style RLHF, and DPO, with room to plug in other algorithms like PPO or reward modeling.
- Dataset adapters that support multiple-choice, free-form generative, and pairwise preference data.
- Adapter abstractions that make it easy to switch between PEFT, Unsloth, or future LoRA providers.
- Pluggable reward sources, including placeholder hooks for LLM judges and learned reward models.
- Optional integrations for experiment tracking (Weights & Biases) and checkpoint management.

### Getting Started

```bash
pip install -e .[dev]

# Run a quick SFT smoke test
tiny-rlhf run --config configs/experiments/quickstart_sft.yaml
```

For detailed documentation, examples, and design notes, visit the `docs/` directory and the example projects in `examples/`.

### Repository Layout

See `docs/index.md` for a guided tour of the folder structure and how to contribute new pipelines, adapters, or datasets.