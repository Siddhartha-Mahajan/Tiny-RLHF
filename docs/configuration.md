# Configuration System

TinyRLHF uses YAML configuration files composed from reusable templates.

- `configs/templates/` contain building blocks for datasets, models, adapters, and trainers.
- `configs/registry.yaml` maps logical names to pipeline entrypoints.
- Experiment configs in `configs/experiments/` specify which pipeline to run and supply custom overrides.

At runtime, the loader resolves relative paths, merges templates, and validates the resulting configuration with Pydantic models defined in `tiny_rlhf.config.schema`.

## Environment Overrides

Any configuration value can be overridden with environment variables using the `TINY_RLHF__SECTION__FIELD` naming convention. For example, to override the learning rate:

```bash
export TINY_RLHF__TRAINER__LEARNING_RATE=5e-5
```

## CLI Overrides

CLI arguments passed after `--set` are applied as OmegaConf-style dot lists:

```bash
tiny-rlhf run --config configs/experiments/quickstart_sft.yaml --set trainer.learning_rate=1e-4
```
