# Quickstart

1. **Install dependencies**

   ```bash
   pip install -e .[dev]
   ```

2. **Prepare a dataset**

   Place dataset files under `data/` using one of the supported adapters (multiple-choice, freeform, or preference pairs). See the templates in `configs/templates/` for field expectations.

3. **Choose an experiment config**

   ```bash
   tiny-rlhf run --config configs/experiments/quickstart_sft.yaml
   ```

4. **Inspect outputs**

   - Check logs in `outputs/<run_name>/events.log`.
   - Saved checkpoints and LoRA adapters live under `outputs/<run_name>/checkpoints/`.
   - If Weights & Biases logging is enabled, visit the linked run page.

5. **Customize**

   Duplicate one of the `quickstart_*` configs and adjust dataset paths, model names, trainer options, adapters, or reward setup.
