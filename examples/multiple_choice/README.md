# Multiple-Choice Example

This example demonstrates running SFT, GRPO, and DPO on a toy multiple-choice dataset.

1. Generate synthetic data:

   ```bash
   python examples/multiple_choice/prepare_data.py
   ```

2. Launch an experiment:

   ```bash
   tiny-rlhf run --config examples/multiple_choice/run_sft.yaml
   ```

Dataset files are written to `data/multiple_choice/`.
