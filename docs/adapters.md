# Adapter Support

TinyRLHF abstracts adapter handling through the `tiny_rlhf.models.lora` package.

- **PEFT LoRA**: default option using Hugging Face PEFT; supports rank, alpha, dropout configuration.
- **Unsloth LoRA**: optional backend for fast LoRA compilation; install with `pip install tiny-rlhf[unsloth]`.
- **Custom adapters**: implement `LoRAAdapter` from `tiny_rlhf.models.lora.base` and register it in `tiny_rlhf.models.lora.registry`.

Configuration fields under the `lora` key select the provider and parameters. Pipelines automatically attach adapters during model initialization.
