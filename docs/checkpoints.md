# Loading and exporting checkpoints

## Load Qev-9B

```python
from qev import Qev

model = Qev.from_pretrained("AustinFu/Qev-9B", device="cuda")
```

The loader downloads the approximately 690 MiB Qev package and its Qwen base model. The model configuration records the compatible base version. Use `revision="v0.1.0"` when you want the original published release; `revision` is optional for normal use.

For a local download:

```bash
hf download AustinFu/Qev-9B --local-dir checkpoints/qev-9b
```

```python
from qev import Qev

model = Qev.from_pretrained("checkpoints/qev-9b", device="cuda")
# Reuse an existing copy of the compatible Qwen base:
model = Qev.from_pretrained(
    "checkpoints/qev-9b", device="cuda", base="/path/to/qwen-base"
)
```

CLI commands accept either a local directory or `AustinFu/Qev-9B`. Append `@v0.1.0` to select a particular release. Qev does not need `trust_remote_code`.

## Export a trained model

```bash
python -m qev.export --checkpoint runs/support/step-000100 \
  --out checkpoints/support
```

Use an actual saved training step and a new output directory. The exporter copies the LoRA adapter, decision head, interaction gate, tokenizer and model configuration. It omits optimizer state and base weights. License files present in the source checkpoint are copied with the model.

If training used a local base directory, supply its public model name with `--base Qwen/Qwen3.5-9B-Base`. `--base-revision` optionally selects a base version. The exporter accepts both current Qev checkpoints and the earlier compatible checkpoint format. Full fine-tuning checkpoints already contain their base weights and do not use this LoRA exporter.

A Qev checkpoint uses `model.json`, `adapter/`, `tokenizer/`, `head.safetensors`, and, when candidate interaction is enabled, `joint.safetensors`. Loading the full package restores all trained components.

## Precision and execution

`weights_dtype="checkpoint"` uses the configured precision. `weights_dtype="fp32"` loads the backbone and adapter in FP32. Qev-9B uses BF16 backbone computation and an FP32 decision head.

The Python API and prediction CLI use shared-prefix caching by default. Set `execution="reference"` in Python or `--reference` in the CLI for the execution used in the benchmark reports. Optional `max_state` and `max_path` settings control accepted input sizes.

## Continue training

`--init-checkpoint` starts a new fine-tuning run from model weights with a fresh optimizer and schedule. `--resume` restores an interrupted run, including optimizer state and the training position, and checks that its data and settings are unchanged. Inference exports support initialization; resuming requires the full training checkpoint.

See [training](training.md) for examples and [licensing](../THIRD_PARTY_NOTICES.md) for the terms covering Qev weights, the Qwen base and tokenizer.
