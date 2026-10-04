# Loading and exporting checkpoints

## Load Qev-9B

```python
from qev import Qev

model = Qev.from_pretrained("AustinFu/Qev-9B", device="cuda")
```

The loader downloads the approximately 690 MiB Qev package and its Qwen base model. The model configuration records the compatible base version. The current 9B release is `v0.3.0` (MMLU-Pro 56.50%, JevBench 187/231), with 4,096-token state and complete-path limits. Use `revision="v0.3.0"` to pin it. This is also the teacher used for Qev-4B. The previous `v0.2.0` release and the original `v0.1.0` teacher used for Qev-2B remain available. Omitting `revision` loads the current release.

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

CLI commands accept either a local directory or `AustinFu/Qev-9B`. Append `@v0.3.0` to select the current 9B release explicitly. Qev does not need `trust_remote_code`.

## Load Qev-4B

```python
model = Qev.from_pretrained("AustinFu/Qev-4B", device="cuda", revision="v0.1.0")
```

The approximately 524 MiB adaptation package contains the model trained directly from Qwen3.5-4B-Base for 2,786 steps. The loader fetches the pinned base separately. Its default state and complete-path limits are both 4,096 tokens.

```bash
hf download AustinFu/Qev-4B --revision v0.1.0 --local-dir checkpoints/qev-4b
```

CLI commands accept `AustinFu/Qev-4B@v0.1.0` with the same request format as 2B and 9B. See the [4B model card](model-card-4b.md) and [training method](training-4b.md).

## Load Qev-2B

Qev-2B uses the same loader and request format:

```python
model = Qev.from_pretrained("AustinFu/Qev-2B", device="cuda")
```

The approximately 284 MiB adaptation package contains the selected 800-step response-distillation model. The compatible Qwen3.5-2B base downloads separately. Use `revision="v0.1.0"` to select the first release, or download it locally:

```bash
hf download AustinFu/Qev-2B --local-dir checkpoints/qev-2b
```

CLI commands also accept `AustinFu/Qev-2B` or `checkpoints/qev-2b`. See the [2B model card](model-card-2b.md) and [distillation guide](distillation.md).

## Export a trained model

```bash
python -m qev.export --checkpoint runs/support/step-000100 \
  --out checkpoints/support
```

Use an actual saved training step and a new output directory. The exporter copies the LoRA adapter, decision head, interaction gate, tokenizer and model configuration. It omits optimizer state and base weights. License files present in the source checkpoint are copied with the model.

If training used a local base directory, supply its public model name with `--base Qwen/Qwen3.5-9B-Base`. `--base-revision` optionally selects a base version. The exporter accepts both current Qev checkpoints and the earlier compatible checkpoint format. Full fine-tuning checkpoints already contain their base weights and do not use this LoRA exporter.

A Qev checkpoint uses `model.json`, `adapter/`, `tokenizer/`, `head.safetensors`, and, when candidate interaction is enabled, `joint.safetensors`. Loading the full package restores all trained components.

## Precision and execution

`weights_dtype="checkpoint"` uses the configured precision. `weights_dtype="fp32"` loads the backbone and adapter in FP32. Qev-2B, Qev-4B and Qev-9B use BF16 backbone computation and an FP32 decision head.

The Python API and prediction CLI use shared-prefix caching by default. Set `execution="reference"` in Python or `--reference` in the CLI for the execution used in the benchmark reports. Optional `max_state` and `max_path` settings control accepted input sizes.

## Continue training

`--init-checkpoint` starts a new fine-tuning run from model weights with a fresh optimizer and schedule. `--resume` restores an interrupted run, including optimizer state and the training position, and checks that its data and settings are unchanged. Inference exports support initialization; resuming requires the full training checkpoint.

See [training](training.md) for examples and [licensing](../THIRD_PARTY_NOTICES.md) for the terms covering Qev weights, the Qwen base and tokenizer.
