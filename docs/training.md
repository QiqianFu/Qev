# Training Qev

All commands run from the repository root. Python 3.12 and the pinned dependencies in `pyproject.toml` match the tested CPU environment and the research training stack. Install a hardware-compatible PyTorch 2.8.0 wheel first. CUDA acceleration packages such as compatible flash-linear-attention kernels are optional; identify the actual execution path when reporting speed.

## Prepare user data

```bash
python -m qev.prepare --input examples/train.jsonl \
  --validation examples/dev.jsonl --out data/support
```

Without `--validation`, the preparer makes a deterministic group split (`--validation-fraction 0.1 --seed 17`). It refuses duplicate IDs, overlapping groups, and exactly repeated record inputs across train/dev. This is an exact guard, not semantic deduplication. Put related examples in the same `group_id` before splitting. Existing output directories are never overwritten.

The trainer consumes `manifest.json` and a `train` partition, checks file hashes and roles, and records all over-length rejections. Records are not silently truncated. All training questions need targets.

## Fine-tune from Qev-9B

```bash
python -m qev.train --config configs/qev-9b-finetune.json \
  --data data/support --out runs/support \
  --init-checkpoint checkpoints/qev-9b
```

This loads LoRA, the set head and the candidate-interaction gate, and starts a fresh optimizer/schedule. Calibration is reset to temperature 1. The configuration must match the checkpoint's model structure. Training data and token limits can differ; `initialization.json` records the source metadata hash.

The single-GPU example uses batch 1 and accumulation 32. This is a configuration example, not a measured minimum-VRAM promise. Adjust context and batching after a short run. The included eight records illustrate formatting and cannot establish fine-tuning gains.

## Train from the base

Omit `--init-checkpoint` to initialize new LoRA and head parameters on the configured Qwen base. Remote bases need a pinned revision. For a local base cache, set `model.base` to that directory and `model.revision` to null in a new config.

The formal 9B recipe is:

```bash
torchrun --standalone --nproc_per_node=4 -m qev.train \
  --config configs/qev-9b.json --data data/qev-research \
  --out runs/qev-9b
```

`data/qev-research` must contain the research `train` and `late_train` partitions. It is not supplied by `qev.prepare` from the toy examples. [Data composition and release boundary](data.md).

| Setting | Selected checkpoint |
|---|---|
| Base | Qwen3.5-9B-Base, `68c46c4b3498877f3ef123c856ecfde50c39f404` |
| Adaptation | LoRA rank 64, alpha 128 |
| Head | 256 channels, 4 attention heads, 2 layers |
| Candidate interaction | `last-full-attention` |
| Main / late records | 34,546 / 1,783 |
| Schedule | 2 epochs, seed 17, global batch 32 |
| Late mixing | Final 50% of main-training steps; late records repeated 3 times |
| Execution | `tree-batched`, BF16 backbone, FP32 head/reductions |
| Selected step | 2327 |

`late_fraction` describes when late mixing starts; it is not the fraction of late examples in the resulting dataset. Changing GPU count requires adjusting microbatch/accumulation to keep the global batch fixed.

## Continue the same run

```bash
python -m qev.train --config runs/support/config.json \
  --data data/support --out runs/support \
  --resume runs/support/step-000100
```

Use an actual saved step. Resume restores optimizer, schedule, RNG and data cursor, and requires matching data/admission hashes and settings. `--allow-repartition` permits supported changes of rank/microbatch partition at unchanged global batch, with zero dropout and other checks. Late-split repartition is not supported.

`--max-steps` is the cumulative optimizer-step stopping point for that invocation. `--wall-hours` stops at an optimizer boundary. Exported inference-only checkpoints have no optimizer state and cannot be used with `--resume`; use `--init-checkpoint`.

## Advanced recipes

- `qev-0.8b.json`: the smaller research configuration; no 0.8B checkpoint is presented as the selected Qev-9B release.
- `qev-9b-independent.json`: no extra backbone cross interaction, zero set-attention layers; projection and shared scorer remain.
- `qev-9b-readout-cross.json`: sibling interaction only reads terminal readouts.
- `qev-9b-fullft.json`: FSDP2 full fine-tuning with layer-wise learning-rate decay. Its historical recipe used knowledge-v1, not the selected Qev-9B data. Requires CUDA; full-FT resume and `--init-checkpoint` are currently unsupported.

## Validation

```bash
python scripts/smoke.py --out runs/smoke
python -m pytest -q
```

The offline smoke run creates a tiny random hybrid Qwen3.5 model and exercises data preparation, training, resume, inference and evaluation. Full tests include CPU DDP subprocesses and GPU-only FSDP tests. [Actual validation record](validation.md).
