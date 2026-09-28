# Checkpoint export and loading

A Qev LoRA checkpoint contains `model.json`, `adapter/`, `tokenizer/`, `head.safetensors`, and, when candidate interaction is enabled, `joint.safetensors`. Training checkpoints also contain `training.pt` with optimizer/RNG/cursor state.

## Export a research checkpoint

```bash
python -m qev.export --checkpoint /path/to/step-002327 \
  --out checkpoints/qev-9b \
  --base Qwen/Qwen3.5-9B-Base \
  --base-revision 68c46c4b3498877f3ef123c856ecfde50c39f404
```

The exporter accepts Qev and compatible BranchKev LoRA checkpoints. It copies inference tensors unchanged, rewrites machine-specific base references, writes the Qev format tag, and records hashes in `SHA256SUMS.json`. It excludes optimizer state and the original Qwen base weights. It requires an unused destination and a full base commit SHA. Full-FT checkpoints are loaded by Qev but are not handled by this LoRA release exporter.

## Local or Hub loading

```python
from qev import Qev

model = Qev.from_pretrained("checkpoints/qev-9b", device="cuda")
# For an already downloaded base:
model = Qev.from_pretrained(
    "checkpoints/qev-9b", device="cuda", base="/path/to/qwen-base"
)
```

A local override must contain the same base revision. Model structure checks cannot prove weight identity; retain the base's original provenance.

After hosting, pass `owner/repository@commit` or use the `revision` argument. Remote checkpoint downloads require an explicit revision. No public Qev Hub repository is configured in this source release, so the generic pattern is not a currently available download URL. Qev loads standard model/tokenizer files without `trust_remote_code`.

`weights_dtype="checkpoint"` retains the stored precision; `"fp32"` loads backbone and adapter tensors in FP32. Qev-9B's reported evaluation uses the former with `execution="reference"`. Defaults for the Python API and prediction CLI use cached inference. Formal benchmark commands explicitly set the execution and precision.

The API and CLI can raise `max_state` and `max_path` for evaluation. This does not retroactively extend the training distribution; record the overrides with results.

## Training semantics

`--init-checkpoint` uses model parameters on a new dataset with a fresh optimizer and schedule. `--resume` requires a full training checkpoint and the same data/admission hashes. An inference-only export is suitable for initialization, not optimizer-state recovery.
