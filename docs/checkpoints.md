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

## Load Qev-0.8B

```python
model = Qev.from_pretrained("AustinFu/Qev-0.8B", device="cuda", revision="v0.1.0")
```

The approximately 192 MiB adaptation package contains the seed-17, step-2,786 model distilled directly from Qwen3.5-0.8B-Base. The pinned Qwen base downloads separately; inference does not need the teacher. Its state and complete-path limits are 4,096 tokens.

```bash
hf download AustinFu/Qev-0.8B --revision v0.1.0 --local-dir checkpoints/qev-0.8b
```

CLI commands accept `AustinFu/Qev-0.8B@v0.1.0`. See the [model card](model-card-0.8b.md) and [training method](training-0.8b.md).

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

`weights_dtype="checkpoint"` uses the configured precision. `weights_dtype="fp32"` loads the backbone and adapter in FP32. All Qev sizes use BF16 backbone computation and an FP32 decision head.

The Python API and prediction CLI use shared-prefix caching by default. Set `execution="reference"` in Python or `--reference` in the CLI for the execution used in the benchmark reports. Optional `max_state` and `max_path` settings control accepted input sizes. `max_path` counts the state too, so raising only `max_state` raises `max_path` by the same amount. States longer than a checkpoint's released limit run, but the models were trained within it; with long unrelated text before JevBench states, Qev-9B accuracy fell from 81.0% to 66.7–81.0% ([long unrelated prefix](validation.md#long-unrelated-prefix-on-jevbench--2026-10-27)). Reference and tree execution process each full path (or packed tree) in one forward and reject inputs over the model's `max_padded_tokens` budget (8,192 tokens); cached execution and sessions have no such limit.

## Incremental sessions

`model.session(state)` returns a `QevSession` for a state that grows over time. `session.extend(text)` appends a chunk; `session.predict(questions)` takes the `questions` mapping of a request and returns the same output as `model.predict`.

With the default `cached` execution the session keeps a cache that contains only state tokens: full-attention KV and Gated DeltaNet convolution/recurrent state. Pending chunks are appended to it on the next prediction, so each state token is computed once per session. Each question runs on a fork of the cache, which is never modified by questions. Sessions created from a `reference` or `tree` model recompute the full state for every prediction and serve as numerical checks.

- **Tokenization.** The state is the concatenation of each chunk's own tokens. Tokenizing the joined text in one request can split a chunk boundary differently, so session predictions can differ from a one-shot request. In the GPU check, appending `"!"` after a sentence changed the tokens: Qev-0.8B then differed by up to 0.14 in probability and changed one prediction, while Qev-9B differed by up to 0.03. Starting a chunk with whitespace or a newline usually avoids this but does not guarantee it. Released models were trained on one-shot tokenization; incremental use has not been benchmarked.
- **Limits.** The state is checked against `max_state` on every `extend`; an overflow raises `ContextOverflow` and leaves the session unchanged. The session cannot drop old tokens, because the recurrent state cannot remove a prefix; start a new session instead.
- **Memory.** A session holds its cache until `close()`. Full-attention KV grows with the state, about 32 KiB per token for Qev-9B in BF16 (1.6 GB at 50K tokens); the DeltaNet state has a fixed size. Questions do not copy that KV. A question fork and its candidate forks reference the state KV (and the question's own KV) in place and store only the tokens they compute; only the fixed-size DeltaNet state is copied per fork. Their attention reads the shared prefix once for all candidate rows and merges it with attention over their own tokens by log-sum-exp. Extra memory for a question therefore depends on question and candidate lengths and `rows_per_forward`, not on state length times candidates; the FP32 readout of a candidate-interaction model holds prefix scores for at most `rows_per_forward` candidates at a time. State tokens are computed in forwards of at most `prefill_chunk` tokens (default 8,192; set it with `Qev.from_pretrained(..., prefill_chunk=...)`), so ingesting a long state needs activations for one chunk; one-shot cached requests chunk the same way. If a forward pass fails, the cache is discarded and rebuilt from the full state on the next prediction.
- **Concurrency.** A session is not safe for concurrent use. Caches are not shared between sessions or with `model.predict`.

## Continue training

`--init-checkpoint` starts a new fine-tuning run from model weights with a fresh optimizer and schedule. `--resume` restores an interrupted run, including optimizer state and the training position, and checks that its data and settings are unchanged. Inference exports support initialization; resuming requires the full training checkpoint.

See [training](training.md) for examples and [licensing](../THIRD_PARTY_NOTICES.md) for the terms covering Qev weights, the Qwen base and tokenizer.
