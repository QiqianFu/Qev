---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B-Base
base_model_relation: adapter
language:
  - en
  - zh
tags:
  - qev
  - qwen3_5
  - lora
  - decision-model
  - choice
  - noul
  - score
metrics:
  - accuracy
---

<p align="center">
  <img src="https://raw.githubusercontent.com/QiqianFu/Qev/main/assets/banner.svg" alt="Qev — decisions, grounded in Qwen" width="100%">
</p>

# Qev-9B

**Qev fine-tunes Qwen into a decision model.** Give it context, a question, and answer options; receive a decision and a probability for each option. One model supports **Choice**, **Noul** (yes/no), and **Score** (ordered ratings).

[Source and documentation](https://github.com/QiqianFu/Qev) · [中文说明](https://github.com/QiqianFu/Qev/blob/main/README.zh-CN.md) · [Model weights](https://huggingface.co/AustinFu/Qev-9B)

This is the **v0.1.0** release of the final Qev-9B research checkpoint: seed 17, step 2327, with the late1783 training mixture repeated three times during the final half of training. It is the full-interaction baseline A used in the architecture ablations.

## Quick start

Use Python 3.12 and install a hardware-compatible PyTorch 2.8.0 build, then install Qev from its source repository:

```bash
git clone https://github.com/QiqianFu/Qev.git
cd Qev
python -m pip install -e .
```

```python
from qev import Qev

model = Qev.from_pretrained(
    "AustinFu/Qev-9B", revision="v0.1.0", device="cuda"
)
answers = model.predict({
    "state": "I was charged twice. Please help immediately.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Charges and refunds", "shipping": "Delivery problems"},
        }
    },
})
print(answers["department"]["choice"])
print(answers["department"]["probabilities"])
```

The Qev loader downloads this checkpoint and the pinned Qwen base separately. Use the Qev Python or JSONL interface to load the adapter, decision head, and interaction gate together. The default execution uses shared-prefix caching. [All three task formats](https://github.com/QiqianFu/Qev/blob/main/examples/requests.jsonl).

For JSONL inference:

```bash
python -m qev.predict \
  --checkpoint AustinFu/Qev-9B@v0.1.0 \
  --input examples/requests.jsonl --out runs/predictions.jsonl \
  --device cuda --weights-dtype checkpoint
```

## Architecture and training

<p align="center">
  <img src="https://raw.githubusercontent.com/QiqianFu/Qev/main/assets/architecture.svg" alt="Qev encodes context, questions and answer options, then scores the options with its decision head." width="100%">
</p>

| Field | Released model |
|---|---|
| Base | Qwen/Qwen3.5-9B-Base |
| Base revision | `68c46c4b3498877f3ef123c856ecfde50c39f404` |
| Adaptation | LoRA rank 64, alpha 128; learned decision head and interaction gate |
| Decision head | Shared 4096→256 projection; two 4-head Transformer layers; scalar scorer |
| Backbone interaction | `last-full-attention` |
| Computation | BF16 backbone; FP32 decision head and key reductions |
| Stored adaptation tensors | FP32 |
| Main training partition | 34,546 records |
| Late partition | 1,419 alignment records + 364 rule-compliance judgments |
| Training schedule | Two epochs, global batch 32, seed 17 |
| Late mixing | Starts halfway through main training; late examples repeat three times |
| Selected checkpoint | Step 2327 |

The checkpoint was called BranchKev during research. Its run ID is `c21-science-wk-late1783-9b-4gpu-r64-late50x3-s17`. The main and late partitions intentionally share 249 replay records. The complete research training corpus is not distributed with this release. [Training guide](https://github.com/QiqianFu/Qev/blob/main/docs/training.md) · [Data recipe](https://github.com/QiqianFu/Qev/blob/main/docs/data.md).

## Evaluation

**Qev-9B uses BF16 backbone computation; Kev-9B uses FP32.**

<p align="center">
  <img src="https://raw.githubusercontent.com/QiqianFu/Qev/main/assets/evaluation.svg" alt="Qev and Kev accuracy on seven benchmarks." width="100%">
</p>

| Benchmark | Jev (reference) | Qwen3.5-9B-Base | Qev-9B | Kev-9B |
|---|---:|---:|---:|---:|
| Decision development · clean | 84.49 | 77.69 | **87.42** | 87.18 |
| Transfer development · clean | 85.67 | 74.39 | **83.99** | 82.16 |
| MMLU-Pro · 1,000 | 83.50 | 50.40 | **54.60** | 51.10 |
| SemIf · 144 handwritten | 96.53 | 90.28 | **93.75** | 90.97 |
| scienthoon · 873 | 75.26 | 68.84 | 72.28 | **75.49** |
| WANLI · 256 | 75.78 | 67.97 | **72.66** | 70.31 |
| JevBench public · 231 | 85.71 | 75.76 | **81.39** | 75.76 |

Accuracy (%). Bold compares Qev with Kev. JevBench is public-set accuracy: Qev answers 188 of 231 questions correctly. It is not the official JevBench composite score.

These are the recorded results for the released checkpoint, using full causal reference execution. The selected model is a single seed, and public benchmarks were observed during research iteration. The comparisons do not isolate architecture gains. [Results, sources, and reproduction commands](https://github.com/QiqianFu/Qev/blob/main/docs/evaluation.md).

## Files

The approximately 690 MiB inference package contains:

- `adapter/`: LoRA configuration and weights.
- `head.safetensors` and `joint.safetensors`: decision head and interaction gate.
- `model.json` and `tokenizer/`: Qev configuration and tokenizer.
- `SHA256SUMS.json`: hashes of the seven inference files.
- `provenance.json`, `training_config.json`, and `benchmarks.json`: checkpoint identity, training configuration, and recorded results.

The adaptation tensors are byte-identical to the selected research checkpoint. Qwen base weights are fetched separately at the revision above. Optimizer state is excluded; initialize new fine-tuning with `--init-checkpoint AustinFu/Qev-9B@v0.1.0`.

## Intended use and license

Qev supports research and development of routing, rule judgments, and rubric ratings over explicit options. Evaluate it on your application's inputs and decision thresholds. Probabilities depend on the supplied options; calibration and production reliability have not been established.

The released Qev adaptation weights and Qev code use Apache-2.0. The Qwen base is separately available under Apache-2.0. Source datasets retain their respective terms. Qev builds on Qwen and adapts conventions from [Jared Palmer's Kev](https://github.com/jaredpalmer/kev); attribution is retained in the included `LICENSE` and `NOTICE`.
