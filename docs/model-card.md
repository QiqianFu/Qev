---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B-Base
base_model_relation: adapter
datasets:
  - AustinFu/Qev-train
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

Qev-9B v0.3.0 combines general decisions, Principle judgments and web-action training with 4K context. Alignment, document-rule, boundary and additional rule/reasoning examples are repeated three times during the second half of training. It uses the full candidate-interaction architecture shown below.

This checkpoint is also the teacher used to train Qev-4B v0.1.0. The previous 9B weights remain available at [v0.2.0](https://huggingface.co/AustinFu/Qev-9B/tree/v0.2.0), and the original Qev-2B teacher at [v0.1.0](https://huggingface.co/AustinFu/Qev-9B/tree/v0.1.0).

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
    "AustinFu/Qev-9B", revision="v0.3.0", device="cuda"
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
  --checkpoint AustinFu/Qev-9B@v0.3.0 \
  --input examples/requests.jsonl --out runs/predictions.jsonl \
  --device cuda --weights-dtype checkpoint
```

## Architecture and training

<p align="center">
  <a href="https://raw.githubusercontent.com/QiqianFu/Qev/main/assets/method.pdf"><img src="https://raw.githubusercontent.com/QiqianFu/Qev/main/assets/method.svg" alt="Qev branch encoding, final-layer candidate interaction and set decision head." width="100%"></a>
</p>

| Field | Released model |
|---|---|
| Base | Qwen/Qwen3.5-9B-Base |
| Adaptation | LoRA rank 64, alpha 128; learned decision head and interaction gate |
| Decision head | Shared 4096→256 projection; two 4-head Transformer layers; scalar scorer |
| Backbone interaction | `last-full-attention` |
| Computation | BF16 backbone; FP32 decision head and key reductions |
| Stored adaptation tensors | FP32 |
| Model release | v0.3.0 |
| Input limits | State 4,096; question 512; candidate 256; complete path 4,096 tokens |
| Main training partition | 38,198 records, including 2,230 Principle judgments and 1,422 web-action records |
| Late partition | 2,883 records: 1,419 alignment, 364 document rules, 600 boundary and 500 additional rule/reasoning questions |
| Training schedule | Two epochs, global batch 32, seed 17 |
| Late mixing | Starts halfway through main training; late examples repeat three times |
| Selected checkpoint | Step 2658 |

The main and late partitions intentionally share 249 replay records. [Qev-train](https://huggingface.co/datasets/AustinFu/Qev-train) publishes 2,442 synthetic alignment, rule-compliance, world-knowledge and HelpSteer3-derived boundary examples, with generation methods and source-specific licenses. The complete mixed training corpus, including the added web-action and rule/reasoning records, is not distributed. [Training guide](https://github.com/QiqianFu/Qev/blob/main/docs/training.md) · [Data recipe](https://github.com/QiqianFu/Qev/blob/main/docs/data.md).

## Evaluation

**Qev-9B uses BF16 backbone computation; Kev-9B uses FP32.**

<p align="center">
  <img src="https://raw.githubusercontent.com/QiqianFu/Qev/main/assets/evaluation-matrix.svg" alt="Qev models and reference models across ten benchmarks." width="100%">
</p>

| Benchmark | Jev (reference) | Qev-9B | Kev-9B |
| --- | ---: | ---: | ---: |
| JevBench public · 231 | 85.71 | **80.95** | 75.76 |
| Decision development · clean | 84.49 | **87.58** | 87.18 |
| Transfer development · clean | 85.67 | **83.69** | 82.16 |
| MMLU-Pro · 1,000 | 83.50 | **56.50** | 51.10 |
| SemIf · 144 handwritten | 96.53 | **93.75** | 90.97 |
| scienthoon · 873 | 75.26 | 74.80 | **75.49** |
| WANLI · 256 | 75.78 | **75.00** | 70.31 |
| GSM8K · multiple choice | 79.87 | **61.75** | 46.36 |
| ChessBench | 17.22 | 10.34 | **11.76** |
| BPoMP · variant mean | 90.92 | **81.03** | 66.93 |

Scores (%). The new GSM8K, ChessBench and BPoMP rows use [Decision Index raw scores](https://github.com/QiqianFu/Qev/blob/main/docs/evaluation.md#decision-index-021); the original rows use accuracy. Bold compares Qev with Kev.

These are the recorded results for the released checkpoint, using full causal reference execution. The selected model is a single seed, and public benchmarks were observed during research iteration. The comparisons do not isolate architecture gains. [Results, sources, and reproduction commands](https://github.com/QiqianFu/Qev/blob/main/docs/evaluation.md).

## Files

The approximately 690 MiB inference package contains:

- `adapter/`: LoRA configuration and weights.
- `head.safetensors` and `joint.safetensors`: decision head and interaction gate.
- `model.json` and `tokenizer/`: Qev configuration and tokenizer.
- `training_config.json` and `benchmarks.json`: training configuration and recorded results.
- `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, and `licenses/`: license terms and attribution.

The adaptation tensors are byte-identical to the selected research checkpoint. Qwen base weights are fetched separately at the version recorded in the model configuration. Optimizer state is excluded; initialize new fine-tuning with `--init-checkpoint AustinFu/Qev-9B`.

## Intended use and license

Qev supports research and development of routing, rule judgments, and rubric ratings over explicit options. Evaluate it on your application's inputs and decision thresholds. Probabilities depend on the supplied options; calibration and production reliability have not been established.

Qev's code, adaptation weights, documentation and original illustrations use Apache-2.0. Qwen models and the bundled tokenizer retain Alibaba Cloud's Apache-2.0 license. Qev adapts conventions from [Jared Palmer's Kev](https://github.com/jaredpalmer/kev), whose Apache-2.0 license and attribution are retained. JevBench tasks and external dependencies retain their own terms; the complete mixed training corpus is not bundled. The synthetic subset is published separately as Qev-train under its component-specific licenses. See the included `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, and `licenses/` for the full texts and scope.
