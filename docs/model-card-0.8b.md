---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B-Base
base_model_relation: adapter
language:
  - en
  - zh
tags:
  - qev
  - lora
  - decision-model
  - knowledge-distillation
---

<p align="center"><img src="https://raw.githubusercontent.com/QiqianFu/Qev/main/assets/banner.svg" alt="Qev — decisions, grounded in Qwen" width="100%"></p>

# Qev-0.8B

Qev-0.8B is the smallest Qev decision model. Give it context, a question and candidate answers; receive a choice and a probability for every option. It supports **Choice**, **Noul** (yes/no) and **Score** (ordered ratings) through the same API as Qev-2B, Qev-4B and Qev-9B.

[Source and installation](https://github.com/QiqianFu/Qev#installation) · [Training method](https://github.com/QiqianFu/Qev/blob/main/docs/training-0.8b.md) · [中文训练说明](https://github.com/QiqianFu/Qev/blob/main/docs/training-0.8b.zh-CN.md)

This model starts **directly from Qwen3.5-0.8B-Base**, with fresh rank-64 LoRA, a decision head and a final-layer interaction gate. It learns Qev-9B v0.3.0 option probabilities for two epochs. There is no supervised warm-start, hard-label mixture or representation-response continuation.

| Property | Qev-0.8B v0.1.0 |
|---|---|
| Base | Qwen3.5-0.8B-Base, revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |
| Adaptation | LoRA rank 64, alpha 128 |
| Decision head | 256 dimensions, four attention heads, two Transformer layers |
| Candidate interaction | Full sibling interaction in the final full-attention layer |
| Teacher | [Qev-9B v0.3.0](https://huggingface.co/AustinFu/Qev-9B/tree/v0.3.0) |
| Training | Teacher-probability cross entropy; 44,576 single-question inputs |
| Schedule | Two epochs, 2,786 steps, two GPUs, global batch 32, seed 17 |
| Temperatures | Teacher 1.563437713227029; student 1 |
| Input limits | 4,096-token state and complete path; question 512, candidate 256 |
| Computation | BF16 backbone, FP32 decision head |
| Download | Approximately 192 MiB; the Qwen base downloads separately |

## Use the model

Install the [Qev source package](https://github.com/QiqianFu/Qev#installation) with Python 3.12 and a hardware-compatible PyTorch 2.8.0 build:

```python
from qev import Qev

model = Qev.from_pretrained("AustinFu/Qev-0.8B", revision="v0.1.0", device="cuda")
answers = model.predict({
    "state": "I was charged twice for the same order.",
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

For JSONL requests:

```bash
python -m qev.predict --checkpoint AustinFu/Qev-0.8B@v0.1.0 \
  --input examples/requests.jsonl --out runs/0.8b-predictions.jsonl \
  --device cuda --weights-dtype checkpoint
```

The loader restores the adapter, decision head, interaction gate and tokenizer, and downloads the pinned Qwen base. No teacher is needed for inference. The package includes training/fine-tuning configurations, evaluation results and license files. Optimizer state and base weights are excluded.

## Recorded evaluation

| Benchmark | Correct / total | Accuracy (%) |
|---|---:|---:|
| JevBench public | 169 / 231 | 73.16 |
| Decision development · clean | 1029 / 1264 | 81.41 |
| Transfer development · clean | 431 / 656 | 65.70 |
| MMLU-Pro | 247 / 1000 | 24.70 |
| SemIf · handwritten | 109 / 144 | 75.69 |
| scienthoon | 596 / 873 | 68.27 |
| WANLI | 169 / 256 | 66.02 |

Results use BF16 backbone computation, an FP32 decision head, temperature 1 and full causal reference execution. All 4,844 questions in the seven complete suites were answered. Development and SemIf rows use the matched subsets in the [Qev evaluation guide](https://github.com/QiqianFu/Qev/blob/main/docs/evaluation.md); `evaluation.json` preserves the full-suite counts. GSM8K, ChessBench and BPoMP have not been measured for this checkpoint.

Both tested distillation seeds reached 169/231 on JevBench; this release selects seed 17. Distillation improved probability quality but did not beat supervised training on every accuracy metric. The Qev releases have different data and objectives, so their results do not isolate model size alone. NanoJev-0.6B is also included in the source repository as a differently trained small-model reference.

## Training and availability

The 44,576-input pool and teacher targets are shared with Qev-4B: general decisions, science and reasoning, Principle judgments, controlled boundaries, web actions and additional rule/reasoning tasks. Each record contains one question. Training uses a single pool with no late-stage repetition or online option edits.

The full training input pool and cached teacher outputs are not distributed. [Qev-train](https://huggingface.co/datasets/AustinFu/Qev-train) remains a separate release of 2,442 synthetic examples with hard labels and synthesis documentation. The source repository provides a runnable training example using the original teacher and user data.

Use `configs/qev-0.8b-finetune.json` with `python -m qev.train --init-checkpoint AustinFu/Qev-0.8B@v0.1.0` for supervised adaptation. Validate performance and probabilities on your own task; calibration and production reliability have not been established.

Qev adaptation weights use Apache-2.0. The Qwen base, tokenizer and source datasets retain their respective terms. [License and attribution](https://github.com/QiqianFu/Qev/blob/main/THIRD_PARTY_NOTICES.md).
