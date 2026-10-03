---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B-Base
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

# Qev-4B

Qev-4B turns context, a question and candidate answers into a decision and a probability for every option. It supports **Choice**, **Noul** (yes/no) and **Score** (ordered ratings) through the same API as Qev-2B and Qev-9B.

[Source and installation](https://github.com/QiqianFu/Qev#installation) · [Training method](https://github.com/QiqianFu/Qev/blob/main/docs/training-4b.md) · [中文训练说明](https://github.com/QiqianFu/Qev/blob/main/docs/training-4b.zh-CN.md)

This release starts **directly from Qwen3.5-4B-Base**, with fresh LoRA and decision-head parameters. A single two-epoch stage learns a 9B teacher's option probabilities. It does not inherit an earlier distilled 4B checkpoint or include representation-response continuation.

| Property | Qev-4B v0.1.0 |
|---|---|
| Base | Qwen3.5-4B-Base, revision `710fd005d44d55ee27b7ad5147e318e546efdbfe` |
| Adaptation | LoRA rank 64, alpha 128 |
| Decision head | 256 dimensions, four attention heads, two Transformer layers |
| Candidate interaction | Full sibling interaction in the final full-attention layer |
| Training | Teacher-probability cross entropy; 44,576 single-question inputs |
| Schedule | Two epochs, 2,786 steps, four GPUs, global batch 32, seed 17 |
| Temperatures | Teacher 1.563437713227029; student 1 |
| Input limits | 4,096-token state and complete path; question 512, candidate 256 |
| Computation | BF16 backbone, FP32 decision head |
| Download | Approximately 524 MiB; the Qwen base downloads separately |

## Use the model

Install the [Qev source package](https://github.com/QiqianFu/Qev#installation) with Python 3.12 and a hardware-compatible PyTorch 2.8.0 build:

```python
from qev import Qev

model = Qev.from_pretrained("AustinFu/Qev-4B", revision="v0.1.0", device="cuda")
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

The loader restores the LoRA adapter, decision head, interaction gate and tokenizer, and downloads the pinned Qwen base. No teacher is needed at inference time. The package contains all trained adaptation tensors, training and fine-tuning configurations, evaluation results and license files; optimizer state and base weights are excluded.

## Recorded evaluation

| Benchmark | Correct / total | Accuracy (%) |
|---|---:|---:|
| Decision development · clean | 1099 / 1264 | 86.95 |
| Transfer development · clean | 538 / 656 | 82.01 |
| MMLU-Pro | 503 / 1000 | 50.30 |
| SemIf · handwritten | 130 / 144 | 90.28 |
| scienthoon | 670 / 873 | 76.75 |
| WANLI | 188 / 256 | 73.44 |
| JevBench public | 190 / 231 | 82.25 |

Results use BF16 backbone computation, an FP32 head, temperature 1 and full causal reference execution. There were zero rejected questions in all seven original evaluations. The development and SemIf rows use the same subsets as the [Qev leaderboard](https://github.com/QiqianFu/Qev#evaluation); full-suite counts are included in `evaluation.json`.

![Qev benchmark matrix](https://raw.githubusercontent.com/QiqianFu/Qev/main/assets/evaluation-matrix.svg)

These are measurements from one checkpoint and seed. Training data and objectives differ across the 2B, 4B and 9B releases; their scores do not isolate model-size effects. The native Qwen3.5-4B-Base baseline is now available: 43.50% MMLU-Pro and 155/231 JevBench. See the [4B comparison](https://github.com/QiqianFu/Qev/blob/main/docs/evaluation.md#4b-model-comparison) for all tasks. The gameplay recordings in the source repository use 9B models.

## Additional evaluation — 2026-10-03

| Benchmark | Qwen3.5-4B-Base | Qev-4B |
|---|---:|---:|
| GSM8K · multiple choice | 37.00 | 54.59 |
| ChessBench · 5,000 | 11.78 | 12.42 |
| Amazon ESCI · macro-F1 | 29.18 | 42.49 |
| BPoMP · variant mean | 68.95 | 78.19 |

These are Decision Index 0.2.1 raw scores (%); ESCI uses macro-F1 and BPoMP averages over variants. [Scoring definitions and sources](https://github.com/QiqianFu/Qev/blob/main/docs/decision-index.md).

## Training and availability

The input pool combines general decisions, science and reasoning, Principle judgments, controlled boundaries, web actions and additional rule/reasoning tasks. Each record contains one question with its hard label removed. Training uses a single pool without late-stage repetition or online option changes.

The recorded teacher is a separate 9B research checkpoint trained with additional web and rule tasks, at step 2,658; it is different from the public Qev-9B v0.2.0. The teacher, full 44,576-input pool and cached teacher outputs are not distributed. [Qev-train](https://huggingface.co/datasets/AustinFu/Qev-train) remains a separate release of 2,442 synthetic examples with hard labels and synthesis documentation. The source repository provides the 4B method and a runnable example using a public teacher and user data.

Use `configs/qev-4b-finetune.json` with `python -m qev.train --init-checkpoint AustinFu/Qev-4B@v0.1.0` for supervised adaptation. Validate performance and probabilities on your own task; calibration and production reliability have not been established.

The Qev adaptation weights use Apache-2.0. The Qwen base, tokenizer and source datasets retain their respective terms. [License and attribution](https://github.com/QiqianFu/Qev/blob/main/THIRD_PARTY_NOTICES.md).
