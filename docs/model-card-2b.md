---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B-Base
base_model_relation: adapter
datasets:
  - AustinFu/Qev-train
language:
  - en
  - zh
tags:
  - qev
  - lora
  - knowledge-distillation
  - decision-model
---

# Qev-2B

Qev-2B is the compact member of the Qev decision-model family. It learns option probabilities and representation responses from Qev-9B v0.1.0, and supports Choice, Noul and Score with the same request format.

[Model weights](https://huggingface.co/AustinFu/Qev-2B) · [Source and installation](https://github.com/QiqianFu/Qev#installation) · [中文说明](https://github.com/QiqianFu/Qev/blob/main/README.zh-CN.md)

| Property | Qev-2B |
|---|---|
| Base | Qwen3.5-2B-Base |
| Adapter | LoRA rank 64, alpha 128 |
| Decision head | 256 dimensions, two Transformer layers |
| Candidate interaction | Full sibling interaction in the final full-attention layer |
| Training | Teacher-probability cross entropy, followed by replay and representation-response distillation |
| Final continuation | 800 steps; seed 17; response weight 0.1 |
| Computation | BF16 backbone, FP32 decision head |
| Inference package | Approximately 284 MiB; base weights downloaded separately |

```python
from qev import Qev

model = Qev.from_pretrained("AustinFu/Qev-2B", device="cuda")
```

Install Qev from the source repository using Python 3.12 and a hardware-compatible PyTorch 2.8.0 build. The loader downloads this adaptation package and the compatible Qwen base separately. The teacher is not needed for inference. Use `revision="v0.1.0"` to select the first published release.

To fine-tune on your own data, use `configs/qev-2b-finetune.json` with `python -m qev.train --init-checkpoint AustinFu/Qev-2B`. To train with a teacher, follow the [distillation guide](distillation.md) or [中文指南](distillation.zh-CN.md).

| Benchmark | Qwen3.5-2B-Base | Qev-2B |
|---|---:|---:|
| JevBench public · 231 | 63.20 | 74.46 |
| Decision development · clean | 65.43 | 85.36 |
| Transfer development · clean | 65.09 | 77.29 |
| MMLU-Pro · 1,000 | 31.20 | 38.70 |
| SemIf · 144 handwritten | 63.89 | 82.64 |
| scienthoon · 873 | 53.84 | 71.94 |
| WANLI · 256 | 50.39 | 67.58 |
| GSM8K · multiple choice | 30.40 | 37.76 |
| ChessBench | 8.84 | 10.98 |
| BPoMP · variant mean | 50.50 | 73.81 |

Scores (%). The three new rows use [Decision Index raw scores](evaluation.md#decision-index-021); the original rows use accuracy. Public JevBench accuracy is 172/231. These are the selected checkpoint's recorded results. The native base uses its language-model head and zero-shot prompts. The comparison therefore includes architecture, training and readout differences; it does not isolate the contribution of response distillation. [Full results](evaluation.md).

The package includes the LoRA adapter, decision head, interaction gate, tokenizer, model configuration, both training configurations, recorded evaluation results and license files. The three trained tensor files preserve the selected checkpoint exactly; optimizer state and base weights are excluded.

[Qev-train v1.0.0](https://huggingface.co/datasets/AustinFu/Qev-train/tree/v1.0.0) publishes the 1,842 synthetic source examples used for this release with original hard labels and a description of their generation. The full mixed training corpus, teacher probability caches and response-distillation data are not distributed.

Qev supports research and development of routing, rule judgments and rubric ratings over explicit options. Evaluate the model on your application's inputs; probability calibration and production reliability have not been established.

Qev adaptation weights use Apache-2.0. Source datasets and the Qwen base retain their own terms. [License and attribution](../THIRD_PARTY_NOTICES.md).
