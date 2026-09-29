---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B-Base
base_model_relation: adapter
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

Qev-2B is the compact member of the Qev decision-model family. It learns option probabilities and representation responses from Qev-9B, and supports Choice, Noul and Score with the same request format.

**Release status:** the selected checkpoint has been exported locally as `checkpoints/qev-2b`. Hugging Face publication is pending; this page does not advertise an available 2B Hub download.

| Property | Qev-2B |
|---|---|
| Base | Qwen3.5-2B-Base |
| Adapter | LoRA rank 64, alpha 128 |
| Decision head | 256 dimensions, two Transformer layers |
| Candidate interaction | Full sibling interaction in the final full-attention layer |
| Candidate preview | None |
| Training | Teacher-probability cross entropy, followed by replay and representation-response distillation |
| Final continuation | 800 steps; seed 17; response weight 0.1 |
| Computation | BF16 backbone, FP32 decision head |
| Inference package | Approximately 284 MiB; base weights downloaded separately |

```python
from qev import Qev

model = Qev.from_pretrained("checkpoints/qev-2b", device="cuda")
```

To fine-tune on your own data, use `configs/qev-2b-finetune.json` with `python -m qev.train --init-checkpoint checkpoints/qev-2b`. To train with a teacher, follow the [distillation guide](distillation.md) or [中文指南](distillation.zh-CN.md).

| Benchmark | Qwen3.5-2B-Base | Qev-2B |
|---|---:|---:|
| Decision development · clean | 65.43 | 85.36 |
| Transfer development · clean | 65.09 | 77.29 |
| MMLU-Pro · 1,000 | 31.20 | 38.70 |
| SemIf · 144 handwritten | 63.89 | 82.64 |
| scienthoon · 873 | 53.84 | 71.94 |
| WANLI · 256 | 50.39 | 67.58 |
| JevBench public · 231 | 63.20 | 74.46 |

Accuracy (%). Public JevBench accuracy is 172/231. These are the selected checkpoint's recorded results. The native base uses its language-model head and zero-shot prompts. The comparison therefore includes architecture, training and readout differences; it does not isolate the contribution of response distillation. [Full results](evaluation.md).

Qev adaptation weights use Apache-2.0. Source datasets and the Qwen base retain their own terms. [License and attribution](../THIRD_PARTY_NOTICES.md).
