# Configurations

`qev-9b.json` is the Qev-9B v0.3.0 rank-64, seed-17, two-epoch recipe: 38,198 main records (including 2,230 Principle judgments and 1,422 web-action records), 2,883 late records, and 2,658 optimizer steps. State and complete-path limits are 4,096 tokens. The late set includes 600 boundary questions and 500 additional rule/reasoning examples. It expects four GPU ranks and main/late training partitions. It includes the model architecture, optimizer settings and schedule used for the published results.

`qev-9b-finetune.json` keeps the same model structure, uses batch 1 / accumulation 32, removes the late-partition requirement and sets a lower backbone LR for user-data initialization. It is a starting recipe, not a measured domain-specific improvement.

`qev-9b-independent.json` and `qev-9b-readout-cross.json` preserve the corresponding research ablation settings. `qev-0.8b.json` and `qev-9b-fullft.json` are additional research recipes; they do not identify the selected Qev-9B checkpoint.

[Training guide](../docs/training.md) · [Architecture](../docs/architecture.md)

## 0.8B model

- `qev-0.8b-distill.json`: the released seed-17 model’s two-epoch teacher-probability recipe, starting directly from Qwen3.5-0.8B-Base. Two GPUs × batch 8 × accumulation 2 gives global batch 32; 44,576 inputs, 2,786 steps.
- `qev-0.8b-finetune.json`: supervised adaptation from Qev-0.8B on your own data, without a teacher.
- `qev-0.8b.json` remains the older rank-16 research recipe and is not the released checkpoint’s configuration.

[0.8B method](../docs/training-0.8b.md) · [中文](../docs/training-0.8b.zh-CN.md).

## 4B model

- `qev-4b.json`: the released model's two-epoch, four-GPU teacher-probability recipe, initialized directly from Qwen3.5-4B-Base. It uses 44,576 single-question inputs, global batch 32, 2,786 steps, teacher temperature 1.563437713227029 and no late split.
- `qev-4b-finetune.json`: supervised adaptation from Qev-4B on your own labelled data, batch 1 / accumulation 32, without a teacher cache.

[4B training method](../docs/training-4b.md) · [中文](../docs/training-4b.zh-CN.md).

## 2B models

- `qev-2b.json`: the 2B architecture and two-epoch supervised-training recipe.
- `qev-2b-finetune.json`: fine-tune a Qev-2B checkpoint on your own data.
- `qev-2b-distill.json`: train on Qev-9B v0.1.0 probability targets with the same data and augmentation schedule.
- `qev-2b-response.json`: 800 continuation steps with teacher-probability cross entropy, original-question replay and representation-response matching.

[Distillation workflow](../docs/distillation.md).
