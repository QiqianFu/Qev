# Configurations

`qev-9b.json` is the selected rank-64, seed-17, two-epoch research recipe. It expects four GPU ranks and main/late training partitions. It includes the model architecture, optimizer settings and schedule used for the published results.

`qev-9b-finetune.json` keeps the same model structure, uses batch 1 / accumulation 32, removes the late-partition requirement and sets a lower backbone LR for user-data initialization. It is a starting recipe, not a measured domain-specific improvement.

`qev-9b-independent.json` and `qev-9b-readout-cross.json` preserve the corresponding research ablation settings. `qev-0.8b.json` and `qev-9b-fullft.json` are additional research recipes; they do not identify the selected Qev-9B checkpoint.

[Training guide](../docs/training.md) · [Architecture](../docs/architecture.md)

## 2B models

- `qev-2b.json`: the 2B architecture and two-epoch supervised-training recipe.
- `qev-2b-finetune.json`: fine-tune a Qev-2B checkpoint on your own data.
- `qev-2b-distill.json`: train on Qev-9B probability targets with the same data and augmentation schedule.
- `qev-2b-response.json`: 800 continuation steps with teacher-probability cross entropy, original-question replay and representation-response matching.

[Distillation workflow](../docs/distillation.md).
