# Configurations

`qev-9b.json` is the selected rank-64, seed-17, two-epoch research recipe. It expects four GPU ranks and main/late training partitions. It includes the model architecture, optimizer settings and schedule used for the published results.

`qev-9b-finetune.json` keeps the same model structure, uses batch 1 / accumulation 32, removes the late-partition requirement and sets a lower backbone LR for user-data initialization. It is a starting recipe, not a measured domain-specific improvement.

`qev-9b-independent.json` and `qev-9b-readout-cross.json` preserve the corresponding research ablation settings. `qev-0.8b.json` and `qev-9b-fullft.json` are additional research recipes; they do not identify the selected Qev-9B checkpoint.

[Training guide](../docs/training.md) · [Architecture](../docs/architecture.md)
