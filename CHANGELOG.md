# Changelog

## Unreleased

- Simplified model downloads, checkpoint export and dataset preparation. Model versions are optional; exports no longer produce checksum inventories.
- Removed the server-specific checkpoint archive handshake. Resume still checks training data automatically.
- Replaced internal experiment names with plain descriptions of the model and training recipe.
- Added explicit licensing for Qev weights, documentation and visuals, retained upstream license texts, and included notices in Python distributions and prepared evaluation data.

## 0.1.0

- Extracted the research implementation as the standalone `qev` package, retaining candidate-tree encoding, set readout, LoRA/FSDP training, and reference/cache/tree execution.
- Kept backward-compatible loading of BranchKev records and checkpoints.
- Added a Python API, explicit prediction precision, portable LoRA checkpoint export, Hub checkpoint loading, local base overrides, and initialization on new data with fresh optimizer state.
- Added user JSONL preparation with train/dev split and overlap checks.
- Added English/Chinese README, architecture diagrams and an adapted interactive decision-head explanation.
- Included fixed-version benchmark summaries and the selected model's original 231 JevBench prediction rows.
- Added an offline tiny-model smoke workflow and release validation.

Qev is the public name of the selected research model; this packaging work does not create a new trained checkpoint or new benchmark result. The weights are available at [AustinFu/Qev-9B](https://huggingface.co/AustinFu/Qev-9B). The complete training corpus is not bundled.
