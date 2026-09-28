# Changelog

## 0.1.0 — source release preparation

- Extracted the research implementation as the standalone `qev` package, retaining candidate-tree encoding, set readout, LoRA/FSDP training, and reference/cache/tree execution.
- Kept backward-compatible loading of BranchKev records and checkpoints.
- Added a Python API, explicit prediction precision, portable LoRA checkpoint export, pinned Hub checkpoint resolution, local base overrides, and initialization on new data with fresh optimizer state.
- Added user JSONL preparation with train/dev role, hash and overlap checks.
- Added English/Chinese README, architecture diagrams and an adapted interactive decision-head explanation.
- Included fixed-version benchmark summaries and the selected model's original 231 JevBench prediction rows.
- Added an offline tiny-model smoke workflow and release validation.

Qev is the public name of the selected research model; this packaging work does not create a new trained checkpoint or new benchmark result. Public model hosting and full training-corpus distribution remain separate release steps.
