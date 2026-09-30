# Changelog

## Unreleased

- Updated Qev-9B to v0.2.0: HelpSteer3 Principle and 600 synthetic boundary questions in the main set, MMLU-Pro 57.40%, public JevBench 192/231. The original v0.1.0 weights and results remain available.
- Expanded Qev-train to v1.1.0 with 2,442 examples, adding 600 controlled boundary tasks and their CC BY 4.0 attribution. The original v1.0.0 dataset is preserved.
- Pinned the released 2B recipe to its original Qev-9B v0.1.0 teacher and protected boundary-task candidate sets during further training.
- Published the initial Qev-train dataset: 1,842 synthetic training examples, bilingual synthesis documentation, compatible JSONL/Parquet formats and component-specific licenses.

- Added Qev-2B teacher-probability training, programmatic context edits, representation-response continuation, and matching training documentation.
- Added the selected 2B student and native Qwen3.5-2B baseline to the model selector, benchmark table and plots.
- Published Qev-2B weights on Hugging Face with the same loading interface as Qev-9B.
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

Qev is the public name of the selected research models; packaging preserves their original trained checkpoints and benchmark results. The weights are available at [AustinFu/Qev-2B](https://huggingface.co/AustinFu/Qev-2B) and [AustinFu/Qev-9B](https://huggingface.co/AustinFu/Qev-9B). The complete training corpus is not bundled.
