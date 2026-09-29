# Frozen benchmark evidence

- [benchmarks.json](benchmarks.json): integer numerators, denominators, fixed model identities and source distinctions for the README tables.
- [Qev-9B JevBench predictions](qev-9b/jevbench-predictions.jsonl): original probabilities and predictions for all 231 public questions, without the benchmark's input text.
- [Prediction provenance](qev-9b/provenance.json): checkpoint, dataset hash, inference limits and prediction-file SHA256.

[validation.json](validation.json) records the source-package checks separately from model benchmark measurements.

[huggingface-release.json](huggingface-release.json) records the public `AustinFu/Qev-9B` v0.1.0 release, its exact commit, and anonymous download verification with file hashes. It is a publication check, not a new benchmark run.

The model was named BranchKev in the source run and is now released as Qev. These files describe the recorded checkpoint and are not results from a new run of the renamed package. See [evaluation.md](../docs/evaluation.md) for protocol differences, ablations and limitations.
