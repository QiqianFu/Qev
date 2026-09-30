# Benchmark results

- [Benchmark metrics](benchmarks.json): correct answers, question counts and model descriptions for the README tables.
- [JevBench predictions](qev-9b/jevbench-predictions.jsonl): Qev's probabilities and predictions for all 231 public questions, without the question text.
- [Qev-2B JevBench predictions](qev-2b/jevbench-predictions.jsonl) and [native 2B predictions](qwen3.5-2b-base/jevbench-predictions.jsonl): the selected 2B student and its original Qwen base.
- [2B full-suite counts](qev-2b/evaluation.json): original full development/SemIf scopes, separate from the matched README subsets.
- [Evaluation metadata](qev-9b/provenance.json): the model and settings used for those predictions.
- [Qev-9B v0.2.0 full-suite counts](qev-9b/evaluation.json): complete development/SemIf counts and the matched README subsets.

The current 9B results describe Qev-9B v0.2.0; 2B results remain Qev-2B v0.1.0. The [first 9B release](history/qev-9b-v0.1.0/README.md) is preserved separately. These are recorded training-run measurements, not new scores inferred from code maintenance. See [evaluation](../docs/evaluation.md) for methods and [validation](../docs/validation.md) for software checks.

The [release history](history/README.md) preserves the original packaging records. [Licensing and attribution](../THIRD_PARTY_NOTICES.md) describe the terms for outputs and upstream evaluation data.
