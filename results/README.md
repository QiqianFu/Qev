# Benchmark results

- [Benchmark metrics](benchmarks.json): correct answers, question counts and model descriptions for the README tables.
- [JevBench predictions](qev-9b/jevbench-predictions.jsonl): Qev's probabilities and predictions for all 231 public questions, without the question text.
- [Qev-2B JevBench predictions](qev-2b/jevbench-predictions.jsonl) and [native 2B predictions](qwen3.5-2b-base/jevbench-predictions.jsonl): the selected 2B student and its original Qwen base.
- [2B full-suite counts](qev-2b/evaluation.json): original full development/SemIf scopes, separate from the matched README subsets.
- [Evaluation metadata](qev-9b/provenance.json): the model and settings used for those predictions.

These are the published model's original results. Code maintenance does not create new benchmark measurements. See [evaluation](../docs/evaluation.md) for methods and [validation](../docs/validation.md) for software checks.

The [release history](history/README.md) preserves the original packaging records. [Licensing and attribution](../THIRD_PARTY_NOTICES.md) describe the terms for outputs and upstream evaluation data.
