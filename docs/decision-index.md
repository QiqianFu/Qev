# Additional benchmarks: Decision Index 0.2.1

The additional evaluations use the official [Decision Index toolkit](https://github.com/apolinario/decision-index/tree/87d4650b42b377c0291a89c1f1a879f9b31082bf), pinned at `87d4650b42b377c0291a89c1f1a879f9b31082bf`. Results were recorded on 2026-10-03 for the released Qev-2B, Qev-4B and Qev-9B checkpoints, their native Qwen bases, Kev-4B/9B and both JevAny-4B variants.

The README and figures display GSM8K, ChessBench and BPoMP using **official raw scores × 100**. The [machine-readable results](../results/decision-index.json) also preserve chance-adjusted skill, coverage and individual tracks. These metrics remain separate from the original suites' integer correct-answer counts in [benchmarks.json](../results/benchmarks.json).

| Benchmark | Inputs | Raw score used in the figures |
|---|---:|---|
| GSM8K | 1,319 source questions, each presented with 4 and 10 options: 2,638 requests | Mean accuracy of the two tracks |
| ChessBench | 5,000 chess positions | Best-move accuracy, accepting tied optimal moves |
| BPoMP | 5,000 poetry questions | Accuracy computed separately for each poem variant, then averaged |

The [official scorer](https://github.com/apolinario/decision-index/blob/87d4650b42b377c0291a89c1f1a879f9b31082bf/decision_index/scoring/index.py) defines these aggregations. In particular, BPoMP's raw score differs from simply dividing the total correct answers by 5,000. Skill is chance-adjusted separately; it is not the value plotted in the cover chart.

## Inputs and scoring

The inputs were rebuilt with the toolkit from its pinned public sources: `openai/gsm8k`, DeepMind's `searchless_chess` and the BPoMP poetry data. The official rebuild includes both GSM8K choice tracks. Requests preserve the complete state, questions, candidate IDs, candidate order and expected answers; all local models answered all requests in these evaluations.

Predictions are converted to the toolkit's result format and scored using its `score_panel` and `index02.benchmark_value` functions. The benchmarks are scored individually; this release does not report a full Decision Index composite score. The original four-benchmark publication, including ESCI, was verified by rerunning the scorer on all ten local models' saved outputs: all 40 raw/skill/coverage results matched. That complete record remains in the machine-readable results; ESCI is omitted from the displayed comparisons.

The suite's rows are not redistributed here. Use the [upstream rebuild instructions](https://github.com/apolinario/decision-index/tree/87d4650b42b377c0291a89c1f1a879f9b31082bf) to obtain the inputs. This repository publishes aggregate results and provenance.

## Models and reference sources

- Qev uses the released checkpoints: **2B v0.1.0**, **4B v0.1.0**, **9B v0.2.0**, with BF16 backbone computation, an FP32 decision head, temperature 1 and full causal reference execution.
- Native Qwen bases use their original language-model heads, zero-shot candidate-code prompts and BF16 computation. The 4B baseline is `Qwen/Qwen3.5-4B-Base@710fd005d44d55ee27b7ad5147e318e546efdbfe`.
- Kev uses the author's inference implementation in FP32 with unmerged adapters and temperature 1. The 4B run uses `jaredpalmer/kev-4b@139fdd9` and Kev code `5920c5f`; the 9B run uses `jaredpalmer/kev-9b@2629c06a` and code `557598f`.
- JevAny's [Pointer](https://huggingface.co/SimpleJev/JevAny-Qwen3.5-4B-LoRA) and [Direct-Token](https://huggingface.co/SimpleJev/JevAny-Qwen3.5-4B-Direct-Token-LoRA) models use the author's `33cb677` inference code in FP32. Their base is the post-trained Qwen3.5-4B. The cover uses **Pointer** for every benchmark; both variants are included in the detailed results.
- Jev 1.13.0 is a published reference from the toolkit's [frozen 0.2.1 leaderboard](https://github.com/apolinario/decision-index/blob/87d4650b42b377c0291a89c1f1a879f9b31082bf/tests/fixtures/board-0.2.1.json). Its reference scores were not produced by a new API run for Qev.

## Interpretation

GSM8K here is a multiple-choice adaptation. Its [deterministic distractor builder](https://github.com/apolinario/decision-index/blob/87d4650b42b377c0291a89c1f1a879f9b31082bf/decision_index/suite/build/adapters_selection.py) creates alternatives using arithmetic transformations of the correct answer. An option-only audit found a strong shortcut in these relationships. These scores should therefore not be presented as standard free-response GSM8K reasoning performance.

The comparisons use one recorded run per model. Training data, initial bases and inference precision differ. Original benchmark results remain attached to their original runs; later numerical spot checks do not replace them selectively.

[4B comparison](evaluation.md#4b-model-comparison) · [Full results](evaluation.md) · [Validation](validation.md)
