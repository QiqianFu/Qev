# Evaluation and provenance

These results describe Qev-9B v0.3.0: Qwen3.5-9B-Base with rank-64 LoRA, candidate interaction, a two-layer decision head and 4K context. The main set includes 2,230 Principle judgments and 1,422 web-action records. The 2,883-record late set adds alignment, document rules, 600 boundary questions and 500 rule/reasoning questions, mixed into the second half of training and repeated three times.

[Machine-readable accuracy results](../results/benchmarks.json) contain integer numerators and denominators. [Decision Index scores](../results/decision-index.json) preserve official raw scores, chance-adjusted skill and track details separately. [Qev's 231 JevBench predictions](../results/qev-9b/jevbench-predictions.jsonl) are preserved from the original evaluation.

Qev-2B uses probability distillation followed by context edits, original-question replay and representation-response matching. Its MMLU-Pro accuracy is 38.70%, and its public JevBench accuracy is 172/231 (74.46%). Both 2B models use BF16 backbone computation. See the [training method](distillation.md).

Qev-4B v0.1.0 starts directly from Qwen3.5-4B-Base and learns teacher probabilities for 2,786 steps. It scores 503/1,000 on MMLU-Pro and 190/231 on public JevBench. The [4B recipe](training-4b.md) uses the teacher now published as Qev-9B v0.3.0 and a different input mixture from the other releases. The native Qwen3.5-4B-Base baseline scores 43.50% on MMLU-Pro and 155/231 on JevBench, using its original language-model head.

## Models and sources

| Model | Identity | Result source |
|---|---|---|
| Qev-9B | Qwen3.5-9B-Base + rank-64 LoRA + 256×2 head, seed 17, step 2658 | Local BF16 full causal reference execution |
| Kev-9B | [jaredpalmer/kev-9b](https://huggingface.co/jaredpalmer/kev-9b) | JevBench rerun locally with Kev's own `kev.benchmark`, FP32 and T=1; other suites from pinned author reports |
| Qwen3.5-9B-Base | [Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base) | Frozen native LM head, zero-shot prompt, probabilities normalized over valid answer codes |
| Qev-4B | Qwen3.5-4B-Base + rank-64 LoRA + 256×2 head, seed 17, step 2786 | Recorded BF16 full causal reference execution |
| Qwen3.5-4B-Base | Qwen3.5-4B-Base at `710fd005` | BF16 native LM head, zero-shot candidate codes |
| Kev-4B | [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b) at `139fdd9`; Kev code `5920c5f` | Author inference implementation, FP32, T=1, unmerged adapter |
| JevAny-4B Pointer / Direct-Token | [Pointer](https://huggingface.co/SimpleJev/JevAny-Qwen3.5-4B-LoRA) / [Direct-Token](https://huggingface.co/SimpleJev/JevAny-Qwen3.5-4B-Direct-Token-LoRA), code `33cb677` | Post-trained Qwen3.5-4B base, author exact FP32 inference path |
| Qev-2B | Qwen3.5-2B-Base + rank-64 LoRA + 256×2 head; no candidate preview | Recorded reference execution after response distillation |
| Qwen3.5-2B-Base | Original language-model head, zero-shot candidate codes | Recorded native-base evaluation |
| Jev | Hosted service; JevBench used Jev 1.13.0 | JevBench API run on 2026-09-26; older suites from Jev reports preserved by the Kev authors; new tasks from the pinned Decision Index 0.2.1 board |

Kev's published results are available in its [evaluation reports](https://github.com/jaredpalmer/kev/tree/557598fced1dada75dfbf36ed144dce309ac6ceb/runs). JevBench was evaluated locally for Qev, Kev, and the Qwen base, and through the Jev API for the hosted reference.

## Benchmark matrix

![Eleven models across ten benchmarks](../assets/evaluation-matrix.svg)

## Full accuracy table

Percent accuracy. Each released Qev model is a single seed. Bold scores mark the higher result between Qev-9B and Kev-9B; ties are unbolded. Jev is a hosted reference.

**Precision comparison: Kev and JevAny use FP32; Qev-9B, Qev-4B and Qev-2B use BF16 backbone computation.** Qev retains FP32 for its decision head and key reductions; its exported LoRA tensors are stored in FP32. These are not identical precision settings.

| Scope | Jev (reference) | Qev-9B | Kev-9B | Qwen3.5-9B-Base | Qev-4B | Qwen3.5-4B-Base | Qev-2B | Qwen3.5-2B-Base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JevBench public · 231 | 85.71 | **80.95** | 75.76 | 75.76 | 82.25 | 67.10 | 74.46 | 63.20 |
| decision_dev all · 1468 | 83.17 | **88.28** | 87.81 | 75.75 | 87.60 | 71.53 | 86.24 | 63.15 |
| decision_dev clean · 1264 | 84.49 | **87.58** | 87.18 | 77.69 | 86.95 | 73.73 | 85.36 | 65.43 |
| transfer_dev all · 764 | 84.69 | **82.33** | 81.15 | 73.43 | 80.89 | 70.29 | 76.44 | 64.53 |
| transfer_dev clean · 656 | 85.67 | **83.69** | 82.16 | 74.39 | 82.01 | 71.49 | 77.29 | 65.09 |
| MMLU-Pro · 1,000 | 83.50 | **56.50** | 51.10 | 50.40 | 50.30 | 43.50 | 38.70 | 31.20 |
| SemIf · 144 handwritten | 96.53 | **93.75** | 90.97 | 90.28 | 90.28 | 77.08 | 82.64 | 63.89 |
| scienthoon · 873 | 75.26 | 74.80 | **75.49** | 68.84 | 76.75 | 74.34 | 71.94 | 53.84 |
| WANLI · 256 | 75.78 | **75.00** | 70.31 | 67.97 | 73.44 | 60.55 | 67.58 | 50.39 |

The all-question dev rows include clean examples and candidate-permutation / None-present / None-absent variants. Clean rows match the clean reporting convention in Kev's README. SemIf uses 144 handwritten questions. Qev's broader 252-question research run also included 108 perturbations; its 96.43% overall score is not the 144-question comparison above.

## 4B model comparison

All rows use the same reporting scopes as the README. Bold compares Qev-4B with Kev-4B; Jev and JevAny are references. The cover uses the standard JevAny Pointer release consistently across all tasks; the Direct-Token variant is also shown here.

| Benchmark | Jev (reference) | Qev-4B | Kev-4B | JevAny-4B Pointer | JevAny-4B Direct-Token | Qwen3.5-4B-Base |
|---|---:|---:|---:|---:|---:|---:|
| JevBench public · 231 | 85.71 | **82.25** | 75.76 | 78.35 | 79.65 | 67.10 |
| Decision development · clean | 84.49 | 86.95 | **87.26** | 86.63 | 86.23 | 73.73 |
| Transfer development · clean | 85.67 | **82.01** | 81.71 | 84.60 | 85.52 | 71.49 |
| MMLU-Pro · 1,000 | 83.50 | 50.30 | **52.40** | 52.30 | 51.10 | 43.50 |
| SemIf · 144 handwritten | 96.53 | **90.28** | 88.89 | 90.28 | 90.28 | 77.08 |
| scienthoon · 873 | 75.26 | **76.75** | 72.28 | 69.30 | 68.84 | 74.34 |
| WANLI · 256 | 75.78 | **73.44** | 68.75 | 71.09 | 71.09 | 60.55 |
| GSM8K · multiple choice | 79.87 | 54.59 | **58.49** | 47.61 | 45.03 | 37.00 |
| ChessBench · 5,000 | 17.22 | **12.42** | 8.80 | 11.32 | 11.72 | 11.78 |
| BPoMP · variant mean | 90.92 | **78.19** | 65.28 | 73.87 | 81.29 | 68.95 |

Qev-4B and Kev-4B start from Qwen3.5-4B-Base; JevAny uses the post-trained Qwen3.5-4B base. Qev uses BF16 backbone computation, while the Kev and JevAny runs use FP32. These are model comparisons, with different training data and objectives, rather than isolated architecture comparisons.

## Decision Index 0.2.1

The three displayed additional benchmarks use the toolkit's official **raw** score, multiplied by 100. BPoMP averages accuracy across poem variants. GSM8K averages the four-choice and ten-choice tracks; ChessBench accepts all tied best moves.

| Benchmark | Jev (reference) | Qev-9B | Kev-9B | Qwen3.5-9B-Base | Qev-4B | Qwen3.5-4B-Base | Qev-2B | Qwen3.5-2B-Base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM8K · multiple choice | 79.87 | **61.75** | 46.36 | 55.53 | 54.59 | 37.00 | 37.76 | 30.40 |
| ChessBench · 5,000 | 17.22 | 10.34 | **11.76** | 13.22 | 12.42 | 11.78 | 10.98 | 8.84 |
| BPoMP · variant mean | 90.92 | **81.03** | 66.93 | 59.39 | 78.19 | 68.95 | 73.81 | 50.50 |

All local models answered every item in these three evaluations. Jev's new scores come from the toolkit's pinned public leaderboard, not a new API run. The [method note](decision-index.md) records exact sources, scoring rules and the GSM8K option-construction limitation. Original benchmark scores remain attached to their original evaluations.

## Public JevBench breakdown

| Subset | Questions | Qev-9B correct | Kev-9B correct | Qwen base correct | Jev correct (reference) | Qev-4B correct |
|---|---:|---:|---:|---:|---:|---:|
| original | 72 | 67 | 65 | 59 | 71 | 69 |
| easy | 48 | 48 | 48 | 48 | 48 | 48 |
| hard | 111 | 72 | 62 | 68 | 79 | 73 |
| all | 231 | 187 | 175 | 175 | 198 | 190 |

This is local argmax accuracy on the public v1.4.2 tasks. It is not the official composite score, which includes other dimensions and nonpublic tasks. The public benchmark was observed during research iteration, so these results are not an untouched final blind test.

Precision, prompts, and implementations differ across models; these are model-level observations, not a controlled architecture comparison.

## Rerun Qev's public evaluation

Download the released checkpoint, then prepare and evaluate the public tasks:

```bash
hf download AustinFu/Qev-9B --revision v0.3.0 --local-dir checkpoints/qev-9b

python scripts/prepare_jevbench.py \
  --root data/jevbench --source-dir data/jevbench/source \
  --tokenizer checkpoints/qev-9b/tokenizer --config configs/qev-9b.json

python -m qev.evaluate --checkpoint checkpoints/qev-9b \
  --data data/jevbench/model_data/jevbench-public-v1.4.2 --split public \
  --out runs/jevbench-reference --device cuda \
  --reference --weights-dtype checkpoint --max-state 4096 --max-path 4096
```

The preparer downloads JevBench v1.4.2, retains the original answers and probability targets, and keeps all 231 public tasks. Every exported view has an `external_evaluation` role. Re-running the preparation command on its completed directory verifies it instead of replacing it. Use `--compare-data` to audit canonical input overlap with supplied training datasets.

To evaluate a prepared user-data dev split:

```bash
python -m qev.evaluate --checkpoint runs/support/step-000100 \
  --data data/support --split dev --out runs/support-eval \
  --reference --weights-dtype checkpoint
```

Use an actual saved step. Reports preserve rejected items and both answered-only accuracy and accuracy counting rejections wrong. Probability checks, Brier, NLL and ECE are also computed; their availability does not mean the released checkpoint has been calibrated.

## Architecture ablations

These historical ablations use the Qev-9B v0.1.0 data recipe, rank 64, seed 17 and final step 2327. Their results remain separate from the current v0.3.0 leaderboard. The released 9B models retain the full interaction architecture.

| Variant | Backbone sibling cross | Set attention layers | JevBench correct | Mean P(gold label) |
|---|---|---:|---:|---:|
| Qev-9B v0.1.0 reference | All sibling tokens | 2 | 188/231 | 0.782 |
| No backbone interaction | Off | 2 | 185/231 | 0.775 |
| No set attention | All sibling tokens | 0 | 189/231 | 0.791 |
| Neither interaction | Off | 0 | 187/231 | 0.780 |
| Readout-only | Sibling readout tokens | 2 | 187/231 | 0.784 |

The single-seed comparisons did not establish a measurable gain from either interaction component. Projection and the scalar scorer remain even when set attention is disabled. Related rank-16 seed sweeps had a JevBench standard deviation of about 4.7 questions; that is context, not a confidence interval or significance test for this final recipe.

scienthoon questions cluster by ticket template, so treating every question as an independent observation understates uncertainty. Jev's large knowledge-benchmark advantage does not identify its undisclosed model architecture or training data. No speed superiority or probability-calibration superiority is claimed by this release.
