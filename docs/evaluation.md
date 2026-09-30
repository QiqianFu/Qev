# Evaluation and provenance

These results describe Qev-9B v0.2.0: Qwen3.5-9B-Base with rank-64 LoRA, candidate interaction, and a two-layer decision head. The main training set includes 4,459 HelpSteer3 Principle judgments and 600 synthetic boundary questions. The 1,783-example alignment/document-rule partition is mixed into the second half of training and repeated three times.

[Machine-readable results](../results/benchmarks.json) contain integer numerators and denominators. [Qev's 231 JevBench predictions](../results/qev-9b/jevbench-predictions.jsonl) are preserved from the original evaluation.

Qev-2B uses probability distillation followed by context edits, original-question replay and representation-response matching. Its MMLU-Pro accuracy is 38.70%, and its public JevBench accuracy is 172/231 (74.46%). Both 2B models use BF16 backbone computation. See the [training method](distillation.md).

## Models and sources

| Model | Identity | Result source |
|---|---|---|
| Qev-9B | Qwen3.5-9B-Base + rank-64 LoRA + 256×2 head, seed 17, step 2643 | Local BF16 full causal reference execution |
| Kev-9B | [jaredpalmer/kev-9b](https://huggingface.co/jaredpalmer/kev-9b) | JevBench rerun locally with Kev's own `kev.benchmark`, FP32 and T=1; other suites from pinned author reports |
| Qwen3.5-9B-Base | [Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base) | Frozen native LM head, zero-shot prompt, probabilities normalized over valid answer codes |
| Qev-2B | Qwen3.5-2B-Base + rank-64 LoRA + 256×2 head; no candidate preview | Recorded reference execution after response distillation |
| Qwen3.5-2B-Base | Original language-model head, zero-shot candidate codes | Recorded native-base evaluation |
| Jev | Hosted service; JevBench used Jev 1.13.0 | JevBench API run on 2026-09-26; other suites from Jev reports preserved by the Kev authors |

Kev's published results are available in its [evaluation reports](https://github.com/jaredpalmer/kev/tree/557598fced1dada75dfbf36ed144dce309ac6ceb/runs). JevBench was evaluated locally for Qev, Kev, and the Qwen base, and through the Jev API for the hosted reference.

## Benchmark matrix

![Accuracy for six models on seven matched subsets](../assets/evaluation-matrix.svg)

## Full accuracy table

Percent accuracy. The selected Qev model is a single seed. Bold scores mark the higher result between Qev and Kev; ties are unbolded. Jev is a hosted reference.

**Precision comparison: Kev-9B uses FP32; Qev-9B uses BF16 backbone computation.** Qev retains FP32 for its decision head and key reductions; its exported LoRA tensors are stored in FP32. These are not identical precision settings.

| Scope | Qev-9B | Kev-9B | Qwen base | Jev (reference) | Qwen 2B base | Qev-2B |
|---|---:|---:|---:|---:|---:|---:|
| decision_dev all · 1468 | **88.22** | 87.81 | 75.75 | 83.17 | 63.15 | 86.24 |
| decision_dev clean · 1264 | **87.42** | 87.18 | 77.69 | 84.49 | 65.43 | 85.36 |
| transfer_dev all · 764 | **82.46** | 81.15 | 73.43 | 84.69 | 64.53 | 76.44 |
| transfer_dev clean · 656 | **83.99** | 82.16 | 74.39 | 85.67 | 65.09 | 77.29 |
| MMLU-Pro · 1000 | **57.40** | 51.10 | 50.40 | 83.50 | 31.20 | 38.70 |
| SemIf handwritten · 144 | **93.06** | 90.97 | 90.28 | 96.53 | 63.89 | 82.64 |
| scienthoon · 873 | 71.02 | **75.49** | 68.84 | 75.26 | 53.84 | 71.94 |
| WANLI · 256 | **71.09** | 70.31 | 67.97 | 75.78 | 50.39 | 67.58 |
| JevBench public · 231 | **83.12** | 75.76 | 75.76 | 85.71 | 63.20 | 74.46 |

The all-question dev rows include clean examples and candidate-permutation / None-present / None-absent variants. Clean rows match the clean reporting convention in Kev's README. SemIf uses 144 handwritten questions. Qev's broader 252-question research run also included 108 perturbations; its 95.63% overall score is not the 144-question comparison above.

## Public JevBench breakdown

| Subset | Questions | Qev-9B correct | Kev-9B correct | Qwen base correct | Jev correct (reference) |
|---|---:|---:|---:|---:|---:|
| original | 72 | 68 | 65 | 59 | 71 |
| easy | 48 | 48 | 48 | 48 | 48 |
| hard | 111 | **76** | 62 | 68 | 79 |
| all | 231 | **192** | 175 | 175 | 198 |

This is local argmax accuracy on the public v1.4.2 tasks. It is not the official composite score, which includes other dimensions and nonpublic tasks. The public benchmark was observed during research iteration, so these results are not an untouched final blind test.

Precision, prompts, and implementations differ across models; these are model-level observations, not a controlled architecture comparison.

## Rerun Qev's public evaluation

Download the released checkpoint, then prepare and evaluate the public tasks:

```bash
hf download AustinFu/Qev-9B --revision v0.2.0 --local-dir checkpoints/qev-9b

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

These historical ablations use the Qev-9B v0.1.0 data recipe, rank 64, seed 17 and final step 2327. Their results remain separate from the current v0.2.0 leaderboard. Both releases retain the full interaction architecture.

| Variant | Backbone sibling cross | Set attention layers | JevBench correct | Mean P(gold label) |
|---|---|---:|---:|---:|
| Qev-9B v0.1.0 reference | All sibling tokens | 2 | 188/231 | 0.782 |
| No backbone interaction | Off | 2 | 185/231 | 0.775 |
| No set attention | All sibling tokens | 0 | 189/231 | 0.791 |
| Neither interaction | Off | 0 | 187/231 | 0.780 |
| Readout-only | Sibling readout tokens | 2 | 187/231 | 0.784 |

The single-seed comparisons did not establish a measurable gain from either interaction component. Projection and the scalar scorer remain even when set attention is disabled. Related rank-16 seed sweeps had a JevBench standard deviation of about 4.7 questions; that is context, not a confidence interval or significance test for this final recipe.

scienthoon questions cluster by ticket template, so treating every question as an independent observation understates uncertainty. Jev's large knowledge-benchmark advantage does not identify its undisclosed model architecture or training data. No speed superiority or probability-calibration superiority is claimed by this release.
