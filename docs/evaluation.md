# Evaluation and provenance

Reported results are from the frozen research runs, not new training performed while preparing this repository. Qev-9B is the research model previously called BranchKev 9B, run `c21-science-wk-late1783-9b-4gpu-r64-late50x3-s17`, step 2327.

[Machine-readable results](../results/benchmarks.json) contain integer numerators and denominators. [Qev's 231 JevBench predictions](../results/qev-9b/jevbench-predictions.jsonl) are copied byte-for-byte from the recorded run, with [hash and checkpoint provenance](../results/qev-9b/provenance.json).

## Models and sources

| Model | Identity | Result source |
|---|---|---|
| Qev-9B | Qwen3.5-9B-Base + rank-64 LoRA + 256×2 head, seed 17, step 2327 | Local BF16 full causal reference execution |
| Kev-9B | `jaredpalmer/kev-9b@2629c06a`, Kev code `557598fced1dada75dfbf36ed144dce309ac6ceb` | JevBench rerun locally with Kev's own `kev.benchmark`, FP32 and T=1; other suites from pinned author reports |
| Qwen3.5-9B-Base | `68c46c4b3498877f3ef123c856ecfde50c39f404` | Frozen native LM head, zero-shot prompt, probabilities normalized over valid answer codes |
| Jev | Hosted service; JevBench used Jev 1.13.0 | JevBench API run on 2026-09-26; other suites from Jev reports preserved by the Kev authors |

The pinned Kev reports include `night2-9b-du`, `r4-kev-9b-*-raw`, and `kev-9b-ekzhang-mmlupro-2`. They are available in the [upstream snapshot](https://github.com/jaredpalmer/kev/tree/557598fced1dada75dfbf36ed144dce309ac6ceb/runs). The consolidated research report hash used for the public table is recorded in `results/benchmarks.json`; code extraction is recorded in `provenance.json`.

## Full accuracy table

Percent accuracy. The selected Qev model is a single seed. Bold scores mark the higher result between Qev and Kev; ties are unbolded. Jev is a hosted reference.

**Precision comparison: Kev-9B uses FP32; Qev-9B uses BF16 backbone computation.** Qev retains FP32 for its decision head and key reductions; its exported LoRA tensors are stored in FP32. These are not identical precision settings.

| Scope | Qev-9B | Kev-9B | Qwen base | Jev (reference) |
|---|---:|---:|---:|---:|
| decision_dev all · 1468 | **88.28** | 87.81 | 75.75 | 83.17 |
| decision_dev clean · 1264 | **87.42** | 87.18 | 77.69 | 84.49 |
| transfer_dev all · 764 | **82.46** | 81.15 | 73.43 | 84.69 |
| transfer_dev clean · 656 | **83.99** | 82.16 | 74.39 | 85.67 |
| MMLU-Pro · 1000 | **54.60** | 51.10 | 50.40 | 83.50 |
| SemIf handwritten · 144 | **93.75** | 90.97 | 90.28 | 96.53 |
| scienthoon · 873 | 72.28 | **75.49** | 68.84 | 75.26 |
| WANLI · 256 | **72.66** | 70.31 | 67.97 | 75.78 |
| JevBench public · 231 | **81.39** | 75.76 | 75.76 | 85.71 |

The all-question dev rows include clean examples and candidate-permutation / None-present / None-absent variants. Clean rows match the clean reporting convention in Kev's README. SemIf uses 144 handwritten questions. Qev's broader 252-question research run also included 108 perturbations; its 95.63% overall score is not the 144-question comparison above.

## Public JevBench breakdown

| Subset | Questions | Qev-9B correct | Kev-9B correct | Qwen base correct | Jev correct (reference) |
|---|---:|---:|---:|---:|---:|
| original | 72 | 65 | 65 | 59 | 71 |
| easy | 48 | 48 | 48 | 48 | 48 |
| hard | 111 | **75** | 62 | 68 | 79 |
| all | 231 | **188** | 175 | 175 | 198 |

This is local argmax accuracy on the public v1.4.2 tasks. It is not the official composite score, which includes other dimensions and nonpublic tasks. The public benchmark was observed during research iteration, so these results are not an untouched final blind test.

Precision, prompts, and implementations differ across models; these are model-level observations, not a controlled architecture comparison.

## Rerun Qev's public evaluation

Download the released checkpoint, then prepare and evaluate the public tasks:

```bash
hf download AustinFu/Qev-9B --revision v0.1.0 --local-dir checkpoints/qev-9b

python scripts/prepare_jevbench.py \
  --root data/jevbench --source-dir data/jevbench/source \
  --tokenizer checkpoints/qev-9b/tokenizer --config configs/qev-9b.json

python -m qev.evaluate --checkpoint checkpoints/qev-9b \
  --data data/jevbench/model_data/jevbench-public-v1.4.2 --split public \
  --out runs/jevbench-reference --device cuda \
  --reference --weights-dtype checkpoint --max-state 4096 --max-path 4096
```

The preparer fetches archive revision `1df665e3956d7aab7fa0208ff6c4f2d8557f9f90`, verifies the archive and upstream task hashes, retains gold distributions, and keeps all tasks even if they exceed training limits. Every exported view has an `external_evaluation` role. Re-running the preparation command on its completed directory verifies it instead of replacing it. Use `--compare-data` to audit canonical input overlap with supplied training datasets.

To evaluate a prepared user-data dev split:

```bash
python -m qev.evaluate --checkpoint runs/support/step-000100 \
  --data data/support --split dev --out runs/support-eval \
  --reference --weights-dtype checkpoint
```

Use an actual saved step. Reports preserve rejected items and both answered-only accuracy and accuracy counting rejections wrong. Probability checks, Brier, NLL and ECE are also computed; their availability does not mean the released checkpoint has been calibrated.

## Architecture ablations

Same data recipe, rank 64, seed 17 and final step 2327. Qev's selected release is A.

| Variant | Backbone sibling cross | Set attention layers | JevBench correct | Mean P(gold label) |
|---|---|---:|---:|---:|
| A · selected | All sibling tokens | 2 | 188/231 | 0.782 |
| B | Off | 2 | 185/231 | 0.775 |
| C | All sibling tokens | 0 | 189/231 | 0.791 |
| D | Off | 0 | 187/231 | 0.780 |
| Readout-only | Sibling readout tokens | 2 | 187/231 | 0.784 |

The single-seed comparisons did not establish a measurable gain from either interaction component. Projection and the scalar scorer remain even when set attention is disabled. Related rank-16 seed sweeps had a JevBench standard deviation of about 4.7 questions; that is context, not a confidence interval or significance test for this final recipe.

scienthoon questions cluster by ticket template, so treating every question as an independent observation understates uncertainty. Jev's large knowledge-benchmark advantage does not identify its undisclosed model architecture or training data. No speed superiority or probability-calibration superiority is claimed by this release.
