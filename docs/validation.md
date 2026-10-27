# Validation of the standalone source package

Validated on 2026-09-28, using Python 3.12, Torch 2.8.0+cpu, Transformers 5.17.0 and PEFT 0.21.0. The package is imported from the standalone Qev checkout; the original research package is not on its import path.

| Check | Observed result |
|---|---|
| Full CPU pytest suite | **284 passed, 10 skipped** in 250.13 seconds |
| Offline hybrid-Qwen smoke run | Two training steps with save/resume; three output types; six dev questions, zero rejections |
| Portable checkpoint unit test | Legacy-format load, export, local base override and exact output roundtrip on a tiny model |
| New-data initialization | New dataset with fresh optimizer/schedule, separately from same-run resume |
| Wheel build and isolated install | Built on a local temporary filesystem; imported outside both source checkouts and reproduced tiny-model predictions exactly |
| JevBench preparation | Downloaded the JevBench release; all 231 tasks retained, all views rejected for training; 40 over training limits retained |
| Formal 9B export | Adapter/head/gate tensors byte-identical to the selected checkpoint; optimizer excluded |
| Visuals | SVGs rendered and inspected; interactive HTML script parses and static DOM references resolve; browser interaction not rerun |
| Release checker | Local file links, source syntax, path independence, table arithmetic and all 231 frozen JevBench predictions |

The skipped tests require CUDA, including two-GPU FSDP cases. A new full-size Qev-9B GPU inference run, GPU numerical regression, and benchmark rerun have not been performed as part of this packaging task. Benchmark tables preserve the earlier research results; package tests and the random tiny-model smoke run do not remeasure those scores.

The 9B inference export preserves tensor bytes and omits optimizer state. The source-package checks above preceded the public weight release documented below.

Reproduce the checks in a suitable environment:

```bash
python -m pytest -q
python scripts/smoke.py --out runs/smoke-new
python scripts/check_release.py
```

A portable [machine-readable validation record](../results/validation.json) stores the observed outcomes. The CPU GitHub Actions workflow is supplied for subsequent pushes; it has not been run on GitHub during this local preparation.

## Hugging Face release verification — 2026-09-29

[AustinFu/Qev-9B](https://huggingface.co/AustinFu/Qev-9B) is public. The `v0.1.0` tag preserves the original published weights.

- Downloaded the release anonymously into a fresh cache through Qev's checkpoint resolver.
- Downloaded and compared all 14 original release files with the local package; the inference tensors match the selected checkpoint.
- Loaded the downloaded tokenizer and read all three safetensors weight-file headers successfully.
- Re-ran the five public-interface tests: **5 passed**, covering export/reload, Python predictions, checkpoint resolution, data preparation, and new-data initialization.

The [original release record](../results/history/huggingface-v0.1.0.json) preserves the publication verification. This check verifies publication and download; full-size GPU inference and benchmark measurements were not rerun.

## Public interface and licensing cleanup — 2026-09-29

- Full CPU suite: **286 passed, 10 skipped**. A subsequent targeted run of the updated model-loading, base-evaluation and JevBench checks passed all **15 tests**, including the new conversion-and-license regression.
- The offline smoke workflow completed data preparation, two training steps with resume, all three prediction types, and six evaluation questions.
- JevBench preparation downloaded and converted all 231 public tasks, preserved their labels and probability targets, and included the original license and third-party notice. Conversion verification passed.
- A built Python wheel includes Qev's license and notices plus the original Kev, Qwen and JevBench license files.
- Hugging Face documentation and license files were downloaded anonymously and compared with the published sources. The three trained weight files are unchanged, and the original `v0.1.0` release is preserved.

These changes simplify packaging and usage; the GPU-only tests remain skipped and the benchmark scores are the original model measurements.

## Qev-2B release preparation — 2026-09-29

- CPU regression: **306 passed, 10 skipped, 6 deselected**. Six distributed-process tests were not run in the preparation environment; the ten skipped cases require CUDA.
- The 25 distillation checks pass, including teacher targets for augmented inputs, label-blind edits, response geometry and gradients, and exact CPU continuation/resume. Probability distillation also runs through the standard trainer and rejects changed teacher targets on resume.
- The selected 2B export preserves all three trained tensor files exactly and omits optimizer state. Zero-preview research metadata loads through the public checkpoint format.
- The 2B student and native-base results are recomputed on the same clean development subsets and 144 handwritten SemIf questions used by the existing tables. All seven benchmark groups and both new JevBench prediction files are checked.
- The updated cover, grouped bars and six-model matrix were rendered and visually inspected. The model's full-size GPU inference and benchmarks were not rerun.

The prepared changes have been integrated into the source repository, preserving the gameplay recordings, README layout and acknowledgments.

## Qev-2B publication verification — 2026-09-29

[AustinFu/Qev-2B](https://huggingface.co/AustinFu/Qev-2B) is public. Its [v0.1.0 release](https://huggingface.co/AustinFu/Qev-2B/tree/v0.1.0) contains the selected 800-step checkpoint.

- Downloaded the release anonymously into a fresh cache and compared all 20 uploaded files with the prepared package; every file matches.
- Read all three safetensors headers and loaded the downloaded tokenizer successfully.
- Resolved the tagged Hub model through Qev's checkpoint loader.

This verifies publication, download and checkpoint resolution. Full-size GPU inference and benchmark measurements were not rerun.

## Qev-train data release — 2026-09-29

[Qev-train v1.0.0](https://huggingface.co/datasets/AustinFu/Qev-train/tree/v1.0.0) contains 1,842 original synthetic training examples, with source-specific licenses and bilingual generation documentation.

- Verified every released example against the original training inputs and hard labels. No duplicate inputs or exact input/group overlaps with the checked development, calibration and test partitions were found.
- Loaded all 1,842 records through Qev's training-data loader. JSONL and Parquet both round-trip through Hugging Face Datasets without record changes.
- Validated dataset-card metadata and local documentation links.
- Downloaded all 10 uploaded files anonymously into a fresh cache and compared them with the prepared package. Anonymous `load_dataset` of the tagged release returned the same 1,842 records.
- Confirmed that the Hub dataset viewer exposes the training split and its first 100 examples, and that all 182 rule-compliance pairs contain one positive and one negative label.

This is a data publication check; model training and benchmarks were not rerun.

## Qev-9B v0.2.0 and Qev-train v1.1.0 — 2026-09-30

The current [9B weights](https://huggingface.co/AustinFu/Qev-9B/tree/v0.2.0) are the selected 2,643-step checkpoint with Principle judgments and 600 boundary questions in the main set. The [dataset update](https://huggingface.co/datasets/AustinFu/Qev-train/tree/v1.1.0) contains 2,442 examples.

- Recomputed every current 9B table entry from saved per-question predictions on the same benchmark subsets: MMLU-Pro 574/1,000 and public JevBench 192/231. All seven original evaluation files have zero rejected questions.
- Compared the three exported tensor files with the selected checkpoint exactly; read their safetensors headers and loaded the tokenizer successfully.
- Loaded and encoded all 2,442 dataset records with Qev and the real 9B tokenizer. Verified that the 600 controlled-boundary records match the training source-prefix exemption in the formal and fine-tuning configurations.
- Compared JSONL and Parquet through Hugging Face Datasets. Checked the new records against the original training set, and checked exact input/group separation from the existing development, calibration and test files.
- Downloaded and compared all 12 dataset files and 24 model files anonymously in fresh caches. The tagged dataset loaded all 2,442 records unchanged.
- Preserved the original model and dataset tags, and archived the original 9B predictions and recipe. Qev-2B weights and results are unchanged; its teacher is explicitly pinned to Qev-9B v0.1.0.
- Passed eight focused CPU tests covering data validation, versioned checkpoint resolution and source-prefix exemptions for None-option augmentation.

These checks verify the selected artifacts and their publication. Full-size GPU inference and benchmark measurements were not rerun during packaging.

## Qev-4B v0.1.0 — 2026-10-02

The [4B release](https://huggingface.co/AustinFu/Qev-4B/tree/v0.1.0) is the 2,786-step checkpoint initialized directly from Qwen3.5-4B-Base.

- Recomputed all seven suites from 4,844 saved predictions, checked probabilities and zero rejections, and matched question IDs, candidate order and labels against the existing 9B evaluation. MMLU-Pro is 503/1,000; JevBench is 190/231.
- Verified the matched README subsets, all 49 matrix cells and 35 bar labels. Previously published models' scores are unchanged.
- Compared the exported adapter, head and gate files byte-for-byte with the selected checkpoint, read their tensor headers and loaded the real 4B tokenizer.
- Encoded all 44,576 label-free training inputs through the public implementation and verified teacher coverage over both epochs: 89,152 views, with the recorded teacher temperature.
- Passed 17 focused CPU tests for teacher temperature, missing/invalid targets, training and resume on labelled and unlabelled data, cache changes, export/loading and versioned Hub resolution.
- Downloaded all 25 published files anonymously in a fresh cache and compared them with the prepared package. Verified the `v0.1.0` tag through Qev's checkpoint resolver.

These checks cover packaging and the training-code changes. The reported model scores come from the original saved evaluation; full-size 4B GPU inference and benchmarks were not rerun for this release.

## Cover chart layout — 2026-10-02

The cover now uses one grouped chart with Qev-2B, Qev-4B, Qev-9B and Qwen3.5-9B-Base, in that order. All 28 labels were checked against the recorded benchmark counts, and the rendered chart was visually inspected. The benchmark matrix and scores are unchanged. English and Chinese image descriptions were updated; no model training or inference code changed.

## Expanded evaluation and 4B cover — 2026-10-03

- Recomputed the native Qwen3.5-4B-Base, Kev-4B and both JevAny-4B results from 19,376 original-suite prediction slots, aligning IDs, candidate sets and labels with the existing Qev evaluation. Checked the clean development and handwritten SemIf subsets separately.
- Reran the pinned Decision Index 0.2.1 scorer on saved outputs from ten local models. All 40 raw scores, skill values and coverage values matched the recorded summaries. Jev's four reference scores come from the toolkit's frozen leaderboard.
- Checked every cover label, matrix cell and bilingual README score against the machine-readable results. The single cover chart has ten benchmarks and four series: Qev-4B, JevAny-4B Pointer, Kev-4B and Jev. Decision dev remains in the tables and matrix. Cover labels use one decimal for readability; tables and the matrix use two.
- Visually inspected the cover and matrix; release and whitespace checks passed. Existing model scores and weights were preserved. No training, GPU inference or benchmark reruns were performed; this update rescored recorded predictions.

## Display scope and order — 2026-10-03

Removed Amazon ESCI from the displayed charts and tables and moved JevBench to the first position. The cover has nine benchmarks and 36 labels; the matrix has ten benchmarks and 110 cells. The recorded metrics remain unchanged. Checked the displayed values and table order, visually inspected the regenerated figures, and ran the release and whitespace checks.

## README layout and method figure — 2026-10-03

Removed the separate model/checkpoint chapter from both READMEs and placed Demos, How decisions are made and Evaluation immediately after Inference, followed by Training. Model-download navigation now targets the existing model selector. The supplied method PDF is included as `assets/method.pdf`, with a self-contained SVG conversion for inline display and a link to the original PDF. Checked section order, navigation targets, SVG references and exact preservation of the supplied PDF.

## Qev-9B v0.3.0 — 2026-10-03

The [9B release](https://huggingface.co/AustinFu/Qev-9B/tree/v0.3.0) contains the 2,658-step checkpoint with web-action and additional rule/reasoning training. Earlier model tags and the [v0.2.0 result snapshot](../results/history/qev-9b-v0.2.0/README.md) preserve the previous release.

- Recomputed the seven original suites from 4,844 saved predictions and checked question IDs, candidate order and labels against the previous evaluation. MMLU-Pro is 565/1,000; JevBench is 187/231. Re-scored all four additional benchmarks with the pinned official Decision Index scorer.
- Checked all 110 matrix cells and 30 bilingual README/model-card values against the recorded results. The 4B cover chart and other models' scores are unchanged.
- Verified the three exported tensor files against the selected checkpoint and the original 4B teacher. Encoded all 41,081 main/late training records, containing 48,386 questions, with the public encoder and real tokenizer. All 2,442 released Qev-train inputs and labels remain in the training mixture.
- Passed three focused CPU tests covering portable export and loading, versioned Hub resolution, and source-prefix exemptions for option augmentation.
- Downloaded and compared all 28 release files anonymously in a fresh cache; resolved the tag through Qev's public loader, loaded the tokenizer and read the three tensor headers.
- Updated the 4B model card to identify its now-published teacher, downloaded and compared all five changed metadata files, and verified that 4B weights and tags are unchanged. Qev-2B retains its original v0.1.0 teacher.

These checks validate packaging, recorded scores and publication. Full-size GPU inference, training and benchmark measurements were not rerun during this release.

## README comparison and chart readability — 2026-10-04

Added the standard JevAny-4B Pointer results to both README tables and the full evaluation tables. The cover now shows seven benchmarks in this order: MMLU-Pro, SemIf, JevBench, scienthoon, WANLI, ChessBench and BPoMP. Transfer development and GSM8K remain in the tables and matrix. Narrower side margins, a smaller canvas width and larger fonts make labels more legible at README width.

Checked all 28 bar values and 32 added table values against the recorded results, preserved existing table cells and the full matrix, and inspected the chart at 940 pixels wide. Ran the plotting script, release checker and whitespace checks; model code and benchmark measurements are unchanged.

## Highlighted README tables — 2026-10-04

The English and Chinese READMEs now display SVG tables with light blue backgrounds for Qev-9B, Qev-4B and Qev-2B. The original Markdown tables remain available in expandable sections for copying. `python scripts/render_results_table.py` regenerates the images from those tables; rendering Chinese requires a Noto Sans CJK font.

Verified all 180 displayed scores against the source tables and recorded metrics, checked the three highlighted columns, and inspected both rendered tables. The cover remains the seven-benchmark version with JevBench third; its public GitHub image contains neither Transfer development nor GSM8K. No benchmark scores or model code changed.

## Consolidated evaluation guide — 2026-10-04

Removed the expandable README tables and replaced the separate evaluation links with one [complete evaluation guide](evaluation.md). It now includes the full matrix, both JevAny variants, Decision Index methods and sources, JevBench details, reproduction commands, ablations and original result files. The old method-note URL points to the consolidated section.

The table renderer now reads `results/benchmarks.json` and `results/decision-index.json` directly. Regeneration preserved all 180 values, labels, bold formatting and blue column backgrounds; both SVGs are visually identical after ignoring generated clipping IDs. The cover and recorded results are unchanged. Documentation and release checks passed.

## Best scores by model size — 2026-10-04

The bilingual tables and full matrix now emphasize the best displayed scores within the 4B and 2B groups, including ties. The 9B Qev/Kev comparison is retained, and Jev remains an unbolded reference. Checked all 290 SVG scores and their emphasis, preserved every Markdown table value, and inspected the rendered table. The cover, blue backgrounds and recorded metrics are unchanged.

## Qev-0.8B and NanoJev — 2026-10-05

Selected the directly initialized, teacher-distilled Qev-0.8B checkpoint at step 2,786, seed 17. Recomputed the matched reporting subsets from 4,844 stored predictions per model (9,688 total) and retained all previously recorded scores. The small models have no Decision Index measurements; their missing cells are displayed as dashes.

Validated 160 bilingual table cells and 100 matrix cells, including emphasis and the four blue Qev columns. Inspected the rendered tables and matrix. The existing 4B cover is byte-identical. Exported adaptation tensors and tokenizer are identical to the original checkpoint; loaded the actual tokenizer, checked all tensor headers, and matched the training/fine-tuning configurations to the exported model metadata.

Three focused CPU tests passed: portable export, legacy loading and Python API; Hub version resolution; teacher temperature and cache preservation. Release consistency and documentation checks passed. Full-size GPU inference, training and benchmark measurements were not rerun.

<!-- qev-0.8b-publication:start -->
Published [Qev-0.8B v0.1.0](https://huggingface.co/AustinFu/Qev-0.8B/tree/v0.1.0) at `9fc71346c2aed987c7c3e5a3d8e01c02a3e97f5c`. Downloaded all 25 files anonymously into a fresh cache and compared them with the prepared package. The tag and main resolve to the verified public checkpoint.
<!-- qev-0.8b-publication:end -->

## Incremental sessions — 2026-10-26

CPU: the full test suite passed (333 passed, 10 skipped; the skips need CUDA or do not apply to an interaction mode). `tests/test_session.py` checks that only appended state tokens are computed, that questions leave the state cache bit-identical, and that cached, reference and tree sessions agree.

GPU (one A40, PyTorch 2.8.0, Transformers 5.17.0): the session, prefix-inference and tree-inference tests passed on CUDA with the fused `causal-conv1d` 1.7.0 and `fla` 0.5.2 kernels and with the PyTorch reference kernels (82 passed, 4 skipped each). With the released Qev-0.8B and Qev-9B v0.3.0 weights in BF16, a six-chunk session computed only appended tokens, including a single-token chunk, and left the state cache bit-identical after questions. Against reference execution on the same tokens, the largest probability difference was 0.025 (0.8B) and 0.020 (9B), already 0.025 before any extension; one 0.8B prediction differed. Session tokenization effects are described in [incremental sessions](checkpoints.md#incremental-sessions). Single-run timing is not a benchmark. Task scores were not measured for sessions.

## Copy-free cache forks and chunked prefill — 2026-10-27

CPU: the full test suite passed (352 passed, 11 skipped; the new skip is a CUDA-only allocator test). `tests/test_cache_forks.py` checks that question and candidate forks reference the parent KV tensors themselves and store only their own tokens, that the parent cache stays bit-identical, that the merged prefix/suffix attention matches dense attention, that shared forks match copying forks for several `rows_per_forward` values and both attention backends, and that chunked prefill matches a single forward.

GPU (one A40 shared with an unrelated load, PyTorch 2.8.0, Transformers 5.17.0): the fork, session, prefix-inference, tree-inference and model tests passed on CUDA with the fused kernels and with the PyTorch reference kernels (126 passed, 4 skipped each). On CUDA, question peak memory no longer grew with state length, while copying forks grew by more than eight state-KV copies.

Released Qev-0.8B and Qev-9B v0.3.0 in BF16 were compared with the previous implementation on 300-, 1,500- and 3,800-token states and choice (4 candidates), noul and score questions. Reference execution was bit-identical. Cached one-shot (multi- and single-question) and three-chunk session probabilities differed from the previous cached execution by at most 0.011 (0.8B) and 0.034 (9B), with no changed prediction among 90 compared questions. Over the same inputs, previous cached execution differed from reference by up to 0.012 (0.8B) and 0.022 (9B); the new one by up to 0.010 and 0.034. The difference comes from merging two BF16 attention outputs instead of one fused softmax.

Qev-9B sessions (single runs on the shared GPU; not a benchmark). Peak is the allocator maximum above the memory held before the step:

| State tokens | Version | Ingest | Ingest peak | Choice (4) | Noul | Score (5) |
|---:|---|---:|---:|---:|---:|---:|
| 4,096 | previous | 3.8 s | 0.93 GiB | 0.26 s, 1.13 GiB | 0.24 s, 0.66 GiB | 0.27 s, 1.36 GiB |
| 4,096 | new | 3.8 s | 0.93 GiB | 0.23 s, 0.28 GiB | 0.22 s, 0.18 GiB | 0.23 s, 0.33 GiB |
| 16,384 | previous | 18.9 s | 3.55 GiB | 0.41 s, 3.75 GiB | 0.33 s, 2.21 GiB | 0.44 s, 4.55 GiB |
| 16,384 | new | 11.7 s | 2.24 GiB | 0.26 s, 0.38 GiB | 0.25 s, 0.28 GiB | 0.26 s, 0.43 GiB |
| 50,000 | previous | 44.2 s | 10.75 GiB | 0.79 s, 10.95 GiB | 0.59 s, 6.45 GiB | 0.90 s, 13.28 GiB |
| 50,000 | new | 50.4 s | 3.99 GiB | 0.32 s, 0.66 GiB | 0.32 s, 0.55 GiB | 0.34 s, 0.72 GiB |

The 50,000-token state cache holds 1.59 GiB. The remaining growth of question peaks with state length is the FP32 copy of one layer's prefix KV in the candidate-interaction readout. On these inputs, long-state predictions agreed with the previous implementation within 0.009 and kept every prediction. Task accuracy beyond the released 4,096-token limit is measured in [long unrelated prefix](#long-unrelated-prefix-on-jevbench--2026-10-27).

## Long unrelated prefix on JevBench — 2026-10-27

Qev-9B v0.3.0 (BF16, cached execution, `max_state` raised to 58,192) answered the 231 public JevBench questions with unrelated text placed before each state (`filler + "\n\n" + state`). The filler was computed once per condition and reused for every record; the reused cache stayed bit-identical, and at 50,000 tokens five records computed from scratch agreed with the reused-prefix answers (largest probability difference 0.028, no changed prediction). Without filler the cached path matched the released result, 187/231. Accuracy and exact McNemar p-values against no filler:

| Filler tokens | Prose (git and PCRE2 manuals) | Python standard-library source |
|---:|---:|---:|
| 8,192 | 73.2% (p = 0.004) | 80.1% (p = 0.85) |
| 16,384 | 66.7% (p < 0.001) | 81.0% (p = 1.00) |
| 32,768 | 76.2% (p = 0.06) | 80.1% (p = 0.86) |
| 50,000 | 74.5% (p = 0.014) | 72.3% (p = 0.004) |

Prose hurt more than code, and accuracy did not fall monotonically with length; 26–59 predictions changed in every condition. Almost all losses were on the `original` subset of short policy questions (67/72 without filler, 43/72 with 16K of prose); `easy` was unchanged. The models were trained on states up to 4,096 tokens, so this measures robustness outside the training range, not a release score. With the filler prefix, the 50,000-token cache took 1.58 GiB, its ingest 51 s, and each record answered over it took a median 0.60 s on an A40 shared with an unrelated load.
