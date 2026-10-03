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
