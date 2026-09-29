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

- CPU regression: **306 passed, 10 skipped, 6 deselected**. The current sandbox prevents loopback TCP, so six distributed-process tests could not run; the ten skipped cases require CUDA.
- The 25 distillation checks pass, including teacher targets for augmented inputs, label-blind edits, response geometry and gradients, and exact CPU continuation/resume. Probability distillation also runs through the standard trainer and rejects changed teacher targets on resume.
- The selected 2B export preserves all three trained tensor files exactly and omits optimizer state. Zero-preview research metadata loads through the public checkpoint format.
- The 2B student and native-base results are recomputed on the same clean development subsets and 144 handwritten SemIf questions used by the existing tables. All seven benchmark groups and both new JevBench prediction files are checked.
- The updated cover, grouped bars and six-model matrix were rendered and visually inspected. The model's full-size GPU inference and benchmarks were not rerun.

These changes were prepared in an isolated checkout because the current session cannot write to the destination repository. The 2B Hub release remains pending; publication status is recorded in its [model card](model-card-2b.md).
