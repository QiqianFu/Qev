# Validation of the standalone source package

Validated on 2026-09-28, using Python 3.12, Torch 2.8.0+cpu, Transformers 5.17.0 and PEFT 0.21.0. The package is imported from the standalone Qev checkout; the original research package is not on its import path.

| Check | Observed result |
|---|---|
| Full CPU pytest suite | **284 passed, 10 skipped** in 250.13 seconds |
| Offline hybrid-Qwen smoke run | Two training steps with save/resume; three output types; six dev questions, zero rejections |
| Portable checkpoint unit test | Legacy-format load, export, local base override and exact output roundtrip on a tiny model |
| New-data initialization | New dataset with fresh optimizer/schedule, separately from same-run resume |
| Wheel build and isolated install | Built on a local temporary filesystem; imported outside both source checkouts and reproduced tiny-model predictions exactly |
| JevBench preparation | Downloaded and hash-verified the pinned archive; all 231 tasks retained, all views rejected for training; 40 over training limits retained |
| Formal 9B export | Seven file hashes checked; adapter/head/gate tensors byte-identical to the selected checkpoint; optimizer excluded |
| Visuals | SVGs rendered and inspected; interactive HTML script parses and static DOM references resolve; browser interaction not rerun |
| Release checker | Local file links, source syntax, path independence, table arithmetic and all 231 frozen JevBench predictions |

The skipped tests require CUDA, including two-GPU FSDP cases. A new full-size Qev-9B GPU inference run, GPU numerical regression, and benchmark rerun have not been performed as part of this packaging task. Benchmark tables preserve the earlier research results; package tests and the random tiny-model smoke run do not remeasure those scores.

The 9B inference export preserves tensor bytes and omits optimizer state. Its `SHA256SUMS.json` records the seven inference files. The source-package checks above preceded the public weight release documented below.

Reproduce the checks in a suitable environment:

```bash
python -m pytest -q
python scripts/smoke.py --out runs/smoke-new
python scripts/check_release.py
```

A portable [machine-readable validation record](../results/validation.json) stores the observed outcomes. The CPU GitHub Actions workflow is supplied for subsequent pushes; it has not been run on GitHub during this local preparation.

## Hugging Face release verification — 2026-09-29

[AustinFu/Qev-9B](https://huggingface.co/AustinFu/Qev-9B) is public. Tag `v0.1.0` resolves to commit `54a0dbfd2b9416b556a54d6c185cb9667a24902b`, containing the selected seed-17, step-2327 checkpoint.

- Downloaded the release anonymously into a fresh cache through Qev's checkpoint resolver.
- Downloaded and checked all 14 published files against the staged package. All SHA256 hashes match; the seven inference files also match the original export inventory.
- Loaded the downloaded tokenizer and read all three safetensors weight-file headers successfully.
- Re-ran the five public-interface tests: **5 passed**, covering export/reload, Python predictions, checkpoint resolution, data preparation, and new-data initialization.

The [release verification record](../results/huggingface-release.json) contains the commit, file sizes and hashes. This check verifies publication and download; full-size GPU inference and benchmark measurements were not rerun.
