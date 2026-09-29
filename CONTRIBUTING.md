# Contributing

Install the pinned dependencies and test extra, then run:

```bash
python -m pytest -q
python scripts/check_release.py
```

CPU tests cover tiny randomly initialized models, DDP, training/resume, prefix and tree execution, data roles, and checkpoint packaging. CUDA-only checks are skipped when suitable GPUs are absent; report skips explicitly.

Keep numerical changes separate from packaging changes. Changes to masks, cache forks, recurrent state, or shared execution should preserve the reference-path output and gradient checks. Keep baseline versions, evaluation denominators and raw results traceable.

Update README examples and the corresponding document when changing public behavior. Do not add machine-specific paths, credentials, optimizer state, dataset caches or downloaded weights. New data releases need their own provenance and licensing records. Do not overwrite frozen prediction evidence to make a changed implementation match an old score.

## Documentation visuals

Regenerate the architecture diagrams with `python scripts/draw_architecture.py`. Regenerate the paired benchmark chart with `python scripts/plot_results.py` (requires Matplotlib); it reads the frozen integer counts in `results/benchmarks.json`. English and Chinese README pages use matching diagram variants.

The README's presentation and paired-bar layout take inspiration from [JevAny's README](https://github.com/weitianxin/JevAny/blob/main/README.zh-CN.md). Qev's diagrams are original vector assets, and its charts use its own recorded results. Gameplay assets and recording details are documented in [the recording guide](docs/demos.md).
