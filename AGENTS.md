# Qev contributor instructions

Qev is a standalone public source package. Read README.md and the relevant docs before changing interfaces.

- Keep the qev package independent of the original research checkout and cluster.
- Update English/Chinese README examples and the corresponding docs for interface changes.
- Preserve compatibility with recorded BranchKev schema/checkpoint formats where documented.
- Architecture changes require appropriate reference/cache/tree output and gradient checks.
- Report observed test scope and GPU skips; never infer benchmark gains from a code refactor.
- Keep weights, data caches, training outputs and machine-local paths outside Git.
- Run python scripts/check_release.py and relevant pytest tests before committing.
