# Qev-9B model card

**Status:** source release preparation with a local portable checkpoint export. No public Hub repository has been assigned. Publication of this repository does not imply the full training dataset is distributed.

| Field | Value |
|---|---|
| Model name | Qev-9B |
| Research name | BranchKev 9B |
| Base | Qwen/Qwen3.5-9B-Base |
| Base revision | `68c46c4b3498877f3ef123c856ecfde50c39f404` |
| Adaptation | LoRA r=64, alpha=128; learned set head and joint gate |
| Selected checkpoint | Step 2327, seed 17 |
| Readout | Shared 4096→256 projection; two 4-head Transformer layers; scalar scorer |
| Backbone interaction | `last-full-attention` |
| Stored precision | BF16 backbone execution; exported LoRA/head/gate tensors are FP32 |
| Temperature | 1.0; no fitted calibration claimed |
| Training input limits | state 1024, question 512, candidate 256, path 2048, candidates 128 |
| Training record counts | 34,546 main and 1,783 late; 249 records intentionally overlap |
| Formal configuration | [qev-9b.json](../configs/qev-9b.json) |

## Intended use

Research and development of decisions over explicit choices: routing, rule judgments, and rubric ratings. The same context can support several independent questions. Evaluate on the actual domain, language, candidate descriptions and decision thresholds of the application.

## Measured performance

Public JevBench accuracy is 188/231 (81.39%). Full results, exact denominators, baseline provenance, numerical settings and the architecture ablations are in [evaluation.md](evaluation.md). The selected checkpoint is a single seed, and public benchmarks were used during research iteration.

## Boundaries

Longer inference limits do not imply equivalent long-context training coverage. Choice probabilities depend on the supplied candidate set. Accuracy differences do not establish calibration, architecture-specific gains, or production reliability. This source release does not include a serving API compatible with every TypeSafe SDK method; its Python and JSONL interfaces are documented in the README.

## Files and availability

The inference export contains adapter weights, a set head, a joint gate, tokenizer, model metadata and SHA256 inventory. It excludes Qwen base weights and optimizer state. The loader retrieves the pinned base separately or accepts a compatible local base cache. [Export instructions](checkpoints.md).

Code uses Apache-2.0 with Kev attribution. Base models and source datasets retain their upstream terms. The complete mixed training corpus is not redistributed in this repository. A hosted model release should publish this card, the exact base revision, inference hashes and applicable weight-license metadata together.
