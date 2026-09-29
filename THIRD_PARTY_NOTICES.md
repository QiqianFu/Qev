# Licensing and attribution

## Qev materials

Copyright 2026 Qiqian Fu and Qev contributors.

Qev's code, documentation, original SVG illustrations, example requests, and generated prediction results are provided under the [Apache License 2.0](LICENSE). The Qev adaptation weights published at [AustinFu/Qev-9B](https://huggingface.co/AustinFu/Qev-9B), including its LoRA adapter, decision head, and interaction gate, use the same license. Contributions are accepted under these terms unless explicitly stated otherwise.

Third-party materials retain the terms listed below. The Qev license does not relicense training datasets, external services, or third-party recordings.

## Included and adapted materials

| Material | Source and copyright | License and scope |
|---|---|---|
| Kev conventions and adapted implementation | [Jared Palmer's Kev](https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer | [Apache-2.0, original notice retained](licenses/Kev-Apache-2.0.txt). Qev adapts delimiter and rendering conventions, LoRA targets, and cache forking. Modified source files identify the Qev adaptation. |
| Qwen model and tokenizer | [Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base), Copyright 2026 Alibaba Cloud | [Apache-2.0](licenses/Qwen3.5-Apache-2.0.txt). The tokenizer is included in the Hugging Face checkpoint; base weights are downloaded separately. |
| JevBench public evaluation tasks | [JevBench](https://github.com/fstandhartinger/jevbench), Copyright 2026 Florian Standhartinger and contributors | [MIT](licenses/JevBench-MIT.txt), with [upstream third-party notices](licenses/JevBench-THIRD-PARTY.md). Tasks are downloaded by the optional preparation script. The original license, third-party notice, and per-question provenance accompany the prepared dataset. |

The reviewed Kev version is [557598f](https://github.com/jaredpalmer/kev/tree/557598fced1dada75dfbf36ed144dce309ac6ceb); the evaluation data use [JevBench v1.4.2](https://github.com/fstandhartinger/jevbench/tree/1df665e3956d7aab7fa0208ff6c4f2d8557f9f90). The 231 downloaded public tasks carry MIT in their individual provenance records; upstream notices describe the scope of other JevBench materials and evaluated services.

The README layout takes inspiration from [JevAny](https://github.com/weitianxin/JevAny). Qev's SVG illustrations and chart code are original; this release includes no JevAny game code or recordings.

## Dependencies installed separately

Dependencies are installed from their own distributions and retain their licenses:

- [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE): BSD-3-Clause, plus its bundled third-party notices.
- [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE), [PEFT](https://github.com/huggingface/peft/blob/main/LICENSE), [Accelerate](https://github.com/huggingface/accelerate/blob/main/LICENSE), [Safetensors](https://github.com/huggingface/safetensors/blob/main/LICENSE), [Hugging Face Hub](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE), and optional [Apache Arrow](https://github.com/apache/arrow/blob/main/LICENSE.txt): Apache-2.0.
- Development tools [pytest](https://github.com/pytest-dev/pytest/blob/main/LICENSE) and [Matplotlib](https://matplotlib.org/stable/project/license.html) retain their MIT and Matplotlib/PSF-based license terms, respectively.

Their source code and dependency distributions are not vendored in Qev's Python package. This list identifies direct dependencies; their own packages include notices for transitive components.

## Training data

The complete research training mixture is not distributed here. Public weight availability does not make the training data available under Apache-2.0. Any future dataset release needs its own source-specific permissions and attribution. The Wikipedia-derived training subset, for example, carries CC BY-SA attribution where applicable.

The small JSONL requests in `examples/` were written for Qev and are covered by its Apache-2.0 license. The benchmark result tables and prediction files contain Qev's recorded outputs, not the original benchmark question text.
