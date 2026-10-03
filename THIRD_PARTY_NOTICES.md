# Licensing and attribution

## Qev materials

Copyright 2026 Qiqian Fu and Qev contributors.

Qev's code, documentation, original SVG illustrations, example requests, and generated prediction results are provided under the [Apache License 2.0](LICENSE). The Qev adaptation weights published at [AustinFu/Qev-2B](https://huggingface.co/AustinFu/Qev-2B), [AustinFu/Qev-4B](https://huggingface.co/AustinFu/Qev-4B) and [AustinFu/Qev-9B](https://huggingface.co/AustinFu/Qev-9B), including their LoRA adapters, decision heads and interaction gates, use the same license. Contributions are accepted under these terms unless explicitly stated otherwise.

Third-party materials retain the terms listed below. The Qev license does not relicense training datasets, external services, or third-party recordings.

## Included and adapted materials

| Material | Source and copyright | License and scope |
|---|---|---|
| Kev conventions and adapted implementation | [Jared Palmer's Kev](https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer | [Apache-2.0, original notice retained](licenses/Kev-Apache-2.0.txt). Qev adapts delimiter and rendering conventions, LoRA targets, and cache forking. Modified source files identify the Qev adaptation. |
| Qwen models and tokenizers | [Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base), [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) and [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base), Copyright 2026 Alibaba Cloud | [Apache-2.0](licenses/Qwen3.5-Apache-2.0.txt). Tokenizers accompany the corresponding Qev checkpoints; base weights are downloaded separately. |
| JevBench public evaluation tasks | [JevBench](https://github.com/fstandhartinger/jevbench), Copyright 2026 Florian Standhartinger and contributors | [MIT](licenses/JevBench-MIT.txt), with [upstream third-party notices](licenses/JevBench-THIRD-PARTY.md). Tasks are downloaded by the optional preparation script. The original license, third-party notice, and per-question provenance accompany the prepared dataset. |
| Crafter visuals in the gameplay recording | [Crafter](https://github.com/danijar/crafter), Copyright 2021 Danijar Hafner | [MIT](licenses/Crafter-MIT.txt). The gameplay recording contains the environment's visual assets; the recording and Qev decision overlay were produced for this project. |

The reviewed Kev version is [557598f](https://github.com/jaredpalmer/kev/tree/557598fced1dada75dfbf36ed144dce309ac6ceb); the evaluation data use [JevBench v1.4.2](https://github.com/fstandhartinger/jevbench/tree/1df665e3956d7aab7fa0208ff6c4f2d8557f9f90). The 231 downloaded public tasks carry MIT in their individual provenance records; upstream notices describe the scope of other JevBench materials and evaluated services.

The README layout takes inspiration from [JevAny](https://github.com/weitianxin/JevAny). Qev's SVG illustrations and chart code are original. The GIF and MP4 demos are this project's recordings of 9B research models. The Crafter research integration used JevAny's environment adapters; this source package does not vendor those adapters or JevAny's bundled recordings.

## Dependencies installed separately

Dependencies are installed from their own distributions and retain their licenses:

- [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE): BSD-3-Clause, plus its bundled third-party notices.
- [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE), [PEFT](https://github.com/huggingface/peft/blob/main/LICENSE), [Accelerate](https://github.com/huggingface/accelerate/blob/main/LICENSE), [Safetensors](https://github.com/huggingface/safetensors/blob/main/LICENSE), [Hugging Face Hub](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE), and optional [Apache Arrow](https://github.com/apache/arrow/blob/main/LICENSE.txt): Apache-2.0.
- Development tools [pytest](https://github.com/pytest-dev/pytest/blob/main/LICENSE) and [Matplotlib](https://matplotlib.org/stable/project/license.html) retain their MIT and Matplotlib/PSF-based license terms, respectively.

Their source code and dependency distributions are not vendored in Qev's Python package. This list identifies direct dependencies; their own packages include notices for transitive components.

## Training data

The [Qev-train dataset](https://huggingface.co/datasets/AustinFu/Qev-train) publishes 2,442 synthetic training examples separately from this code repository. Its 1,534 original alignment and rule-compliance examples use Apache-2.0 where copyright applies. Its 308 Wikipedia-grounded knowledge examples retain CC BY-SA 4.0, with article versions, contributor links and changes recorded in `ATTRIBUTION.jsonl`. Its 600 boundary examples adapt NVIDIA HelpSteer3 Preference contexts and retain [CC BY 4.0](licenses/HelpSteer3-CC-BY-4.0.txt), with source lineage and changes in `HELPSTEER3_ATTRIBUTION.jsonl`. The [dataset license](https://huggingface.co/datasets/AustinFu/Qev-train/blob/main/LICENSE.md) identifies the scope of each component; combining them does not relicense the derived components under Apache-2.0.

Qev-9B v0.2.0 also uses 4,459 Principle judgments from [NVIDIA HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3), by Zhilin Wang and collaborators, under CC BY 4.0. The source revision is `f6d145777bcbde96137596340fab89793acd1031`; those original judgments are not part of the synthetic Qev-train release. No endorsement by NVIDIA or the source authors is implied.

The complete research training mixture and distillation data are not distributed. Other source datasets retain their own terms.

The small JSONL requests in `examples/` were written for Qev and are covered by its Apache-2.0 license. The benchmark result tables and prediction files contain Qev's recorded outputs, not the original benchmark question text.
