<p align="center">
  <img src="assets/banner.svg" alt="Qev — decisions, grounded in Qwen" width="100%">
</p>

<p align="center">
  <a href="pyproject.toml"><img src="assets/badges/python.svg" alt="Python 3.12"></a>
  <a href="docs/model-card.md"><img src="assets/badges/model.svg" alt="Built on Qwen3.5"></a>
  <a href="LICENSE"><img src="assets/badges/license.svg" alt="Apache-2.0"></a>
</p>

<p align="center"><strong>English</strong> | <a href="README.zh-CN.md">简体中文</a> | <a href="https://huggingface.co/AustinFu/Qev-2B">🤗 Qev-2B</a> | <a href="https://huggingface.co/AustinFu/Qev-4B">🤗 Qev-4B</a> | <a href="https://huggingface.co/AustinFu/Qev-9B">🤗 Qev-9B</a> | <a href="https://huggingface.co/datasets/AustinFu/Qev-train">🤗 Training data</a></p>

**Qev fine-tunes Qwen into a decision model.** Give it context, a question, and candidate answers; get a choice and a probability for every option. One model handles **Choice**, **Noul** (yes/no), and **Score** (ordered ratings).

This repository provides the model architecture, training and evaluation code, Python interface, and checkpoint tools. Train on your own examples or load a Qev checkpoint for inference.

| Start here | What you can do |
|---|---|
| **[Get model weights](#model-and-checkpoints)** | Compare Qev-2B, Qev-4B and Qev-9B and find their checkpoint details |
| **[Run a model](#inference)** | Get decisions and option probabilities through Python or JSONL |
| **[Train a model](#training)** | Prepare labelled data, train from Qwen, or fine-tune an existing Qev checkpoint |

<p align="center">
  <img src="assets/evaluation.svg" alt="Qev-2B, Qev-4B, Qev-9B and Qwen3.5-9B-Base compared on seven matched benchmarks." width="100%">
</p>

| Choose a model | Qev-2B | Qev-4B | Qev-9B |
|---|---|---|---|
| Role | Compact model with response distillation | Mid-sized model trained directly from the Qwen base | Largest decision model |
| Get started | [2B model weights](https://huggingface.co/AustinFu/Qev-2B) | [4B model weights](https://huggingface.co/AustinFu/Qev-4B) | [9B model weights](https://huggingface.co/AustinFu/Qev-9B) |

## Model and checkpoints

| Model | Base and architecture | Availability |
|---|---|---|
| **Qev-2B** | Qwen3.5-2B-Base, rank-64 LoRA, two-layer 256-dimensional head | [Hugging Face · Download](https://huggingface.co/AustinFu/Qev-2B) |
| **Qev-4B** | Qwen3.5-4B-Base, rank-64 LoRA, two-layer 256-dimensional head | [Hugging Face · Download](https://huggingface.co/AustinFu/Qev-4B) |
| **Qev-9B** | Qwen3.5-9B-Base, rank-64 LoRA, two-layer 256-dimensional set head | [Hugging Face · Download](https://huggingface.co/AustinFu/Qev-9B) |

Qev-9B v0.2.0 combines general decision data with HelpSteer3 Principle and 600 synthetic boundary questions in the main training set. Additional alignment and document-rule examples are mixed into the second half of training and repeated three times. The download includes the LoRA adapter, decision head, interaction gate, tokenizer and configuration.

Qev-4B starts directly from Qwen3.5-4B-Base and learns a 9B teacher's option probabilities over 44,576 inputs. It uses a single two-epoch training stage and supports 4,096-token paths. [4B training method](docs/training-4b.md).

Qev-2B learns from Qev-9B through probability distillation and programmatic context edits, with an additional loss on representation changes. The adaptation packages download automatically: approximately **284 MiB for 2B**, **524 MiB for 4B** and **690 MiB for 9B**; the loader fetches the corresponding Qwen base separately. See [checkpoint export and loading](docs/checkpoints.md) and the model cards for [2B](docs/model-card-2b.md), [4B](docs/model-card-4b.md) and [9B](docs/model-card.md).

## Installation

Use Python 3.12. From the repository root, create an environment, install a hardware-compatible PyTorch 2.8.0 build inside it, then install Qev:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# Install the PyTorch 2.8.0 build for your hardware in this environment first.
python -m pip install -e .
```

For development and testing, use `python -m pip install -e '.[test]'`. You can first check the complete pipeline on CPU without downloading a model:

```bash
python scripts/smoke.py --out runs/smoke
```

This uses a tiny random Qwen to exercise data preparation, training, resume and inference. See the [training guide](docs/training.md) for environment and device details.

## Inference

### Python API

Load the model once in your process, then submit requests:

```python
from qev import Qev

# Choose "AustinFu/Qev-2B", "AustinFu/Qev-4B" or "AustinFu/Qev-9B".
model = Qev.from_pretrained(
    "AustinFu/Qev-9B", device="cuda"
)
answers = model.predict({
    "state": "I was charged twice. Please help immediately.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Charges and refunds", "shipping": "Delivery problems"},
        }
    },
})
print(answers["department"]["choice"])
print(answers["department"]["probabilities"])
```

Use `noul` for a yes/no question and receive `P(true)`. Use `score` for a rubric and receive its level distribution and expectation. [All three tasks](examples/requests.jsonl) · [Input/output contract](docs/data.md).

### JSONL inference

```bash
python -m qev.predict \
  --checkpoint AustinFu/Qev-9B \
  --input examples/requests.jsonl --out runs/predictions.jsonl \
  --device cuda --weights-dtype checkpoint
```

The default uses shared-prefix caching. Add `--reference` for the execution used in the reported evaluation. Output files must be new; precision and length options are in the [loading guide](docs/checkpoints.md).

## Training

### Prepare data

Training examples use the same `state` and `questions` as inference, with a `label` on each question. Choice labels are candidate IDs, Noul labels are booleans, and Score labels are zero-based level indices. Related records should share a `group_id`.

| Data | Contents | Entry point |
|---|---|---|
| Included examples | Six training and two validation requests, covering all three tasks | [examples/](examples/README.md) |
| **Qev-train** | **2,442 synthetic training examples**: alignment, document rules, world knowledge and controlled boundary questions | [Dataset and synthesis methods](https://huggingface.co/datasets/AustinFu/Qev-train) |
| Your data | Labelled or soft-target JSONL requests | [Data format](docs/data.md) |
| 9B research recipe | 39,605 main and 1,783 late records; the full corpus is not bundled | [Composition and availability](docs/data.md#research-recipe-and-availability) |

```bash
python -m qev.prepare \
  --input examples/train.jsonl --validation examples/dev.jsonl \
  --out data/support
```

Without an explicit validation file, the preparer splits by group deterministically. It checks train/dev overlap in IDs, groups and exact inputs. The small included examples demonstrate the workflow; they cannot establish training gains.

### Supervised fine-tuning

On a CUDA GPU, initialize a new domain-training run from a Qev-9B checkpoint:

```bash
python -m qev.train \
  --config configs/qev-9b-finetune.json \
  --data data/support --out runs/support \
  --init-checkpoint AustinFu/Qev-9B
```

`--init-checkpoint` loads model parameters and starts a fresh optimizer and schedule. `--resume` continues the same run with its original data checks. Omit initialization to start from the Qwen base in the configuration.

The [formal four-GPU recipe](configs/qev-9b.json) uses rank 64, global batch 32, two epochs, and a late-training partition. Single-GPU use, distributed training, resume and full fine-tuning are documented in the [training guide](docs/training.md).

### Train the 4B model

Use [configs/qev-4b.json](configs/qev-4b.json) to train from the Qwen base with teacher-probability cross entropy, or [configs/qev-4b-finetune.json](configs/qev-4b-finetune.json) to adapt `AustinFu/Qev-4B` on labelled data. [Training method and commands](docs/training-4b.md).

### Distill a 2B model

Qev-2B first learns the Qev-9B v0.1.0 teacher's option probabilities with cross entropy. It then learns from programmatic context edits mixed with original-question replay, matching both teacher probabilities and representation changes. See the [2B distillation guide](docs/distillation.md) for data preparation, both training stages and the loss equations.

## Demos

Recorded Snake and Crafter decision replays from the 9B research models showing action selection and candidate probabilities in the environment. Click either animation for the MP4 version. The recordings retain the research name, BranchKev, in their interface.

<table>
  <tr>
    <td width="50%" align="center">
      <a href="assets/demos/snake.mp4"><img src="assets/demos/snake.gif" alt="Snake decision replay, with selected actions and probabilities" width="100%"></a>
      <br><strong>Snake · Sequential action selection</strong>
    </td>
    <td width="50%" align="center">
      <a href="assets/demos/crafter.mp4"><img src="assets/demos/crafter.gif" alt="Crafter decision replay, with goals, actions and probabilities" width="100%"></a>
      <br><strong>Crafter · Survival and crafting</strong>
    </td>
  </tr>
</table>

[Recording details](docs/demos.md).

## How decisions are made

<p align="center">
  <img src="assets/architecture.svg" alt="Qev encodes the context, question and answer options into numeric summaries, then applies four decision-head stages. The diagram explains candidates and vectors e1, e2 and e3." width="100%">
</p>

Qev organizes inputs as **state → question → candidate**. Candidate branches read the shared context and produce their own summaries. The set head combines the question and option summaries, using the same scoring function for variable numbers of options. The implementation provides full causal reference execution, prefix caching, and tree execution with branched DeltaNet state.

[Architecture and equations](docs/architecture.md) · [Interactive Chinese decision-head walkthrough](docs/decision-head.html#set-head)

## Evaluation

**Precision: Qev-9B, Qev-4B and Qev-2B use BF16 backbone computation; Kev-9B uses FP32.** Qev's decision head remains in FP32.

| Benchmark | Jev (reference) | Qev-9B | Kev-9B | Qwen3.5-9B-Base | Qev-4B | Qev-2B | Qwen3.5-2B-Base |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decision development · clean | 84.49 | **87.42** | 87.18 | 77.69 | 86.95 | 85.36 | 65.43 |
| Transfer development · clean | 85.67 | **83.99** | 82.16 | 74.39 | 82.01 | 77.29 | 65.09 |
| MMLU-Pro · 1,000 | 83.50 | **57.40** | 51.10 | 50.40 | 50.30 | 38.70 | 31.20 |
| SemIf · 144 handwritten | 96.53 | **93.06** | 90.97 | 90.28 | 90.28 | 82.64 | 63.89 |
| scienthoon · 873 | 75.26 | 71.02 | **75.49** | 68.84 | 76.75 | 71.94 | 53.84 |
| WANLI · 256 | 75.78 | **71.09** | 70.31 | 67.97 | 73.44 | 67.58 | 50.39 |
| JevBench public · 231 | 85.71 | **83.12** | 75.76 | 75.76 | 82.25 | 74.46 | 63.20 |

Accuracy (%). Bold marks the higher score between Qev-9B and Kev-9B; Jev and the Qwen base are references.

[Six-model benchmark matrix](assets/evaluation-matrix.svg) · [Full results, ablations and settings](docs/evaluation.md) · [Machine-readable metrics](results/benchmarks.json) · [All 231 predictions](results/qev-9b/jevbench-predictions.jsonl)

## Documentation and contributions

[Architecture](docs/architecture.md) · [Training](docs/training.md) · [2B distillation](docs/distillation.md) · [Checkpoints](docs/checkpoints.md) · [Data and outputs](docs/data.md) · [Evaluation](docs/evaluation.md) · [Contributing](CONTRIBUTING.md)

Run `python -m pytest -q` and `python scripts/check_release.py` to check the code and documentation. The [validation record](docs/validation.md) lists the actual checks and GPU skips.

Qev builds on Qwen and adapts delimiter, rendering, LoRA-target and cache-fork conventions from [Jared Palmer's Kev](https://github.com/jaredpalmer/kev). Qev code, adaptation weights, documentation and original illustrations use [Apache-2.0](LICENSE). Attribution and the licenses for Qwen, Kev, JevBench and external dependencies are listed in [third-party notices](THIRD_PARTY_NOTICES.md).

## Acknowledgments

We thank the following projects and author for their work and inspiration:

- **BranchKev**: the research work on candidate-branch encoding, decision heads and training workflows provided the foundation for Qev's standalone release. See [NOTICE](NOTICE) and [code lineage](results/history/source-extraction.json) for the code lineage.
- **[Jev / the TypeSafe team](https://typesafe.ai/)**: thank you for advancing decision models and providing [public API documentation](https://docs.typesafe.ai/introduction) for typed decisions and probability outputs.
- **[Archer Hume — Jev’s Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked/)**: thank you for the independent API experiments and architectural analysis, offering useful perspectives on shared-state computation, question isolation and interactions between candidate answers.
