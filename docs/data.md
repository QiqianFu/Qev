# Data and output formats

## User requests and labels

Each JSONL row is one request. `state` may be a string or a JSON value. `questions` maps stable question IDs to task definitions. See [unlabelled requests](../examples/requests.jsonl), [labelled training examples](../examples/train.jsonl), and [validation examples](../examples/dev.jsonl).

| Type | `criteria` | `label` | Output |
|---|---|---|---|
| `choice` | Object mapping candidate IDs to descriptions, or null descriptions | Candidate ID | `choice`, `prediction`, probabilities by ID |
| `noul` | Optional descriptions keyed by `false`/`true` | Boolean | `noul = P(true)`, `prediction`, both probabilities |
| `score` | List of ordered level descriptions | Zero-based level index | Expected `score`, modal `prediction`, probabilities by level |

All outputs also include `type`, `none_candidate_id`, and `predicted_none`. Choice None behavior follows the checkpoint policy; selecting None is retained. Probability is not a separately measured confidence or accuracy guarantee.

Question targets can instead be soft distributions, using a `target` list in candidate order or a mapping by candidate ID. Labels, targets and rationale metadata are never inserted into the model input by the typed adapter. The supplied examples are original, small, English support messages; they are format fixtures rather than the training corpus used for reported results.

## Canonical records

`qev.prepare` writes `qev.record.v1` JSONL and a `manifest.json` describing the dataset splits. Canonical records have `id`, `group_id`, `source`, `state`, and a list of questions with candidate IDs/text, optional label, and target probabilities. Earlier record formats remain supported.

Each split lists its file, purpose and record/question counts. Only `role: train` partitions enter training. The preparer writes `train` and `dev`; JevBench uses `external_evaluation`. Evaluation rejects test-role partitions unless explicitly enabled with `--allow-test`.

The model supports variable candidate counts; the selected configuration admits at most 128 candidates, state 1024 tokens, question 512, candidate 256 and full path 2048. Runtime overrides for longer evaluation inputs are described separately from training limits.

## Research recipe and availability

| Partition | Records / questions | Composition |
|---|---:|---|
| Main | 34,546 / 40,828 | Science-reviewed + AQuA pool (34,238 records), then 308 world-knowledge records |
| Late | 1,783 / 1,783 | 1,419 alignment records and 364 rule-compliance judgments |

The alignment pack contains 249 selected earlier records and 1,170 synthetic records. The 249 selected records also occur in the main partition as intentional replay; late/main are training partitions, not an evaluation split. The 364 rule judgments consist of 182 positive/negative pairs. Late examples start at the final 50% of main-training steps and repeat three times.

**The complete research corpus and its original generation/review pipeline are not included in this source release.** This repository supports training on user data and rerunning public evaluation with the [released Qev-9B weights](https://huggingface.co/AustinFu/Qev-9B). Rebuilding the exact reported training run additionally requires the frozen dataset. Code availability alone is not a claim of full data reproducibility.

Data licensing and attribution are described in [third-party notices](../THIRD_PARTY_NOTICES.md). Any later data release must retain its source-specific terms. In particular, the original Wikipedia-based synthetic world-knowledge pack carried CC BY-SA attribution where applicable. These terms are separate from the code's Apache-2.0 license.

## JevBench

[prepare_jevbench.py](../scripts/prepare_jevbench.py) downloads JevBench v1.4.2. It retains every public task and its gold probabilities, performs explicit Noul label mapping, and exports evaluation-only views. Token-length checks record overflows without dropping tasks. See [evaluation.md](evaluation.md) for commands and the distinction between local accuracy and the official score.

## Distillation data

`qev.teacher logits` prepares teacher probabilities for every training input, including option augmentation. `qev.teacher responses` creates label-blind context edits, teacher probabilities, internal-response matrices and original-question replay. Diagnostic parent groups and identical inputs are kept outside training. See the [two-stage distillation workflow](distillation.md).
