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

## Released synthetic training data

[AustinFu/Qev-train](https://huggingface.co/datasets/AustinFu/Qev-train) provides **2,442 examples** from Qev-9B v0.2.0 training: 1,170 alignment questions, 364 document-rule judgments, 308 Wikipedia-grounded knowledge questions and 600 controlled boundary questions adapted from HelpSteer3 contexts. The dataset card documents generation, blind review, filtering and component-specific licenses in English and Chinese.

```bash
hf download AustinFu/Qev-train --repo-type dataset --local-dir data/qev-train
```

The download includes canonical `train.jsonl`, a Qev `manifest.json`, an equivalent Parquet file, exact statistics and source attribution. Pass `--data data/qev-train` to `qev.train` with a configuration that does not require a separate late partition, such as `configs/qev-9b-finetune.json` or `configs/qev-2b-finetune.json`. No additional format conversion is needed. The current dataset is `v1.1.0`; `v1.0.0` preserves the original 1,842 examples used for the first 9B and released 2B models.

The fine-tuning configurations exempt `synthetic/hs3_preference_boundary/` from online None-option insertion. Retain that exemption with a custom configuration: changing the candidate set changes these reviewed boundary tasks.

All records retain the original input, candidate order and hard labels. `target` contains one-hot label encodings; teacher probability caches, response targets and distillation pairs are not distributed. Metadata records the component, domain, planned language, license and related `group_id`. There is one training split: these examples are already seen by Qev-9B v0.2.0 and are not an independent evaluation set.

## Research recipe and availability

| Partition | Records / questions | Composition |
|---|---:|---|
| Main | 39,605 / 45,887 | Previous 34,546-record pool, plus 4,459 HelpSteer3 Principle judgments and 600 synthetic boundary questions |
| Late | 1,783 / 1,783 | 1,419 alignment records and 364 rule-compliance judgments |

The alignment pack contains 249 selected earlier records and 1,170 synthetic records. The 249 selected records also occur in the main partition as intentional replay; late/main are training partitions, not an evaluation split. The 364 rule judgments consist of 182 positive/negative pairs. Late examples start at the final 50% of main-training steps and repeat three times.

The selected 9B recipe places all 600 boundary questions in the **main** set; they are not added to the late partition. HelpSteer3 Principle supplies judgments about whether responses satisfy stated principles. The boundary tasks instead adapt Preference contexts into new hard-label tasks; original preference rankings are not used. The upstream [NVIDIA HelpSteer3 dataset](https://huggingface.co/datasets/nvidia/HelpSteer3) is pinned at revision `f6d145777bcbde96137596340fab89793acd1031`.

**Qev-train publishes the synthetic subset; the complete mixed research corpus and its original generation/review pipeline are not included in this source release.** This repository supports training on user data and rerunning public evaluation with the released weights. Rebuilding the exact reported training run additionally requires the remaining data and original training settings.

Data licensing and attribution are described in [third-party notices](../THIRD_PARTY_NOTICES.md) and the [dataset license](https://huggingface.co/datasets/AustinFu/Qev-train/blob/main/LICENSE.md). The 1,534 original alignment/rule examples use Apache-2.0 where copyright applies; the 308 Wikipedia-grounded examples retain CC BY-SA 4.0; the 600 HelpSteer3-derived boundary examples retain CC BY 4.0. Both derived components carry per-example attribution.

## Qev-4B training inputs

Qev-4B uses a separate 44,576-question training pool, with one question per record and hard labels removed. It combines general decisions, Principle judgments, controlled boundaries, web actions and additional reasoning/rule tasks. World-knowledge records and held-out response-probe parent questions were excluded; there is no separate late split or online option augmentation.

The teacher supplies the entire target distribution. The [4B training guide](training-4b.md) describes the preparation and objective. This input pool, its research teacher and its cached outputs are not part of Qev-train v1.1.0. Qev-train remains the separately released 2,442-example synthetic subset with its original hard labels.

## JevBench

[prepare_jevbench.py](../scripts/prepare_jevbench.py) downloads JevBench v1.4.2. It retains every public task and its gold probabilities, performs explicit Noul label mapping, and exports evaluation-only views. Token-length checks record overflows without dropping tasks. See [evaluation.md](evaluation.md) for commands and the distinction between local accuracy and the official score.

## Distillation data

`qev.teacher logits` prepares teacher probabilities for every training input, including option augmentation. `qev.teacher responses` creates label-blind context edits, teacher probabilities, internal-response matrices and original-question replay. Diagnostic parent groups and identical inputs are kept outside training. See the [two-stage distillation workflow](distillation.md).
