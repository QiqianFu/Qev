# Training Qev-4B

[中文](training-4b.zh-CN.md) · [Model card](model-card-4b.md) · [General training guide](training.md)

Qev-4B v0.1.0 initializes directly from **Qwen3.5-4B-Base**, with fresh rank-64 LoRA parameters, a 256-dimensional two-layer decision head and a final-layer candidate-interaction gate. It learns a 9B teacher's option probabilities in one training stage. It does not initialize from a previously distilled 4B checkpoint or use representation-response continuation.

## Objective

For each question, the teacher scores exactly the supplied candidates. Its unscaled logits are cached once; training uses

\[
p_T(i\mid x)=\operatorname{softmax}(z_T(x)/1.563437713227029)_i,
\qquad \mathcal L=-\sum_i p_T(i\mid x)\log p_S(i\mid x).
\]

The student temperature is 1. There is no hard-label mixture, response loss or extra temperature-squared multiplier. Labels and targets are removed from teacher requests; input identity includes the state, instructions, task type and candidate IDs, text and order. Missing teacher inputs abort training. [Implementation](../qev/distillation.py).

## Recorded recipe

| Setting | Qev-4B v0.1.0 |
|---|---|
| Base revision | `710fd005d44d55ee27b7ad5147e318e546efdbfe` |
| Adaptation | LoRA rank 64, alpha 128; 256-dimensional head, 4 heads, 2 layers |
| Candidate interaction | Full sibling interaction in the last full-attention layer |
| Input limits | State 4,096; question 512; candidate 256; complete path 4,096 tokens |
| Training inputs | 44,576 single-question records; no hard labels |
| Schedule | Two epochs, 2,786 steps, seed 17 |
| Batch | 8 per GPU × 4 GPUs; global batch 32 |
| Learning rates | LoRA 2e-5; head 1e-4; interaction gate 0.01 |
| Warmup | First 100 steps train only the head; learning-rate warmup 20 steps |
| Computation | BF16 backbone, FP32 head and readout reductions; tree-batched training |
| Sampling | One main pool; no late split, None-option insertion or online text edits |

The input pool combines general decisions, science and reasoning questions, 2,230 Principle judgments, controlled boundaries, web actions and additional rule/reasoning tasks. Records are expanded to one question each and deduplicated by their exact input and candidate order. The 308 world-knowledge questions and 3,253 questions belonging to held-out response-probe parents were excluded before training. All 44,576 remaining inputs were admitted.

The recorded teacher is now available as [Qev-9B v0.3.0](https://huggingface.co/AustinFu/Qev-9B/tree/v0.3.0), trained with additional web and rule tasks at step 2,658. Its trained tensors match the teacher used for this 4B release. The full 4B input pool and cached logits are not distributed. The released 4B weights preserve the trained model; rerunning the method with another teacher or dataset does not reproduce its benchmark numbers.

## Run the method on your data

Install Qev as described in the root README. Prepare labelled requests using the usual data tool; the teacher sees only their inputs:

```bash
python -m qev.prepare --input examples/train.jsonl \
  --validation examples/dev.jsonl --out data/support-4b

python -m qev.teacher logits \
  --teacher AustinFu/Qev-9B@v0.3.0 \
  --data data/support-4b --config configs/qev-4b.json \
  --out data/teacher-logits-4b --device cuda

torchrun --standalone --nproc_per_node=4 -m qev.train \
  --config configs/qev-4b.json --data data/support-4b \
  --out runs/qev-4b
```

This runnable example uses the original 9B teacher and the small example dataset. Use an appropriate teacher and more training data for your task. The teacher's own input limits still apply when collecting logits. Omit `--init-checkpoint` to start from the Qwen base. The cache stays at temperature 1; `training.distillation.temperature` applies the teacher temperature during training. For one GPU, set `batch_size: 1` and `accum: 32` to preserve the global batch.

Prebuilt canonical JSONL data may omit `label` and `target` when `training.distillation.weight` is 1. Its manifest must designate the training split with `role: train`. Mixed teacher/hard-label training still requires original targets. `qev.prepare` remains a labelled-data preparer.

## Fine-tune the released model

For supervised adaptation, use the separate configuration without a teacher cache:

```bash
python -m qev.train --config configs/qev-4b-finetune.json \
  --data data/support-4b --out runs/qev-4b-support \
  --init-checkpoint AustinFu/Qev-4B@v0.1.0
```

This starts a fresh optimizer and schedule. The single-GPU batch settings are an example, not a minimum-memory guarantee. See [checkpoint loading and export](checkpoints.md) for inference, local bases and resume semantics.
