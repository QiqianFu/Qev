# Training Qev-2B with a Qev-9B teacher

Qev-2B uses the same decision architecture as Qev-9B with a smaller Qwen base. Its training has two stages: learn the teacher's option probabilities with cross entropy, then continue with programmatic context edits, original-question replay and a loss on how the teacher's internal representations change.

The selected model has no candidate preview. It reaches 38.70% on MMLU-Pro and 172/231 on public JevBench. The [evaluation table](evaluation.md) compares both model sizes on identical question subsets. [中文说明](distillation.zh-CN.md).

## 1 Learn teacher probabilities

The first stage starts from Qwen3.5-2B-Base and trains a rank-64 LoRA adapter, the two-layer 256-dimensional decision head and the candidate-interaction gate. The teacher is Qev-9B.

For each input and candidate set, cache the teacher's full distribution. The student minimizes cross entropy against that distribution. The published student uses teacher weight 1: original labels are not an additional loss term. The general trainer also supports mixing original targets with teacher probabilities by reducing `training.distillation.weight`.

The teacher cache is built for the actual student inputs, including any added or removed “None of the above” options. Teacher inference sees the context, question and options, with labels and training targets removed. A missing teacher entry fails explicitly rather than falling back to another target.

```bash
python -m qev.teacher logits \
  --teacher AustinFu/Qev-9B \
  --data data/decisions --config configs/qev-2b-distill.json \
  --out data/teacher-logits --device cuda

torchrun --standalone --nproc_per_node=2 -m qev.train \
  --config configs/qev-2b-distill.json \
  --data data/decisions --out runs/qev-2b-probabilities
```

The supplied configuration reproduces the training schedule used for the selected model. It expects `train` and `late_train` partitions. For your own dataset prepared by `qev.prepare`, remove `late_split`, `late_fraction` and `late_repeats` when you have only a `train` partition. Set `training.distillation.cache` to the directory produced by the first command.

The original run used 34,546 main records, 1,783 additional records introduced in the second half and repeated three times, two epochs and global batch 32. The complete research corpus is not included. A dataset with different contents or size will have different training steps and results.

## Prepare context edits and teacher responses

Programmatic edits change a number, reverse a comparison, or remove an observation sentence. The question and candidate descriptions stay fixed. Edit selection is deterministic for a given seed and does not inspect labels. These proposals can be noisy; teacher confidence is a selection heuristic, not a guarantee that the edit is semantically valid.

For the selected recipe, both endpoints must have teacher top-option probability above 0.8. Pairs are split by parent group; overlapping inputs are removed from training. Original-question replay uses unedited training questions outside the diagnostic groups. The world-knowledge subset is excluded in this continuation recipe.

```bash
python -m qev.teacher responses \
  --teacher AustinFu/Qev-9B \
  --data data/decisions --config configs/qev-2b-distill.json \
  --out data/teacher-responses --device cuda
```

This writes `replay.jsonl`, `pairs.jsonl`, `probe.jsonl` and a dataset description. Probabilities and representation targets come from the same teacher forward pass. The original continuation used 37,835 replay questions and 15,806 training pairs. The diagnostic pairs never enter the optimizer or the response normalization scale.

The preparation command builds new targets for your supplied data with the same edit operators and confidence rule. Exact repetition of the published run additionally requires its original selected training data and teacher targets, which are not bundled.

## 2 Continue with probability and representation losses

For the original and edited versions of a question, collect the decision head's question vector followed by its candidate vectors. Normalize each vector independently. If these matrices are $H$ and $H'$, their signed response is:

$$R=(H'-H)H^\top.$$

The response describes changes relative to the original question and options. Teacher and student compute their own matrices; their hidden neurons do not need matching coordinates.

With $B=32$ input examples per optimizer step, the objective is:

$$L=\frac{1}{B}\left[\sum_x \mathrm{CE}(\mathrm{softmax}(z_T(x)/T_T),\mathrm{softmax}(z_S(x)))+\frac{2\lambda}{s}\sum_{(x,x')}\mathrm{MSE}(R_S,R_T)\right].$$

Each pair contains two input examples, hence the factor 2. The scale $s$ is the mean squared teacher response over training pairs, with a small floor for degenerate data. For the selected training data it was approximately 0.00671145. Student temperature is 1, teacher temperature is 1.563437713227029, and $\lambda=0.1$.

```bash
python -m qev.distill \
  --checkpoint runs/qev-2b-probabilities/step-002327 \
  --data data/teacher-responses --config configs/qev-2b-response.json \
  --out runs/qev-2b-response --device cuda
```

Use the actual final checkpoint from your first stage. The selected configuration runs on one GPU for 800 steps. Each step mixes 24 original questions and four original/edited pairs. The two pools cycle through separately shuffled orders. Microbatches hold up to 16 input examples and keep the two members of a pair together.

| Setting | Selected recipe |
|---|---|
| LoRA / head / gate learning rates | 0.00001 / 0.00005 / 0.005 |
| Warmup | 20 steps |
| Schedule | Cosine decay to 10% of the initial rate |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Seed | 17 |
| Saved continuation steps | 400 and 800 |

Resume an interrupted continuation by adding `--resume runs/qev-2b-response/step-000400` to the same command. Model parameters, optimizer state, random state and sampling position are restored. The data and configuration must remain unchanged.

## Export and use the model

```bash
python -m qev.export \
  --checkpoint runs/qev-2b-response/step-000800 \
  --out checkpoints/qev-2b

python -m qev.predict --checkpoint checkpoints/qev-2b \
  --input examples/requests.jsonl --out runs/qev-2b-predictions.jsonl --device cuda
```

The exported model uses the standard Qev interface and does not need the teacher at inference time. [Model card](model-card-2b.md) · [Data format](data.md) · [Licensing](../THIRD_PARTY_NOTICES.md).
