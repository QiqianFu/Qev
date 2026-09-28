# Qev architecture

Qev-9B fine-tunes Qwen3.5-9B-Base into a model that scores explicit candidates. Its selected release uses rank-64 LoRA, a final-layer gated candidate interaction, and a two-layer set decision head. The configuration is [qev-9b.json](../configs/qev-9b.json).

![Architecture](../assets/architecture.svg)

## 1. Context becomes a candidate tree

A record contains a state and one or more questions. Each question has instructions and candidates. The encoder uses reserved Qwen delimiters and a terminal readout marker:

```text
state
├── question 1 → question readout h_q
│   ├── candidate A → candidate readout e_A
│   ├── candidate B → candidate readout e_B
│   └── candidate C → candidate readout e_C
└── question 2 → ...
```

Ordinary backbone layers keep each candidate on its own causal path. A candidate reads its state, its question, and its own preceding tokens. Sibling order does not change its logical positions. Question readouts only read state and question. User text resembling a reserved delimiter is escaped before tokenization.

The reference execution expands full causal paths. Cached inference computes shared prefixes once, then forks their KV and recurrent state. Tree execution evaluates shared prefixes and branches in one traversal. Qwen3.5 mixes full attention with Gated DeltaNet: branching the recurrent state is necessary for isolation; an attention mask alone does not handle those recurrent layers.

Implementations: [encoding.py](../qev/encoding.py), [cache.py](../qev/cache.py), [tree.py](../qev/tree.py).

## 2. Final-layer candidate interaction

In the selected 32-layer base, layer 31 is a full-attention layer. Each candidate's terminal readout receives an own-branch update and a separate update from sibling-candidate tokens:

```text
own_h   = softmax(scores_h + own_mask) V_h
cross_h = softmax(scores_h + sibling_mask) V_h
mixed_h = own_h + tanh(g_h) × cross_h
```

The gate `g_h` is a learned scalar per backbone query head, initialized to zero. This extra gate is separate from Qwen's native input-dependent output gate. The two softmax operations have separate denominators; the combined coefficients are not another normalized attention distribution. Question readouts have no sibling update.

After mixing, Qwen's output gate, output projection, residual, MLP residual, and final normalization produce the readout vectors. Sibling values are read synchronously at the layer input.

`candidate_interaction` offers three modes:

| Mode | Sibling information |
|---|---|
| `last-full-attention` | All tokens in sibling candidates; selected Qev-9B release |
| `last-readout-cross` | Only sibling terminal readout positions |
| `none` | No extra backbone interaction |

See [model.py](../qev/model.py) and `joint_readouts` in [tree.py](../qev/tree.py).

## 3. The set decision head, step by step

For K candidates, the head receives **one question vector `h_q` and K candidate vectors `e_i`**. These are 4,096-channel summaries. The following operations happen separately for each question; questions do not share an attention matrix in the head.

### Project each summary

```text
q   = W_p LayerNorm(h_q) + b_p + r_q     [256]
c_i = W_p LayerNorm(e_i) + b_p           [256]
X   = stack(q, c_1, ..., c_K)[None]      [1, K+1, 256]
```

Question and candidates share LayerNorm and projection parameters. LayerNorm acts along a vector's channels, not across candidates. The learned question role `r_q` starts at zero. Candidates have no position-index embeddings.

### Let the set exchange information

The selected head applies two pre-norm Transformer encoder layers. Each has four attention heads of width 64, an FFN of `256 → 1024 → 256`, GELU, zero dropout, and residual connections. All K+1 vectors can attend to each other. The head has no RoPE, absolute position encoding, or causal mask.

After the first layer, the question vector can summarize the candidate set. In the second, each candidate can read that updated question as well as the updated candidates. The two layers have separate parameters. There is no extra encoder-wide normalization after them.

### Apply the same scalar scorer to every candidate

Let `q'` and `c_i'` denote the updated vectors:

```text
u_i = concat(c_i', q')                   [512]
z_i = Linear(256, 1)(
        GELU(Linear(512, 256)(
          LayerNorm(512)(u_i))))         [1]
```

The candidate occupies the first 256 channels and the question the second 256. All K rows pass through the same scorer. The final linear layer always outputs **one scalar**; changing K changes the row count. This supports arbitrary candidate descriptions without a fixed output vocabulary of classes.

The complete implementation is `SetDecisionHead` in [model.py](../qev/model.py). An [interactive Chinese walkthrough](decision-head.html#set-head) explains the same computation with matrices and task examples.

### Normalize and interpret

`p_i = softmax(z / T)_i`. The selected checkpoint uses `T = 1`; no fitted calibration is claimed.

| Task | Candidates | Returned interpretation |
|---|---|---|
| Choice | IDs with descriptions | Highest-probability ID and all probabilities |
| Noul | `false`, `true` | `P(true)`, plus the discrete prediction |
| Score | Ordered rubric levels `0..K−1` | `Σ i × p_i`, plus probabilities and the modal level |

Score is a distribution over rubric levels, with its expectation as a convenience output. The default training objective remains cross-entropy against hard or soft candidate targets; optional ordinal losses are implemented separately.

## 4. Symmetry and execution

Keeping IDs and descriptions unchanged while reordering candidates permutes the outputs in ideal arithmetic. Finite-precision reductions can perturb nearly tied scores. Adding, removing, or duplicating an option changes attention and normalization, so existing logits and odds can change.

The backbone runs in BF16. The exported LoRA adapter tensors, decision-head tensors, and joint gate are stored in FP32; the head and joint reductions also execute in FP32. Reference, cached, and tree execution implement the same intended computation, with numerical differences documented and tested. Tree training uses differentiable shared prefixes; it does not reuse detached inference caches.

## 5. What the experiments establish

Architecture motivates an implementation; the current comparisons do not isolate its benefit. The single-seed 2×2 experiment found comparable results with the extra backbone interaction and set-head attention disabled. The selected release retains the evaluated full configuration, and the independent and readout-only variants remain available for research. See [evaluation.md](evaluation.md).
