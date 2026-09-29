# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Gradient-correct leaf-row training and two-level prefix-cache inference.

The cache fork and LoRA module conventions follow Kev (Apache-2.0). Candidate
encoding and the permutation-equivariant readout are new, incompatible layouts.
"""
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .encoding import LAYOUT, leaf_rows
from .cache import FunctionalCache, fork_inference_cache
from .execution import READOUT_PRECISION, fp32_einsum


# "last-full-attention": the final decoder layer (a full-attention layer on
# Qwen3.5) lets each candidate readout token also attend to sibling candidates.
# last-full-attention: at the last full-attention layer each candidate readout also attends (gated, in a separate
# softmax) to every token of its sibling candidates. last-readout-cross: the same, but the sibling keys are only the
# siblings' readout (final) tokens, which already summarize their whole branch.
INTERACTIONS = ("none", "last-full-attention", "last-readout-cross")


class _Deferred(Exception):
    """Carries the residual stream and HF-built kwargs at the joint layer input."""
    def __init__(self, hidden, kwargs):
        super().__init__("deferred joint layer")
        self.hidden = hidden
        self.kwargs = kwargs


def _base_model(backbone):
    return backbone.get_base_model() if hasattr(backbone, "get_base_model") else backbone


def joint_layer_index(backbone):
    base = _base_model(backbone)
    n = len(base.layers)
    types = list(getattr(base.config, "layer_types", None) or ["full_attention"] * n)
    full = [i for i, t in enumerate(types) if t == "full_attention"]
    if not full or full[-1] != n - 1:
        raise NotImplementedError("candidate interaction requires the final decoder layer to be full attention")
    attn = base.layers[-1].self_attn
    heads = base.config.num_attention_heads
    if attn.q_proj.out_features != 2 * heads * attn.head_dim:
        raise NotImplementedError("candidate interaction is implemented for gated (Qwen3.5) attention only")
    return n - 1


@dataclass
class ModelSpec:
    base: str
    revision: str | None = None
    head_dim: int = 256
    head_heads: int = 4
    head_layers: int = 2
    lora_rank: int = 16
    lora_dropout: float = 0.0
    weights_dtype: str = "fp32"
    attention: str = "sdpa"
    rows_per_forward: int = 8
    max_padding_ratio: float = 1.25
    max_padded_tokens: int = 8192
    layout: str = LAYOUT
    choice_none_policy: str = "as-provided"
    candidate_interaction: str = "none"

    def __post_init__(self):
        from .choice_policy import POLICIES
        if self.choice_none_policy not in POLICIES:
            raise ValueError("unknown Choice None policy")
        if self.layout != LAYOUT:
            raise ValueError("incompatible model layout")
        if self.candidate_interaction not in INTERACTIONS:
            raise ValueError("unknown candidate interaction")
        if self.head_dim < 1 or self.head_heads < 1 or self.head_dim % self.head_heads:
            raise ValueError("head_dim must be divisible by head_heads")
        if self.head_layers < 0 or self.lora_rank < 0 or self.rows_per_forward < 1:
            raise ValueError("invalid model dimensions")
        if self.weights_dtype not in {"fp32", "bf16"}:
            raise ValueError("unsupported weights dtype")
        if self.max_padding_ratio < 1 or self.max_padded_tokens < 1:
            raise ValueError("invalid row-packing limits")


class SetDecisionHead(nn.Module):
    """No candidate-index embeddings; only semantic candidate vectors interact."""
    def __init__(self, hidden_size, dim=256, heads=4, layers=2):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.project = nn.Linear(hidden_size, dim)
        self.query_role = nn.Parameter(torch.zeros(dim))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.scorer = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, question, candidates):
        # A single question at a time: there is no K-padding or cross-question mask.
        # Training wraps the whole model in BF16 autocast. Protect the head at
        # its entry point so train, prediction and direct calls use one policy.
        with torch.autocast(question.device.type, enabled=False):
            q = self.project(self.norm(question.float())) + self.query_role
            cs = self.project(self.norm(candidates.float()))
            x = torch.cat([q[None], cs], 0)[None]
            for layer in self.layers:
                x = layer(x)  # bidirectional within this one candidate set
            q, cs = x[0, 0], x[0, 1:]
            return self.scorer(torch.cat([cs, q.expand_as(cs)], -1)).squeeze(-1).float()


def lora_modules(config):
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if "linear_attention" in getattr(config, "layer_types", []):
        targets += ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
    return targets


def load_backbone(spec, device, adapter=None):
    if spec.weights_dtype == "fp32":
        from .execution import configure_fp32
        configure_fp32()
    from transformers import AutoModel
    dtype = torch.bfloat16 if spec.weights_dtype == "bf16" else torch.float32
    loaded = AutoModel.from_pretrained(spec.base, revision=spec.revision, dtype=dtype,
                                       attn_implementation=spec.attention, trust_remote_code=False)
    # Qwen3.5 checkpoints may wrap a text tower and a vision tower. Keep only text.
    backbone = getattr(loaded, "language_model", loaded)
    backbone.requires_grad_(False)
    if adapter:
        from peft import PeftModel
        backbone = PeftModel.from_pretrained(backbone, adapter, is_trainable=True)
    elif spec.lora_rank:
        from peft import LoraConfig, get_peft_model
        backbone = get_peft_model(backbone, LoraConfig(
            task_type="FEATURE_EXTRACTION", r=spec.lora_rank, lora_alpha=2 * spec.lora_rank,
            lora_dropout=spec.lora_dropout, target_modules=lora_modules(backbone.config)))
    return backbone.to(device)


class QevModel(nn.Module):
    def __init__(self, backbone, spec, pad_id):
        super().__init__()
        self.backbone = backbone
        self.spec = spec
        self.pad_id = pad_id
        self.head = SetDecisionHead(backbone.config.hidden_size, spec.head_dim, spec.head_heads, spec.head_layers)
        self.head_only = False
        self.temperature = 1.0
        self.lora_merged = False
        self.head_precision = "fp32"
        self.readout_precision = READOUT_PRECISION
        self.prefix_execution = "leaf-rows"
        self.shared_checkpointing = False
        self.tree_checkpointing = False
        self.head.to(next(backbone.parameters()).device)
        self.joint_layer = None
        if spec.candidate_interaction != "none":
            self.joint_layer = joint_layer_index(backbone)
            # Per-head tanh gate, zero at initialisation: the model then equals
            # the independent-branch model exactly, so existing weights warm-start.
            self.joint_gate = nn.Parameter(torch.zeros(backbone.config.num_attention_heads, device=self.device))

    @property
    def device(self):
        # FSDP CPU offload keeps sharded parameters on the host; compute stays on the GPU.
        return getattr(self, "compute_device", None) or next(self.backbone.parameters()).device

    def _run_rows(self, rows, *, offset=0, cache=None, defer=False):
        """Each row has an independent recurrence and identical prefix offset.

        Real positions do not depend on right-padding. HF's cache-aware mask
        builder handles the [new_length, past_length+new_length] causal alignment.
        """
        width = max(map(len, rows))
        device = self.device
        pinned = device.type == "cuda"
        # Two bulk transfers replace O(rows) tiny host-to-device copies/kernels.
        ids = torch.tensor([list(row) + [self.pad_id] * (width - len(row)) for row in rows],
                           dtype=torch.long, pin_memory=pinned).to(device, non_blocking=pinned)
        lengths = torch.tensor([len(row) for row in rows], dtype=torch.long,
                               pin_memory=pinned).to(device, non_blocking=pinned)
        columns = torch.arange(width, device=device)[None, :]
        valid = columns < lengths[:, None]
        positions = (columns + offset).expand(len(rows), -1).masked_fill(~valid, 0)
        mask = valid.long()
        if cache is not None:
            mask = torch.cat([torch.ones(len(rows), offset, dtype=mask.dtype, device=self.device), mask], -1)
        inputs = dict(input_ids=ids, attention_mask=mask, position_ids=positions,
                      past_key_values=cache, use_cache=cache is not None, return_dict=True)
        if not defer:
            return self.backbone(**inputs)
        # Stop before the joint layer; it needs all sibling candidates at once.
        def stop(module, args, kwargs):
            raise _Deferred(args[0] if args else kwargs["hidden_states"], kwargs)
        # The hook must fire outside torch.utils.checkpoint: raising inside a
        # checkpointed call leaks its saved-tensor hooks into later operations.
        layer = _base_model(self.backbone).layers[self.joint_layer]
        checkpointed = getattr(layer, "gradient_checkpointing", False)
        layer.gradient_checkpointing = False
        handle = layer.register_forward_pre_hook(stop, with_kwargs=True)
        try:
            self.backbone(**inputs)
        except _Deferred as deferred:
            # This exception is internal control flow. Its traceback retains the
            # decoder frames and their large cache forks until cyclic GC runs.
            # Callers only need residuals and RoPE, not the writable child cache.
            deferred.__traceback__ = None
            deferred.kwargs = {"position_embeddings": deferred.kwargs["position_embeddings"]}
            return deferred
        finally:
            handle.remove()
            layer.gradient_checkpointing = checkpointed
        raise RuntimeError("joint layer was not reached")

    def reference_features(self, encoded):
        """Independent full paths; no inference cache, detach, or sibling visibility."""
        if self.joint_layer is not None:
            return self.reference_batch_features([encoded])[0]
        rows, pointers = leaf_rows(encoded)
        qs = [None] * len(encoded.questions)
        cs = [[] for _ in encoded.questions]
        context = torch.no_grad() if self.head_only else nullcontext()
        with context:
            for start in range(0, len(rows), self.spec.rows_per_forward):
                chunk = rows[start:start + self.spec.rows_per_forward]
                hidden = self._run_rows(chunk).last_hidden_state
                for b, (qi, ci, qpos, cpos) in enumerate(pointers[start:start + len(chunk)]):
                    if ci == 0:
                        qs[qi] = hidden[b, qpos].float()
                    cs[qi].append(hidden[b, cpos].float())
        return list(zip(qs, [torch.stack(c) for c in cs]))

    def reference_batch_features(self, records):
        # Every leaf remains a separate causal row. Sorting/pooling these rows
        # across records changes padding work, not visibility or logical positions.
        rows = []
        queries = [[None] * len(r.questions) for r in records]
        candidates = [[[None] * len(q.candidates) for q in r.questions] for r in records]
        for ri, record in enumerate(records):
            paths, pointers = leaf_rows(record)
            rows.extend((path, ri, *pointer) for path, pointer in zip(paths, pointers, strict=True))
        rows.sort(key=lambda x: len(x[0]))
        context = torch.no_grad() if self.head_only else nullcontext()
        with context:
            chunks, chunk, tokens = [], [], 0
            for row in rows:
                width = len(row[0])
                if width > self.spec.max_padded_tokens:
                    raise ValueError("a leaf exceeds max_padded_tokens; increase the execution token budget")
                proposed = (len(chunk) + 1) * width
                if chunk and (len(chunk) >= self.spec.rows_per_forward
                              or proposed > self.spec.max_padded_tokens
                              or proposed > self.spec.max_padding_ratio * (tokens + width)):
                    chunks.append(chunk)
                    chunk, tokens = [], 0
                chunk.append(row)
                tokens += width
            if chunk:
                chunks.append(chunk)
            for chunk in chunks:
                if self.joint_layer is not None:
                    deferred = self._run_rows([row[0] for row in chunk], defer=True)
                    cos, sin = deferred.kwargs["position_embeddings"]
                    for b, (path, ri, qi, ci, qpos, _) in enumerate(chunk):
                        n = len(path)
                        candidates[ri][qi][ci] = (deferred.hidden[b, :n], cos[b, :n], sin[b, :n])
                        queries[ri][qi] = qpos + 1
                    continue
                hidden = self._run_rows([row[0] for row in chunk]).last_hidden_state
                for b, (_, ri, qi, ci, qpos, cpos) in enumerate(chunk):
                    if ci == 0:
                        queries[ri][qi] = hidden[b, qpos].float()
                    candidates[ri][qi][ci] = hidden[b, cpos].float()
            if self.joint_layer is not None:
                # The gate is a backbone-side parameter: frozen with it in head-only warmup.
                return [[self._joint_features(cs, prefix) for prefix, cs in zip(qs, css, strict=True)]
                        for qs, css in zip(queries, candidates, strict=True)]
        return [[(q, torch.stack(cs)) for q, cs in zip(qs, css, strict=True)]
                for qs, css in zip(queries, candidates, strict=True)]

    def forward(self, records):
        self._validate_choice_policy(records)
        if self.training and self.lora_merged:
            raise RuntimeError("merged LoRA model is an inference-only copy")
        if self.prefix_execution == "tree-batched":
            features = self.tree_batch_features(records)
        elif self.prefix_execution == "tree":
            features = [self.tree_features(r) for r in records]
        elif self.prefix_execution == "shared-prefix":
            context = torch.no_grad() if self.head_only else nullcontext()
            with context:
                features = [self.shared_features(r) for r in records]
        else:
            features = self.reference_batch_features(records)
        return [[self.head(q, cs) for q, cs in item] for item in features]

    def _validate_choice_policy(self, records):
        if self.spec.choice_none_policy in ("always", "always-varied"):
            from .choice_policy import none_candidate
            for record in records:
                for q in record.questions:
                    if q.question.type == "choice" and none_candidate(q.question) is None:
                        raise ValueError("checkpoint requires a None candidate; use its saved Encoder policy")

    def shared_features(self, encoded):
        """Gradient-preserving S→Q→C execution; no detached prefix tensors."""
        from torch.utils.checkpoint import checkpoint
        if any(getattr(m, "gradient_checkpointing", False) for m in self.backbone.modules()):
            raise ValueError("shared prefixes require segment checkpointing, not HF layer checkpointing")
        if self.spec.lora_dropout or getattr(self.backbone.config, "attention_dropout", 0):
            raise ValueError("shared-prefix training requires zero dropout")
        if self.joint_layer is not None:
            raise NotImplementedError("shared-prefix execution does not implement candidate interaction")

        def segment(rows, parent, offset=0, copies=1):
            if offset and max(map(len, rows)) == 1:
                raise ValueError("functional caches do not support in-place single-token decode")

            def run():
                # Replay always forks the unchanged parent, never a mutated child.
                cache = parent.fork(copies)
                hidden = self._run_rows(rows, offset=offset, cache=cache).last_hidden_state
                return hidden, cache

            if self.shared_checkpointing and self.training and torch.is_grad_enabled():
                return checkpoint(run, use_reentrant=False, preserve_rng_state=False)
            return run()

        _, cache_s = segment([encoded.state], FunctionalCache(self.backbone.config))
        state_length = len(encoded.state)
        features = []
        for q in encoded.questions:
            hidden, cache_sq = segment([q.prefix], cache_s, state_length)
            hq = hidden[0, len(q.prefix) - 1].float()
            candidates = []
            offset = state_length + len(q.prefix)
            # Keep candidate ordering while enforcing the execution token budget.
            start = 0
            while start < len(q.candidates):
                end = start
                width = 0
                while end < len(q.candidates) and end - start < self.spec.rows_per_forward:
                    proposed_width = max(width, len(q.candidates[end]))
                    if (end - start + 1) * proposed_width > self.spec.max_padded_tokens:
                        break
                    width = proposed_width
                    end += 1
                if end == start:
                    raise ValueError("a candidate exceeds max_padded_tokens")
                chunk = q.candidates[start:end]
                output, _ = segment(chunk, cache_sq, offset, len(chunk))
                candidates.extend(output[b, len(c) - 1].float() for b, c in enumerate(chunk))
                start = end
            features.append((hq, torch.stack(candidates)))
        return features

    def _joint_features(self, rows, prefix_len):
        """Final decoder layer evaluated only at readout tokens.

        rows[i] = (residual stream, cos, sin) of candidate i's full leaf row at
        the joint layer input. All rows share the same S+Q prefix of length
        prefix_len, so every candidate span starts at the same logical position.
        The question readout (prefix_len - 1, taken from row 0 as in the plain
        path) never sees candidates. Each candidate's last token additionally
        attends to every token of the *other* candidates' spans, gated per head
        by tanh(joint_gate); the cross term is order-free, so the layer stays
        permutation-equivariant.
        """
        base = _base_model(self.backbone)
        layer = base.layers[self.joint_layer]
        attn = layer.self_attn
        rope = __import__("sys").modules[type(attn).__module__].apply_rotary_pos_emb
        config = base.config
        heads, kv_heads, d = config.num_attention_heads, config.num_key_value_heads, attn.head_dim
        from torch.nn.utils.rnn import pad_sequence
        k_count = len(rows)
        widths = torch.tensor([r[0].shape[0] for r in rows], device=self.device)
        width = int(widths.max())
        if int(widths.min()) <= prefix_len:
            raise ValueError("every candidate row must extend past its shared prefix")
        h = pad_sequence([r[0] for r in rows], batch_first=True)
        cos = pad_sequence([r[1] for r in rows], batch_first=True)
        sin = pad_sequence([r[2] for r in rows], batch_first=True)
        valid = torch.arange(width, device=self.device)[None, :] < widths[:, None]
        x = layer.input_layernorm(h)
        k = attn.k_norm(attn.k_proj(x).view(k_count, width, kv_heads, d)).transpose(1, 2)
        v = attn.v_proj(x).view(k_count, width, kv_heads, d).transpose(1, 2)
        k, _ = rope(k, k, cos, sin)
        k = k.float().repeat_interleave(heads // kv_heads, 1)  # [K, heads, W, d], HF repeat_kv order
        v = v.float().repeat_interleave(heads // kv_heads, 1)
        # Readout tokens: each candidate's last token, then the question token.
        rows_index = torch.arange(k_count, device=self.device)
        last = widths - 1
        index = torch.cat([last, last.new_tensor([prefix_len - 1])])
        source = torch.cat([rows_index, rows_index.new_zeros(1)])
        q, gate = attn.q_proj(x[source, index]).view(k_count + 1, heads, 2 * d).chunk(2, -1)
        q = attn.q_norm(q)[:, :, None]  # [K+1, heads, 1, d]
        q, _ = rope(q, q, cos[source, index][:, None], sin[source, index][:, None])
        q = q[:, :, 0].float()
        # Own branch: causal keys of the reading row (all valid tokens up to the readout).
        own_keys = valid[source] & (torch.arange(width, device=self.device)[None, :] <= index[:, None])
        scores = fp32_einsum("bhd,bhwd->bhw", q, k[source]) * attn.scaling
        scores = scores.masked_fill(~own_keys[:, None, :], float("-inf"))
        own = fp32_einsum("bhw,bhwd->bhd", scores.softmax(-1), v[source])
        # Cross: candidate readouts attend to all tokens of sibling candidate spans (or only their readouts).
        span = valid[:, prefix_len:]  # [K, M]
        if self.spec.candidate_interaction == "last-readout-cross":
            span = torch.arange(width - prefix_len, device=self.device)[None, :] == (last - prefix_len)[:, None]
        ks, vs = k[:, :, prefix_len:], v[:, :, prefix_len:]
        cross_scores = fp32_einsum("bhd,shmd->bhsm", q[:k_count], ks) * attn.scaling
        allowed = span[None, :, :] & ~torch.eye(k_count, dtype=torch.bool, device=self.device)[:, :, None]
        cross_scores = cross_scores.masked_fill(~allowed[:, None], float("-inf")).flatten(2)
        has_keys = allowed.flatten(1).any(-1)  # a lone candidate has no siblings
        weights = torch.where(has_keys[:, None, None], cross_scores, 0.0).softmax(-1)
        weights = weights * has_keys[:, None, None]
        cross = fp32_einsum("bhn,bhnd->bhd", weights, vs.permute(1, 0, 2, 3).flatten(1, 2)[None].expand(k_count, -1, -1, -1))
        cross = torch.cat([cross, cross.new_zeros(1, heads, d)])
        mix = torch.cat([torch.tanh(self.joint_gate.float()).expand(k_count, -1), self.joint_gate.new_zeros(1, heads).float()])
        mixed = (own + mix[:, :, None] * cross).reshape(k_count + 1, heads * d).to(x.dtype)
        out = attn.o_proj(mixed * torch.sigmoid(gate.reshape(k_count + 1, heads * d)))
        hidden = h[source, index] + out
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        hidden = base.norm(hidden).float()
        return hidden[k_count], hidden[:k_count]

    def prepare_inference(self, *, merge_lora=False):
        self.eval()
        if merge_lora and hasattr(self.backbone, "merge_and_unload"):
            self.backbone = self.backbone.merge_and_unload(safe_merge=True)
            self.lora_merged = True
        return self

    def _probabilities(self, features):
        with torch.autocast(self.device.type, enabled=False):
            logits = [self.head(q, cs) for q, cs in features]
        return [torch.softmax(z.float() / self.temperature, -1) for z in logits]

    @torch.no_grad()
    def predict_batch(self, records, *, tree=False):
        """Pool independent leaf rows or trees, preserving record order."""
        self._validate_choice_policy(records)
        if self.training:
            raise RuntimeError("call eval() before prediction")
        features = self.tree_batch_features(records) if tree else self.reference_batch_features(records)
        return [self._probabilities(item) for item in features]

    def fork_cache(self, cache, copies=1):
        return fork_inference_cache(cache, copies)

    def _candidate_chunks(self, candidates):
        """Preserve candidate order while bounding newly computed padded tokens."""
        start = 0
        while start < len(candidates):
            end, width = start, 0
            while end < len(candidates) and end - start < self.spec.rows_per_forward:
                proposed_width = max(width, len(candidates[end]))
                if (end - start + 1) * proposed_width > self.spec.max_padded_tokens:
                    break
                width = proposed_width
                end += 1
            if end == start:
                raise ValueError("a candidate exceeds max_padded_tokens")
            yield candidates[start:end]
            start = end

    def _cached_joint_features(self, rows, cache, question):
        """Joint final layer from candidate-only residuals and shared prefix KV.

        The ordinary causal prefix forward already computed the question readout
        and final-layer prefix KV. Prefix scores are broadcast across candidates;
        neither prefix projections nor prefix hidden states are recomputed here.
        Candidate branches remain independent until this final readout.
        """
        from torch.nn.utils.rnn import pad_sequence
        base = _base_model(self.backbone)
        layer = base.layers[self.joint_layer]
        attn = layer.self_attn
        rope = __import__("sys").modules[type(attn).__module__].apply_rotary_pos_emb
        heads, kv_heads, d = base.config.num_attention_heads, base.config.num_key_value_heads, attn.head_dim
        count = len(rows)
        lengths = [r[0].shape[0] for r in rows]
        width = max(lengths)
        widths = torch.tensor(lengths, device=self.device)
        h, cos, sin = (pad_sequence([r[i] for r in rows], batch_first=True) for i in range(3))
        valid = torch.arange(width, device=self.device)[None, :] < widths[:, None]
        x = layer.input_layernorm(h)
        k = attn.k_norm(attn.k_proj(x).view(count, width, kv_heads, d)).transpose(1, 2)
        v = attn.v_proj(x).view(count, width, kv_heads, d).transpose(1, 2)
        k, _ = rope(k, k, cos, sin)
        k = k.float().repeat_interleave(heads // kv_heads, 1)
        v = v.float().repeat_interleave(heads // kv_heads, 1)
        parent = cache.layers[self.joint_layer]
        pk = parent.keys[0].float().repeat_interleave(heads // kv_heads, 0)
        pv = parent.values[0].float().repeat_interleave(heads // kv_heads, 0)
        ri, last = torch.arange(count, device=self.device), widths - 1
        q, gate = attn.q_proj(x[ri, last]).view(count, heads, 2 * d).chunk(2, -1)
        q = attn.q_norm(q)[:, :, None]
        q, _ = rope(q, q, cos[ri, last][:, None], sin[ri, last][:, None])
        q = q[:, :, 0].float()
        prefix_scores = fp32_einsum("bhd,hpd->bhp", q, pk) * attn.scaling
        own_scores = fp32_einsum("bhd,bhwd->bhw", q, k) * attn.scaling
        own_scores = own_scores.masked_fill(~valid[:, None, :], float("-inf"))
        weights = torch.cat([prefix_scores, own_scores], -1).softmax(-1)
        prefix_len = pk.shape[-2]
        own = (fp32_einsum("bhp,hpd->bhd", weights[:, :, :prefix_len], pv)
               + fp32_einsum("bhw,bhwd->bhd", weights[:, :, prefix_len:], v))
        # Same sibling-only attention and tanh gate as the reference final layer.
        scores = fp32_einsum("bhd,shwd->bhsw", q, k) * attn.scaling
        keys = valid
        if self.spec.candidate_interaction == "last-readout-cross":
            keys = torch.arange(width, device=self.device)[None, :] == last[:, None]
        allowed = keys[None] & ~torch.eye(count, dtype=torch.bool, device=self.device)[:, :, None]
        scores = scores.masked_fill(~allowed[:, None], float("-inf")).flatten(2)
        has_keys = allowed.flatten(1).any(-1)
        weights = torch.where(has_keys[:, None, None], scores, 0.0).softmax(-1)
        weights = weights * has_keys[:, None, None]
        cross = fp32_einsum("bhn,hnd->bhd", weights, v.permute(1, 0, 2, 3).flatten(1, 2))
        mixed = (own + torch.tanh(self.joint_gate.float())[None, :, None] * cross).reshape(count, heads * d).to(x.dtype)
        out = attn.o_proj(mixed * torch.sigmoid(gate.reshape(count, heads * d)))
        hidden = h[ri, last] + out
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return question, base.norm(hidden).float()

    @torch.no_grad()
    def cached_features(self, encoded):
        """Compute S once per record and S+Q once per question, including joint models."""
        if self.training:
            raise RuntimeError("inference cache cannot be used for training")
        from transformers import DynamicCache
        if not encoded.questions:
            return []
        ls = len(encoded.state)
        single = len(encoded.questions) == 1
        if not single:
            cache_s = DynamicCache(config=self.backbone.config)
            self._run_rows([encoded.state], cache=cache_s)
        features = []
        for q in encoded.questions:
            if single:
                # No other question needs S alone: avoid a separate forward/fork.
                cache_sq = DynamicCache(config=self.backbone.config)
                qout = self._run_rows([encoded.state + q.prefix], cache=cache_sq)
            else:
                cache_sq = self.fork_cache(cache_s)
                qout = self._run_rows([q.prefix], offset=ls, cache=cache_sq)
            hq = qout.last_hidden_state[0, -1].float()
            p = ls + len(q.prefix)
            cs = []
            for chunk in self._candidate_chunks(q.candidates):
                fork = self.fork_cache(cache_sq, len(chunk))
                if self.joint_layer is None:
                    hidden = self._run_rows(chunk, offset=p, cache=fork).last_hidden_state
                    cs.extend(hidden[b, len(c) - 1].float() for b, c in enumerate(chunk))
                else:
                    deferred = self._run_rows(chunk, offset=p, cache=fork, defer=True)
                    cos, sin = deferred.kwargs["position_embeddings"]
                    cs.extend((deferred.hidden[b, :len(c)], cos[b, :len(c)], sin[b, :len(c)])
                              for b, c in enumerate(chunk))
            features.append((hq, torch.stack(cs)) if self.joint_layer is None else
                            self._cached_joint_features(cs, cache_sq, hq))
        return features

    def tree_features(self, encoded):
        from .tree import tree_features
        context = nullcontext() if self.training and not self.head_only else torch.no_grad()
        with context:
            return tree_features(self, encoded)

    def tree_batch_features(self, records):
        from .tree import tree_batch_features
        context = nullcontext() if self.training and not self.head_only else torch.no_grad()
        with context:
            return tree_batch_features(self, records)

    @torch.no_grad()
    def predict(self, encoded, *, cached=True, tree=False):
        self._validate_choice_policy([encoded])
        if self.training:
            raise RuntimeError("call eval() before prediction")
        if tree and not cached:
            raise ValueError("tree and reference execution are mutually exclusive")
        features = (self.tree_features(encoded) if tree else
                    self.cached_features(encoded) if cached else self.reference_features(encoded))
        return self._probabilities(features)


def score_order(question):
    """Score levels come from integer candidate IDs, never from list position."""
    try:
        levels = [int(c.id) for c in question.candidates]
    except ValueError:
        raise ValueError(f"score question {question.id} needs integer level IDs") from None
    if len(set(levels)) != len(levels):
        raise ValueError(f"score question {question.id} repeats a level")
    return sorted(range(len(levels)), key=levels.__getitem__)


def hl_gauss_target(target, order, sigma):
    """HL-Gauss smoothing of a Score target over its ordered levels.

    Each level is a unit-width bin at its rank; the mass at rank j spreads as a
    Gaussian N(j, sigma^2) integrated over every bin, renormalised to the
    available levels (Farebrother et al., 2024). Soft targets are smoothed
    per level, so the result is target @ kernel in rank space.
    """
    k = len(order)
    ranks = torch.arange(k, dtype=torch.float64, device=target.device)
    normal = torch.distributions.Normal(torch.tensor(0.0, dtype=torch.float64, device=target.device), sigma)
    edges = normal.cdf((ranks[None, :] + 0.5 - ranks[:, None])) - normal.cdf((ranks[None, :] - 0.5 - ranks[:, None]))
    kernel = edges / edges.sum(-1, keepdim=True)  # row j: where mass at rank j lands
    order = torch.as_tensor(order, device=target.device)
    smoothed = torch.empty_like(target)
    smoothed[order] = (target[order].double() @ kernel).to(target.dtype)
    return smoothed


def record_loss(logits, encoded, ord_w=0.0, score_hl_sigma=0.0):
    """Mean across questions; callers average records, never candidate rows.

    With ord_w > 0, Score questions add ord_w times the normalised ranked
    probability score, mean((CDF_p - CDF_target)^2) over the first K-1 levels.
    With score_hl_sigma > 0, Score questions use cross-entropy against the
    HL-Gauss-smoothed target instead of the given target (in level units).
    """
    losses = []
    for z, q in zip(logits, encoded.questions, strict=True):
        if q.question.target is None:
            raise ValueError(f"missing training target: {encoded.record.id}/{q.question.id}")
        target = torch.tensor(q.question.target, dtype=torch.float32, device=z.device)
        if score_hl_sigma and q.question.type == "score" and len(target) > 1:
            target = hl_gauss_target(target, score_order(q.question), score_hl_sigma)
        log_p = F.log_softmax(z.float(), -1)
        loss = -(target * log_p).sum()
        if ord_w and q.question.type == "score" and len(target) > 1:
            order = torch.tensor(score_order(q.question), device=z.device)
            cdf = log_p.exp()[order].cumsum(-1)[:-1]
            loss = loss + ord_w * (cdf - target[order].cumsum(-1)[:-1]).square().mean()
        losses.append(loss)
    return torch.stack(losses).mean()
