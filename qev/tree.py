# Adapted for Qev in 2026; see NOTICE and provenance.json.
"""Layer-wise Qwen3.5 trees: packed projections and differentiable branch states.

An attention mask alone cannot isolate a Gated DeltaNet. Its convolution uses
the node's ancestor history, and its recurrence starts from the parent's final
state. Only these stateful operations are segmented; expensive linear maps and
MLPs process every unique token together. No weights or checkpoint layout change.
"""
from dataclasses import dataclass
import sys

import torch
from torch.nn import functional as F
from .execution import fp32_einsum


@dataclass(frozen=True)
class Node:
    start: int
    end: int
    parent: int
    depth: int
    path: tuple[int, ...]


class TreeLayout:
    def __init__(self, encoded, device, max_tokens):
        self._build([encoded], device, max_tokens)

    @classmethod
    def from_records(cls, records, device, max_tokens):
        layout = cls.__new__(cls)
        layout._build(records, device, max_tokens)
        return layout

    def _build(self, records, device, max_tokens):
        count = sum(unique_tokens(r) for r in records)
        if count > max_tokens:
            raise ValueError(f"packed tree has {count} unique tokens, exceeding max_padded_tokens={max_tokens}")
        ids, positions, owners, question_ids, candidate_ids = [], [], [], [], []
        self.nodes, self.readouts = [], []
        self.record_questions = [len(r.questions) for r in records]
        self.record_spans = []
        self.batched = len(records) > 1

        def add(tokens, parent, qi=-1, ci=-1):
            start, index = len(ids), len(self.nodes)
            path = self.nodes[parent].path if parent >= 0 else ()
            ids.extend(tokens)
            positions.extend(range(len(path), len(path) + len(tokens)))
            owners.extend([index] * len(tokens))
            question_ids.extend([qi] * len(tokens))
            candidate_ids.extend([ci] * len(tokens))
            self.nodes.append(Node(start, len(ids), parent,
                                   self.nodes[parent].depth + 1 if parent >= 0 else 0,
                                   path + tuple(range(start, len(ids)))))
            return index

        qi = 0
        for encoded in records:
            if not encoded.questions:
                continue
            start = len(ids)
            # Independent roots reset positions, convolution and recurrence.
            single = len(encoded.questions) == 1
            root = add(encoded.state + encoded.questions[0].prefix if single else encoded.state, -1)
            for q in encoded.questions:
                parent = root if single else add(q.prefix, root, qi)
                readouts = [self.nodes[parent].end - 1]
                for ci, candidate in enumerate(q.candidates):
                    node = add(candidate, parent, qi, ci)
                    readouts.append(self.nodes[node].end - 1)
                self.readouts.append(readouts)
                qi += 1  # Globally unique: sibling interaction cannot cross records.
            self.record_spans.append((start, len(ids)))

        def tensor(values):
            return torch.tensor(values, dtype=torch.long, device=device)

        self.ids, self.positions = tensor(ids)[None], tensor(positions)[None]
        self.owners = tensor(owners)
        self.question_ids, self.candidate_ids = tensor(question_ids), tensor(candidate_ids)
        ancestors = torch.zeros(len(self.nodes), len(self.nodes), dtype=torch.bool, device=device)
        # Build once on the CPU, then transfer; no per-edge CUDA launches.
        edges = []
        for i, node in enumerate(self.nodes):
            p = node.parent
            while p >= 0:
                edges.append((i, p))
                p = self.nodes[p].parent
        if edges:
            edge = tensor(edges)
            ancestors[edge[:, 0], edge[:, 1]] = True
        self.ancestors = ancestors
        self.indices = torch.arange(count, device=device)
        self._allow = None
        self.read_indices = tensor([i for group in self.readouts for i in group])
        self._groups = {}
        self._attention_groups = {}

    def allow_between(self, query, key):
        qo, ko = self.owners[query], self.owners[key]
        return (self.ancestors[qo[..., :, None], ko[..., None, :]]
                | ((qo[..., :, None] == ko[..., None, :])
                   & (query[..., :, None] >= key[..., None, :])))

    @property
    def allow(self):
        if self._allow is None:
            self._allow = self.allow_between(self.indices, self.indices)
        return self._allow

    def attention_groups(self, spec, dtype):
        """Pad only attention, within length buckets; projections stay packed."""
        key = spec.max_padding_ratio, spec.max_padded_tokens, spec.rows_per_forward, dtype
        if key in self._attention_groups:
            return self._attention_groups[key]
        spans = sorted(self.record_spans, key=lambda r: r[1] - r[0])
        chunks, chunk, tokens = [], [], 0
        for span in spans:
            length = span[1] - span[0]
            padded = (len(chunk) + 1) * length
            if chunk and (len(chunk) >= spec.rows_per_forward or padded > spec.max_padded_tokens
                          or padded > spec.max_padding_ratio * (tokens + length)):
                chunks.append(chunk)
                chunk, tokens = [], 0
            chunk.append(span)
            tokens += length
        if chunk:
            chunks.append(chunk)
        groups, ordered = [], []
        device, total = self.ids.device, self.ids.numel()
        for chunk in chunks:
            width = max(b - a for a, b in chunk)
            gather, unpad = [], []
            for b, (start, end) in enumerate(chunk):
                gather.append(list(range(start, end)) + [total] * (width - (end - start)))
                unpad.extend(range(b * width, b * width + end - start))
                ordered.extend(range(start, end))
            gather = torch.tensor(gather, device=device)
            valid = gather != total
            safe = gather.clamp_max(total - 1)
            allow = self.allow_between(safe, safe) & valid[:, :, None] & valid[:, None, :]
            # Finite discarded padding outputs also avoid NaN gradients in eager attention.
            allow |= ~valid[:, :, None] & torch.eye(width, device=device, dtype=torch.bool)[None]
            mask = torch.zeros(allow.shape, device=device, dtype=dtype).masked_fill_(~allow, -torch.inf)
            groups.append((gather, torch.tensor(unpad, device=device), mask[:, None]))
        inverse = [0] * total
        for i, physical in enumerate(ordered):
            inverse[physical] = i
        result = groups, torch.tensor(inverse, device=device)
        self._attention_groups[key] = result
        return result

    def groups(self, kernel, max_rows, max_tokens):
        """Precompute gathers; padded parent steps use exactly zero gate and beta."""
        key = kernel, max_rows, max_tokens
        if key in self._groups:
            return self._groups[key]
        parents = {n.parent for n in self.nodes}
        buckets = {}
        for i, n in enumerate(self.nodes):
            # Batched trees may pad parents: beta=0 and g=0 make padding an
            # identity recurrence, including gradients to the prefix state.
            bucket = n.depth, n.end - n.start if i in parents and not self.batched else -1
            buckets.setdefault(bucket, []).append(i)
        groups = []
        total, device = self.ids.numel(), self.ids.device
        for _, nodes in sorted(buckets.items()):
            start = 0
            while start < len(nodes):
                stop, width = start, 0
                while stop < len(nodes) and stop - start < max_rows:
                    n = self.nodes[nodes[stop]]
                    proposed = max(width, n.end - n.start)
                    if (stop - start + 1) * proposed > max_tokens:
                        break
                    width, stop = proposed, stop + 1
                if stop == start:
                    raise ValueError("tree segment exceeds execution token budget")
                group = nodes[start:stop]
                gathers, history = [], []
                for i in group:
                    n = self.nodes[i]
                    gathers.append(list(range(n.start, n.end)) + [total] * (width - (n.end - n.start)))
                    path = self.nodes[n.parent].path if n.parent >= 0 else ()
                    past = list(path[-kernel:])
                    history.append([total] * (kernel - len(past)) + past)
                groups.append((group, torch.tensor(gathers, device=device),
                               torch.tensor(history, device=device)))
                start = stop
        self._groups[key] = groups
        return groups


def unique_tokens(encoded):
    if not encoded.questions:
        return 0
    return len(encoded.state) + sum(len(q.prefix) + sum(map(len, q.candidates)) for q in encoded.questions)


def full_attention_tree(attn, hidden, rope_values, layout, spec):
    """Project all records together; attention remains a batch of independent trees."""
    impl = sys.modules[type(attn).__module__]
    n, d = hidden.shape[1], attn.head_dim
    q, gate = attn.q_proj(hidden).view(1, n, -1, 2 * d).chunk(2, -1)
    gate = gate.reshape(1, n, -1)
    q = attn.q_norm(q).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(hidden).view(1, n, -1, d)).transpose(1, 2)
    v = attn.v_proj(hidden).view(1, n, -1, d).transpose(1, 2)
    q, k = impl.apply_rotary_pos_emb(q, k, *rope_values)

    def with_padding(t):
        t = t[0].transpose(0, 1)
        return torch.cat([t, t.new_zeros(1, *t.shape[1:])])

    q, k, v = map(with_padding, (q, k, v))
    backend = impl.ALL_ATTENTION_FUNCTIONS.get_interface(attn.config._attn_implementation,
                                                         impl.eager_attention_forward)
    groups, inverse = layout.attention_groups(spec, hidden.dtype)
    outputs = []
    for gather, unpad, mask in groups:
        out, _ = backend(attn, q[gather].transpose(1, 2), k[gather].transpose(1, 2),
                         v[gather].transpose(1, 2), mask, dropout=0.0,
                         scaling=attn.scaling, is_causal=False)
        outputs.append(out.reshape(-1, out.shape[-2], d).index_select(0, unpad))
    out = torch.cat(outputs).index_select(0, inverse).reshape(1, n, -1)
    return attn.o_proj(out * gate.sigmoid())


def linear_tree(module, hidden, layout, spec):
    """Qwen3.5 mixer with one projection per unique token and forked recurrence."""
    impl = sys.modules[type(module).__module__]
    total = hidden.shape[1]
    projected = module.in_proj_qkv(hidden)[0]
    z = module.in_proj_z(hidden).reshape(-1, module.head_v_dim)
    beta = module.in_proj_b(hidden)[0].sigmoid()
    g = -module.A_log.float().exp() * F.softplus(module.in_proj_a(hidden)[0].float() + module.dt_bias)

    def with_zero(x):
        return torch.cat([x, x.new_zeros(1, *x.shape[1:])])

    projected, beta, g = map(with_zero, (projected, beta, g))
    states, outputs = {}, [None] * len(layout.nodes)
    parents = {n.parent for n in layout.nodes}
    for nodes, indices, history in layout.groups(module.conv_kernel_size, spec.rows_per_forward,
                                                 spec.max_padded_tokens):
        batch, width = indices.shape
        conv_input = torch.cat([projected[history], projected[indices]], 1).transpose(1, 2).contiguous()
        mixed = impl.causal_conv1d_fn(conv_input, module.conv1d.weight.squeeze(1), module.conv1d.bias,
                                     activation=module.activation)[..., -width:].transpose(1, 2)
        q, k, v = torch.split(mixed, [module.key_dim, module.key_dim, module.value_dim], -1)
        q, k = (x.reshape(batch, width, module.num_k_heads, module.head_k_dim) for x in (q, k))
        v = v.reshape(batch, width, module.num_v_heads, module.head_v_dim)
        copies = module.num_v_heads // module.num_k_heads
        if copies > 1:
            q, k = q.repeat_interleave(copies, 2), k.repeat_interleave(copies, 2)
        initial = (torch.cat([states[layout.nodes[i].parent] for i in nodes], 0)
                   if layout.nodes[nodes[0]].parent >= 0 else None)
        keep_state = any(i in parents for i in nodes)
        # Fused recurrent is an inference kernel; training needs the chunk
        # kernel's gradients with respect to both initial and final states.
        op = (impl.torch_recurrent_gated_delta_rule if not torch.is_grad_enabled() and initial is not None and width == 1
              else impl.torch_chunk_gated_delta_rule)
        out, final = op(q, k, v, g=g[indices], beta=beta[indices], initial_state=initial,
                        output_final_state=keep_state, use_qk_l2norm_in_kernel=True)
        for row, i in enumerate(nodes):
            n = layout.nodes[i]
            outputs[i] = out[row, :n.end - n.start]
            if i in parents:
                states[i] = final[row:row + 1]
    core = torch.cat(outputs).reshape(-1, module.head_v_dim)
    core = module.norm(core, z).reshape(1, total, -1)
    return module.out_proj(core)


def joint_readouts(model, hidden, rope_values, layout):
    """The checkpoint's two separately normalized attention terms, unchanged."""
    from .model import _base_model
    base = _base_model(model.backbone)
    layer = base.layers[model.joint_layer]
    attn = layer.self_attn
    rope = sys.modules[type(attn).__module__].apply_rotary_pos_emb
    x = layer.input_layernorm(hidden)
    heads, kv_heads, d = base.config.num_attention_heads, base.config.num_key_value_heads, attn.head_dim
    n = hidden.shape[1]
    k = attn.k_norm(attn.k_proj(x).view(1, n, kv_heads, d)).transpose(1, 2)
    v = attn.v_proj(x).view(1, n, kv_heads, d).transpose(1, 2)
    k, _ = rope(k, k, *rope_values)
    k, v = (t.float().repeat_interleave(heads // kv_heads, 1)[0] for t in (k, v))
    read = layout.read_indices
    q, gate = attn.q_proj(x[0, read]).view(-1, heads, 2 * d).chunk(2, -1)
    q = attn.q_norm(q).transpose(0, 1)[None]
    q, _ = rope(q, q, *(t[:, read] for t in rope_values))
    scores = fp32_einsum("hrd,hnd->hrn", q[0].float(), k) * attn.scaling
    own_mask = layout.allow_between(read, layout.indices)
    own = fp32_einsum("hrn,hnd->rhd", scores.masked_fill(~own_mask[None], -torch.inf).softmax(-1), v)
    qi, ci = layout.question_ids, layout.candidate_ids
    cross_mask = ((qi[read, None] == qi[None]) & (ci[read, None] >= 0)
                  & (ci[None] >= 0) & (ci[read, None] != ci[None]))
    if model.spec.candidate_interaction == "last-readout-cross":
        is_read = torch.zeros(n, dtype=torch.bool, device=cross_mask.device)
        is_read[read] = True  # candidate readouts are their branches' final tokens
        cross_mask &= is_read[None]
    has_keys = cross_mask.any(-1)
    cross_scores = scores.masked_fill(~cross_mask[None], -torch.inf)
    weights = torch.where(has_keys[None, :, None], cross_scores, 0.0).softmax(-1)
    weights = weights * has_keys[None, :, None]
    cross = fp32_einsum("hrn,hnd->rhd", weights, v)
    mixed = (own + torch.tanh(model.joint_gate.float())[None, :, None] * cross).flatten(1).to(x.dtype)
    out = attn.o_proj(mixed * gate.flatten(1).sigmoid())
    result = hidden[0, read] + out
    return base.norm(result + layer.mlp(layer.post_attention_layernorm(result))).float()


def tree_features(model, encoded):
    return _execute_tree(model, [encoded])[0]


def tree_batch_features(model, records):
    """Pool complete records, respecting the unique-token memory budget."""
    if getattr(model, "fsdp_layer_calls", False):
        # FSDP collectives require every rank to call each layer the same number of times,
        # so a sharded forward is exactly one tree pass; it may not split by budget.
        total = sum(unique_tokens(r) for r in records)
        if total > model.spec.max_padded_tokens:
            raise ValueError(f"FSDP microbatch has {total} unique tokens, over max_padded_tokens="
                             f"{model.spec.max_padded_tokens}; raise the budget or lower batch_size")
        return _execute_tree(model, list(records))
    output, chunk, tokens = [], [], 0
    for encoded in records:
        count = unique_tokens(encoded)
        if count > model.spec.max_padded_tokens:
            raise ValueError("a record exceeds the tree unique tokens budget")
        if chunk and tokens + count > model.spec.max_padded_tokens:
            output.extend(_execute_tree(model, chunk))
            chunk, tokens = [], 0
        chunk.append(encoded)
        tokens += count
    if chunk:
        output.extend(_execute_tree(model, chunk))
    return output


def _execute_tree(model, records):
    from .model import _base_model
    base = _base_model(model.backbone)
    if base.config.model_type != "qwen3_5_text":
        raise NotImplementedError("packed tree execution currently supports Qwen3.5 text models")
    if base.config._attn_implementation not in {"eager", "sdpa"}:
        raise ValueError("tree masks require eager or sdpa attention")
    if model.training:
        if any(getattr(m, "gradient_checkpointing", False) for m in base.modules()):
            raise ValueError("tree training uses tree_checkpointing, not HF layer checkpointing")
        if model.spec.lora_dropout or getattr(base.config, "attention_dropout", 0):
            raise ValueError("tree training requires zero dropout")
    if not any(r.questions for r in records):
        return [[] for _ in records]
    layout = TreeLayout.from_records(records, model.device, model.spec.max_padded_tokens)
    hidden = base.embed_tokens(layout.ids)
    rope = base.rotary_emb(hidden, layout.positions[None].expand(3, -1, -1))
    mask = None
    if not layout.batched:
        mask = torch.zeros(layout.allow.shape, device=hidden.device, dtype=hidden.dtype)
        mask.masked_fill_(~layout.allow, -torch.inf)
        mask = mask[None, None]
    checkpointing = model.training and model.tree_checkpointing and torch.is_grad_enabled()

    def run(fn, h):
        if checkpointing:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(fn, h, use_reentrant=False, preserve_rng_state=False)
        return fn(h)

    # A single traversal of the decoder stack. No model-global hooks or cache.
    # Under FSDP each decoder layer must be entered through its own module call so the
    # parameter all-gather, reshard and gradient reduce-scatter hooks fire (layer_call).
    layer_call = getattr(model, "fsdp_layer_calls", False)
    for i, layer in enumerate(base.layers):
        if i == model.joint_layer:
            joint = lambda h: joint_readouts(model, h, rope, layout)
            result = run((lambda h, layer=layer: layer(h, joint)) if layer_call else joint, hidden)
            break
        # Bind the layer for checkpoint replay after the outer loop has finished.
        def decoder(h, layer=layer):
            x = layer.input_layernorm(h)
            if layer.block_type == "linear_attention":
                mixed = linear_tree(layer.linear_attn, x, layout, model.spec)
            elif layout.batched:
                mixed = full_attention_tree(layer.self_attn, x, rope, layout, model.spec)
            else:
                mixed, _ = layer.self_attn(x, position_embeddings=rope, attention_mask=mask)
            h = h + mixed
            return h + layer.mlp(layer.post_attention_layernorm(h))
        hidden = run((lambda h, layer=layer, decoder=decoder: layer(h, decoder)) if layer_call else decoder, hidden)
    else:
        result = base.norm(hidden[0, layout.read_indices]).float()
    features, start = [], 0
    for reads in layout.readouts:
        features.append((result[start], result[start + 1:start + len(reads)]))
        start += len(reads)
    output, start = [], 0
    for count in layout.record_questions:
        output.append(features[start:start + count])
        start += count
    return output
