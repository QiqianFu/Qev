# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Hybrid-cache forks that read the parent's attention KV in place.

A fork copies only the small fixed-size DeltaNet states. Its full-attention
layers keep references to the parent's KV and store only the fork's own tokens;
attention over the shared prefix and the own suffix is merged by log-sum-exp.
"""
from contextlib import contextmanager
import contextvars
import copy

import torch
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer, LinearAttentionLayer
from transformers.modeling_utils import AttentionInterface

# Not registered as a mask interface: HF then builds no 4D [rows, new, prefix + new] mask.
SHARED_PREFIX = "qev_shared_prefix"
_ACTIVE = contextvars.ContextVar("qev_shared_prefix_cache", default=None)


class SharedPrefixLayer(DynamicLayer):
    """Full-attention KV of a fork. Parent segments are read, never copied or written.

    ``segments`` are single-row (keys, values) pairs [1, kv_heads, length, head_dim]
    that every new token attends to in full; ``keys``/``values`` hold only the
    fork's own tokens, one row per branch.
    """

    def __init__(self, segments=()):
        super().__init__()
        self.segments = tuple(segments)
        self.prefix_length = sum(keys.shape[-2] for keys, _ in self.segments)
        self.unread = False

    def update(self, key_states, value_states, *args, **kwargs):
        # Own KV alone is wrong under any other attention implementation.
        if _ACTIVE.get() is None:
            raise RuntimeError("run shared-prefix forks under shared_prefix_attention")
        keys, values = super().update(key_states, value_states)
        self.unread = True
        return keys, values

    def get_seq_length(self):
        return self.prefix_length + super().get_seq_length()


class SharedPrefixCache(Cache):
    """A fork of a single-row inference cache; run its forwards in ``shared_prefix_attention``."""

    def __init__(self, layers):
        super().__init__(layers=layers)


def kv_segments(layer):
    """Single-row (keys, values) segments of an attention cache layer, in position order."""
    segments = tuple(getattr(layer, "segments", ()))
    if layer.is_initialized and layer.keys.shape[-2]:
        if layer.keys.shape[0] != 1:
            raise ValueError("prefix cache must contain exactly one parent row")
        segments += ((layer.keys, layer.values),)
    return segments


def fork_inference_cache(cache, copies=1, *, share_prefix=False):
    """Fork a single-row cache into ``copies`` independent rows.

    With ``share_prefix`` the fork references the parent's attention KV instead of
    copying it; only callers that run the fork under ``shared_prefix_attention``
    (audited attention modules) may request it. Otherwise every tensor is copied.
    """
    if copies < 1:
        raise ValueError("cache copies must be positive")
    types = {type(layer) for layer in cache.layers}
    if SharedPrefixLayer in types and not share_prefix:
        raise ValueError("a shared-prefix cache can only be forked with share_prefix")
    # Limit the optimized path to the audited Qwen3/Qwen3.5 cache layouts.
    if not types <= {DynamicLayer, LinearAttentionLayer, SharedPrefixLayer}:
        fork = copy.deepcopy(cache)
        if copies != 1:
            tensor = next(t for layer in cache.layers for t in vars(layer).values() if isinstance(t, torch.Tensor))
            fork.reorder_cache(torch.zeros(copies, dtype=torch.long, device=tensor.device))
        return fork
    memo = {}
    for layer in cache.layers:
        if type(layer) is LinearAttentionLayer:
            tensors = [*layer.conv_states.values(), *layer.recurrent_states.values()]
        else:
            tensors = [] if share_prefix else [layer.keys, layer.values]
        for tensor in tensors:
            if tensor is None or id(tensor) in memo:
                continue
            if tensor.shape[0] != 1:
                raise ValueError("prefix cache must contain exactly one parent row")
            # deepcopy's memo skips a redundant clone before batch expansion.
            memo[id(tensor)] = (tensor.clone() if copies == 1 else tensor.index_select(
                0, torch.zeros(copies, dtype=torch.long, device=tensor.device)))
    if not share_prefix:
        return copy.deepcopy(cache, memo)
    return SharedPrefixCache([copy.deepcopy(layer, memo) if type(layer) is LinearAttentionLayer
                              else SharedPrefixLayer(kv_segments(layer)) for layer in cache.layers])


@contextmanager
def shared_prefix_attention(config, cache):
    """Run full attention of one forward over ``cache``'s parent segments and own suffix.

    The backbone selects attention by ``config._attn_implementation``, so this is
    not safe while another thread runs the same backbone.
    """
    previous, token = config._attn_implementation, _ACTIVE.set(cache)
    config._attn_implementation = SHARED_PREFIX
    try:
        yield
    finally:
        config._attn_implementation = previous
        _ACTIVE.reset(token)
    if any(getattr(layer, "unread", False) for layer in cache.layers):
        raise RuntimeError("shared-prefix KV was not read by shared-prefix attention")


def _attend(query, key, value, scale, causal):
    """Attention output and natural log-sum-exp per query; inputs [batch, heads, length, dim]."""
    if query.device.type == "cuda":
        output, lse = torch.ops.aten._scaled_dot_product_efficient_attention(
            query, key, value, None, True, 0.0, causal, scale=scale)[:2]
        return output, lse[..., :query.shape[-2]]
    if query.device.type == "cpu":
        return torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(query, key, value, 0.0, causal, scale=scale)
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
    if causal:
        scores = scores.masked_fill(torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device).triu(1),
                                    float("-inf"))
    lse = scores.logsumexp(-1)
    return torch.matmul((scores - lse[..., None]).exp(), value.float()).to(query.dtype), lse


def _attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0,
                       sliding_window=None, **kwargs):
    """Fork attention without materializing the parent KV per row.

    Parent segments are visible to every new token, so they need no mask (see
    ``segment_attention``). Own tokens are causal; right padding is only visible
    to padding. Partial results are merged by log-sum-exp in FP32.
    """
    cache = _ACTIVE.get()
    layer = None if cache is None else cache.layers[module.layer_idx]
    if type(layer) is not SharedPrefixLayer or attention_mask is not None or sliding_window or dropout:
        raise NotImplementedError("shared-prefix attention supports full causal attention over a fork only")
    layer.unread = False
    heads, length, dim = query.shape[1:]
    groups = heads // key.shape[1]
    scale = dim ** -0.5 if scaling is None else scaling
    if groups > 1:  # own tokens are few; HF repeat_kv order: query head h reads KV head h // groups
        key, value = (x.repeat_interleave(groups, 1) for x in (key, value))
    past = key.shape[-2] - length
    parts = [_attend(query, key[:, :, past:], value[:, :, past:], scale, True)]
    if past:
        parts.append(_attend(query, key[:, :, :past], value[:, :, :past], scale, False))
    parts.extend(segment_attention(query, keys, values, scale) for keys, values in layer.segments)
    return merge_attention(parts).to(query.dtype).transpose(1, 2).contiguous(), None


def segment_attention(query, keys, values, scale):
    """Attention of every row of ``query`` [rows, heads, length, dim] to one fully visible single-row segment.

    The rows and each KV head's query group fold into one sequence: one fused
    call, and the segment is read in place instead of repeated per row or head.
    """
    rows, heads, length, dim = query.shape
    kv_heads = keys.shape[1]
    groups = heads // kv_heads
    folded = query.unflatten(1, (kv_heads, groups)).transpose(0, 1).reshape(1, kv_heads, -1, dim)
    output, lse = _attend(folded, keys, values, scale, False)
    return (output.reshape(kv_heads, rows, groups, length, dim).transpose(0, 1).flatten(1, 2),
            lse.reshape(kv_heads, rows, groups, length).transpose(0, 1).flatten(1, 2))


def merge_attention(parts):
    """Combine (output, log-sum-exp) attention over disjoint key sets, in FP32."""
    total = torch.logsumexp(torch.stack([lse for _, lse in parts]), 0)
    return sum(o.float() * (lse - total).exp()[..., None] for o, lse in parts)


AttentionInterface.register(SHARED_PREFIX, _attention_forward)


class FunctionalLinearAttentionLayer(LinearAttentionLayer):
    """Assign new states instead of modifying tensors saved for backward."""

    def update_conv_state(self, conv_states, state_idx=0, conv_kernel_size=None, **kwargs):
        size = self.conv_kernel_size[state_idx] or conv_kernel_size or conv_states.shape[-1]
        self.conv_kernel_size[state_idx] = size
        self.dtype, self.device = conv_states.dtype, conv_states.device
        if self.has_previous_state[state_idx]:
            full = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)
        else:
            full = torch.nn.functional.pad(conv_states, (max(0, size - conv_states.shape[-1]), 0))
        self.conv_states[state_idx] = full[..., -size:]
        self.is_conv_states_initialized[state_idx] = True
        self.has_previous_state[state_idx] = True
        return full

    def update_recurrent_state(self, recurrent_states, state_idx=0, **kwargs):
        self.recurrent_states[state_idx] = recurrent_states
        self.is_recurrent_states_initialized[state_idx] = True
        return recurrent_states


class FunctionalCache(DynamicCache):
    """Differentiable tree prefixes for audited attention/DeltaNet layouts.

    Forks own metadata but share read-only parent tensors; every state update
    allocates/assigns a new tensor. Do not use single-token cached decode, whose
    fused convolution kernel mutates state outside the cache update method.
    """

    def __init__(self, config):
        super().__init__(config=config)
        for i, layer in enumerate(self.layers):
            if type(layer) is LinearAttentionLayer:
                self.layers[i] = FunctionalLinearAttentionLayer()
            elif type(layer) is not DynamicLayer:
                raise ValueError(f"shared training does not support {type(layer).__name__}")

    def fork(self, copies=1):
        if copies < 1:
            raise ValueError("cache copies must be positive")
        result = copy.copy(self)
        result.layers = []
        for parent in self.layers:
            child = copy.copy(parent)
            # Flags and state maps must not alias their parent containers.
            for name, value in vars(parent).items():
                if isinstance(value, dict):
                    setattr(child, name, value.copy())
            if copies != 1:
                def expand(tensor):
                    if tensor is None:
                        return None
                    if tensor.shape[0] != 1:
                        raise ValueError("tree cache must contain one parent row")
                    return tensor.expand(copies, *tensor.shape[1:])
                if type(child) is DynamicLayer:
                    child.keys, child.values = expand(child.keys), expand(child.values)
                else:
                    child.conv_states = {k: expand(v) for k, v in child.conv_states.items()}
                    child.recurrent_states = {k: expand(v) for k, v in child.recurrent_states.items()}
            result.layers.append(child)
        return result
