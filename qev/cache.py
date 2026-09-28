# Adapted for Qev in 2026; see NOTICE and provenance.json.
"""Independent hybrid-cache forks with a single tensor copy per branch."""
import copy

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer


def fork_inference_cache(cache, copies=1):
    if copies < 1:
        raise ValueError("cache copies must be positive")
    # Limit the optimized path to the audited Qwen3/Qwen3.5 cache layouts.
    if any(type(layer) not in (DynamicLayer, LinearAttentionLayer) for layer in cache.layers):
        fork = copy.deepcopy(cache)
        if copies != 1:
            tensor = next(t for layer in cache.layers for t in vars(layer).values() if isinstance(t, torch.Tensor))
            fork.reorder_cache(torch.zeros(copies, dtype=torch.long, device=tensor.device))
        return fork
    memo = {}
    for layer in cache.layers:
        tensors = ([layer.keys, layer.values] if type(layer) is DynamicLayer else
                   [*layer.conv_states.values(), *layer.recurrent_states.values()])
        for tensor in tensors:
            if tensor is None or id(tensor) in memo:
                continue
            if tensor.shape[0] != 1:
                raise ValueError("prefix cache must contain exactly one parent row")
            # deepcopy's memo skips a redundant clone before batch expansion.
            memo[id(tensor)] = (tensor.clone() if copies == 1 else tensor.index_select(
                0, torch.zeros(copies, dtype=torch.long, device=tensor.device)))
    return copy.deepcopy(cache, memo)


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
