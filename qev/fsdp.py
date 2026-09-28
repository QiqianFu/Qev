# Adapted for Qev in 2026; see NOTICE and provenance.json.
"""Full-parameter training with FSDP2 and CPU-offloaded FP32 master weights / AdamW state.

Only tree execution is supported: `_execute_tree` enters each decoder layer through its
module call (model.fsdp_layer_calls), so FSDP's all-gather, reshard and reduce-scatter
hooks fire even though the layer body is Qev's tree computation, not HF forward.
"""
from types import MethodType

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

from .model import _base_model


def _tree_forward(self, hidden, fn):
    return fn(hidden)


def shard_full_model(model, device, world, cpu_offload=True):
    """Shard every decoder layer, then the FP32 root (embeddings, norm, joint gate, decision head)."""
    if model.prefix_execution not in {"tree", "tree-batched"}:
        raise ValueError("full fine-tuning requires tree or tree-batched execution")
    model.float()
    model.backbone.requires_grad_(True)
    mesh = init_device_mesh(device.type, (world,))
    # With cpu_offload=False the FP32 shards, gradients and AdamW state live on the GPU (needs enough ranks).
    offload = CPUOffloadPolicy(pin_memory=True) if cpu_offload else None
    # Master weights stay FP32; decoder layers gather in spec.weights_dtype. The root unit
    # (embeddings, final norm, joint gate, decision head) gathers in FP32, once per forward:
    # the head runs once per question, and question counts differ across ranks, so it must
    # not be its own FSDP unit (unequal collective counts deadlock NCCL).
    compute = torch.bfloat16 if model.spec.weights_dtype == "bf16" else torch.float32
    layers = MixedPrecisionPolicy(param_dtype=compute, reduce_dtype=torch.float32)
    root = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
    for layer in _base_model(model.backbone).layers:
        # The tree executor supplies the layer body; HF forward is not used in this mode.
        layer.forward = MethodType(_tree_forward, layer)
        fully_shard(layer, mesh=mesh, mp_policy=layers, **({"offload_policy": offload} if offload else {}))
    fully_shard(model, mesh=mesh, mp_policy=root, **({"offload_policy": offload} if offload else {}))
    model.compute_device = device
    model.fsdp_layer_calls = True
    return model


def full_state_dict(model):
    """Collective: every rank must call. Rank 0 receives full CPU tensors."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    return get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))


def gathered(tensor):
    """Collective for a DTensor; plain tensors pass through."""
    return tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor
