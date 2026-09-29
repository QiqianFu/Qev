# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Execution policies kept separate from model weights and input semantics."""

READOUT_PRECISION = "fp32-head-and-joint-reductions.v1"


def fp32_einsum(equation, *operands):
    """Keep sensitive attention contractions in FP32, even under outer AMP.

    Casting inputs alone does not protect matmul from autocast. Leave the
    surrounding projections in their configured precision and keep gradients.
    """
    import torch

    with torch.autocast(operands[0].device.type, enabled=False):
        return torch.einsum(equation, *(x.float() for x in operands))


def configure_fp32():
    """Use IEEE FP32 in PyTorch AND the pinned FLA/Triton implementation.

    Must run before compiling GPU kernels. PyTorch's TF32 flags alone do not
    control Triton dot products or FLA's explicitly selected triangular solve.
    """
    import importlib.util
    import os
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    os.environ["FLA_TRIL_PRECISION"] = "ieee"
    if torch.cuda.is_available() and importlib.util.find_spec("fla") is not None:
        import importlib.metadata
        if importlib.metadata.version("fla-core") != "0.5.2":
            raise RuntimeError("FP32 FLA policy is audited for fla-core==0.5.2")
        import triton
        import triton.language as tl
        import fla.ops.gated_delta_rule.chunk_fwd as chunk_fwd
        triton.knobs.language.fp32_default = "ieee"
        chunk_fwd.SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")
    return {"weights": "fp32", "autocast": False, "torch_tf32": False,
            "triton_dot": "ieee", "fla_triangular_solve": "ieee"}


def configure_checkpointing(backbone, enabled=True, layers=None):
    base = backbone.get_base_model() if hasattr(backbone, "get_base_model") else backbone
    if layers is not None:
        if not enabled or not hasattr(base, "layers"):
            raise ValueError("selective checkpointing requires enabled checkpointing and decoder layers")
        if not isinstance(layers, list) or any(type(i) is not int or not 0 <= i < len(base.layers) for i in layers):
            raise ValueError("gradient_checkpointing_layers must be a list of valid decoder indices")
        if len(set(layers)) != len(layers):
            raise ValueError("duplicate gradient checkpointing layer")
    if enabled:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if layers is not None:
            for i, layer in enumerate(base.layers):
                layer.gradient_checkpointing = i in layers
    else:
        backbone.gradient_checkpointing_disable()
