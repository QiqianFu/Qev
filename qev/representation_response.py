# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Task-head response geometry, independent of neuron-coordinate alignment."""


def response_shard(row, shards):
    if shards < 1:
        raise ValueError('positive shard count required')
    return int(row['original_key'][:16], 16) % shards


class CaptureTaskStates:
    """Observe scorer input without adding hooks to TransformerEncoderLayer.

    Encoder-layer hooks can disable PyTorch's fused inference path. The scorer
    receives [query, candidate] concatenations after that path has completed.
    Holding its input changes neither its values nor the head computation.
    """
    def __init__(self, head):
        self.head = head
        self.inputs = []
        self.handle = None

    def __enter__(self):
        if self.handle is not None:
            raise RuntimeError('capture context is already active')
        def capture(module, args):
            value = args[0]
            if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != 2*self.head.project.out_features:
                raise ValueError('unexpected task-head scorer input')
            self.inputs.append(value)
        self.handle = self.head.scorer[0].register_forward_pre_hook(capture)
        return self

    @property
    def features(self):
        import torch
        d = self.head.project.out_features
        # SetDecisionHead concatenates [candidate, repeated query], in that order.
        return [torch.cat([value[:1, d:], value[:, :d]], dim=0) for value in self.inputs]

    def __exit__(self, kind, value, traceback):
        self.handle.remove()
        self.handle = None


def response_geometry(original, edited):
    """Normalize rows, then express finite feature changes in within-task axes.

    Gram(D) preserves change magnitudes and co-movement. D @ F0.T also
    records signed movement relative to the original query/candidate states.
    Global rotations or isometric changes of feature width leave both intact.
    """
    import torch
    if original.ndim != 2 or original.shape != edited.shape or original.shape[0] < 2:
        raise ValueError('feature rows must align within a model and pair')
    dtype = torch.float64 if original.dtype == torch.float64 else torch.float32
    with torch.autocast(original.device.type, enabled=False):
        a, b = original.to(dtype), edited.to(dtype)
        a_norm = a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        b_norm = b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        a = a / a_norm
        b = b / b_norm
        delta = b-a
        return {'original_gram': a @ a.T, 'edited_gram': b @ b.T,
                'cross_gram': a @ b.T, 'sensitivity': delta @ delta.T,
                'anchored_response': delta @ a.T,
                'log_norm_ratio': (b_norm/a_norm).log().squeeze(-1)}


def response_geometry_loss(original, edited, target, kind='sensitivity'):
    """Pure teacher feature target; zero-target controls can pass zero matrices."""
    import torch
    if kind not in {'sensitivity', 'anchored_response'}:
        raise ValueError('unknown representation response objective')
    value = response_geometry(original, edited)[kind]
    expected = torch.as_tensor(target, device=value.device, dtype=value.dtype).detach()
    if value.shape != expected.shape or not torch.isfinite(expected).all():
        raise ValueError('unaligned or nonfinite teacher response matrix')
    return (value-expected).square().mean()
