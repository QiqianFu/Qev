# Adapted for Qev in 2026; see NOTICE and provenance.json.
"""FP32 readouts must survive outer BF16 autocast, including backward replay."""
from dataclasses import replace
import os

import pytest
import torch

from qev.execution import fp32_einsum
from qev.model import QevModel, ModelSpec, SetDecisionHead, lora_modules, record_loss
from conftest import tiny_backbone


@pytest.mark.parametrize('input_dtype', [torch.float32, torch.bfloat16])
def test_head_outputs_and_gradients_identical_inside_autocast(input_dtype):
    torch.manual_seed(77)
    device = os.environ.get('QEV_TEST_DEVICE', 'cpu')
    head = SetDecisionHead(64, dim=32, heads=4, layers=1).to(device).train()
    q = torch.randn(64, device=device, dtype=input_dtype, requires_grad=True)
    cs = torch.randn(5, 64, device=device, dtype=input_dtype, requires_grad=True)
    outputs, gradients = [], []
    for amp in (False, True):
        head.zero_grad(set_to_none=True)
        q.grad = cs.grad = None
        with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
            z = head(q, cs)
            assert z.dtype == torch.float32
            loss = torch.nn.functional.cross_entropy(z[None], torch.tensor([2], device=device))
            assert torch.is_autocast_enabled(device) == amp
        loss.backward()
        outputs.append(z.detach())
        gradients.append([q.grad.clone(), cs.grad.clone(), *(p.grad.clone() for p in head.parameters())])
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_fp32_contraction_matches_full_precision_output_and_gradients():
    device = os.environ.get('QEV_TEST_DEVICE', 'cpu')
    torch.manual_seed(42)
    a = torch.randn(2, 7, 16, device=device, requires_grad=True)
    b = torch.randn(2, 11, 16, device=device, requires_grad=True)
    expected = torch.einsum('hrd,hnd->hrn', a, b)
    old = torch.autograd.grad(expected.square().sum(), (a, b))
    with torch.autocast(device, dtype=torch.bfloat16):
        actual = fp32_einsum('hrd,hnd->hrn', a, b)
        assert actual.dtype == torch.float32
        assert torch.is_autocast_enabled(device)
    new = torch.autograd.grad(actual.square().sum(), (a, b))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for x, y in zip(old, new, strict=True):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


@pytest.fixture
def precision_model(tokenizer):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(87)
    device = os.environ.get('QEV_TEST_DEVICE', 'cpu')
    base = tiny_backbone(len(tokenizer), hybrid=True).to(device=device, dtype=torch.bfloat16)
    base = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=lora_modules(base.config)))
    spec = ModelSpec('tiny', head_dim=32, head_heads=4, head_layers=1, lora_rank=2,
                     weights_dtype='bf16', attention='eager', candidate_interaction='last-full-attention')
    model = QevModel(base, spec, tokenizer.pad_token_id)
    with torch.no_grad():
        model.joint_gate.fill_(0.6)
        for name, p in model.named_parameters():
            if 'lora_B' in name:
                p.normal_(std=.01)
    return model


def trace_precision(model, monkeypatch):
    contractions, head, backbone, handles = [], [], [], []
    original = torch.einsum

    def einsum(equation, *operands, **kwargs):
        result = original(equation, *operands, **kwargs)
        # These equations occur only in the custom final attention layer.
        if equation in {'hrd,hnd->hrn', 'hrn,hnd->rhd', 'bhd,bhwd->bhw', 'bhw,bhwd->bhd',
                        'bhd,shmd->bhsm', 'bhn,bhnd->bhd', 'bhd,hpd->bhp', 'bhp,hpd->bhd',
                        'bhd,shwd->bhsw', 'bhn,hnd->bhd'}:
            contractions.append(result.dtype)
            assert not torch.is_autocast_enabled(result.device.type)
        return result

    monkeypatch.setattr(torch, 'einsum', einsum)
    handles.append(model.head.project.register_forward_hook(lambda m, a, out: head.append(out.dtype)))
    for name, module in model.backbone.named_modules():
        if name.endswith('in_proj_qkv'):
            handles.append(module.register_forward_hook(lambda m, a, out: backbone.append(out.dtype)))
    return contractions, head, backbone, handles


@pytest.mark.parametrize('execution', ['leaf-rows', 'tree', 'tree-batched'])
@pytest.mark.parametrize('checkpointing', [False, True])
def test_readout_precision_with_backward_and_checkpointing(precision_model, encoder, record, monkeypatch,
                                                          execution, checkpointing):
    from qev.execution import configure_checkpointing
    m = precision_model.train()
    m.prefix_execution = execution
    if execution == 'leaf-rows':
        configure_checkpointing(m.backbone, checkpointing)
    else:
        m.tree_checkpointing = checkpointing
    batch = [encoder(record), encoder(replace(record, id='second', state='Different context.'))]
    contractions, head, backbone, handles = trace_precision(m, monkeypatch)
    try:
        with torch.autocast(m.device.type, dtype=torch.bfloat16):
            logits = m(batch)
            loss = sum(record_loss(z, r) for z, r in zip(logits, batch, strict=True)) / len(batch)
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    assert contractions and set(contractions) == {torch.float32}
    assert head and set(head) == {torch.float32}
    assert backbone and set(backbone) == {torch.bfloat16}
    assert torch.isfinite(loss)
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert m.joint_gate.grad.abs().sum() > 0
    assert any(p.grad.abs().sum() > 0 for n, p in m.named_parameters() if 'lora_A' in n)


def test_cached_readouts_preserve_fp32_under_outer_autocast(precision_model, encoder, record, monkeypatch):
    m = precision_model.eval()
    contractions, head, backbone, handles = trace_precision(m, monkeypatch)
    try:
        with torch.autocast(m.device.type, dtype=torch.bfloat16):
            predictions = m.predict(encoder(record))
            assert torch.is_autocast_enabled(m.device.type)
    finally:
        for handle in handles:
            handle.remove()
    assert contractions and set(contractions) == {torch.float32}
    assert head and set(head) == {torch.float32}
    assert backbone and set(backbone) == {torch.bfloat16}
    assert all(torch.isfinite(p).all() for p in predictions)
