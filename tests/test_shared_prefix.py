# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
from dataclasses import replace

import pytest
import torch

from qev.model import record_loss


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("lora", [False, True])
def test_shared_prefix_outputs_gradients_and_update_match_leaf_rows(model, encoder, record, checkpointing, lora):
    if lora:
        from peft import LoraConfig, get_peft_model
        from qev.model import lora_modules
        model.backbone = get_peft_model(model.backbone, LoraConfig(r=2, lora_alpha=4,
                                            target_modules=lora_modules(model.backbone.config)))
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "lora_B" in name:
                    p.normal_(std=0.03)
    model.train()
    model.spec.rows_per_forward = 2
    records = [encoder(record), encoder(replace(record, id="other", state="More context. " + record.state))]
    original = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    baseline = model(records)
    sum(record_loss(z, r) for z, r in zip(baseline, records, strict=True)).backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    model.prefix_execution = "shared-prefix"
    model.shared_checkpointing = checkpointing
    actual = model(records)
    for aa, bb in zip(actual, baseline, strict=True):
        for a, b in zip(aa, bb, strict=True):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    sum(record_loss(z, r) for z, r in zip(actual, records, strict=True)).backward()
    for n, p in model.named_parameters():
        if n in gradients:
            assert p.grad is not None, n
            torch.testing.assert_close(p.grad, gradients[n], atol=3e-5, rtol=3e-4, msg=n)
    if lora:
        assert any(gradients[n].abs().sum() > 0 for n in gradients if "lora_A" in n)
    else:
        assert model.backbone.get_input_embeddings().weight.grad[records[0].state[1]].abs().sum() > 0
    torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01).step()
    for n, p in model.named_parameters():
        if n in gradients:
            torch.testing.assert_close(p, original[n] - 0.01 * gradients[n], atol=2e-6, rtol=2e-5)


def test_shared_prefix_head_only_and_checkpoint_guard(model, encoder, record):
    model.train()
    model.prefix_execution = "shared-prefix"
    model.head_only = True
    model.shared_checkpointing = True
    encoded = encoder(record)
    record_loss(model([encoded])[0], encoded).backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert model.head.project.weight.grad is not None
    model.head_only = False
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    with pytest.raises(ValueError, match="segment checkpointing"):
        model([encoded])
