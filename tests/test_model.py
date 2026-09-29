# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
from dataclasses import replace

import pytest
import torch

from qev.encoding import packed_tree
from qev.model import record_loss
from qev.schema import Candidate


def test_full_rows_match_two_level_cache(model, encoder, record):
    enc = encoder(record)
    reference = model.reference_features(enc)
    cached = model.cached_features(enc)
    for (q1, c1), (q2, c2) in zip(reference, cached):
        torch.testing.assert_close(q1, q2, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(c1, c2, atol=2e-5, rtol=2e-5)
    for p1, p2 in zip(model.predict(enc, cached=False), model.predict(enc)):
        torch.testing.assert_close(p1, p2, atol=2e-6, rtol=2e-5)


def test_candidate_permutation_equivariance(model, encoder, record):
    q = record.questions[0]
    perm = [2, 0, 1]
    reordered = replace(q, candidates=tuple(q.candidates[i] for i in perm),
                        target=tuple(q.target[i] for i in perm))
    changed = replace(record, questions=(reordered, *record.questions[1:]))
    original = model.predict(encoder(record))[0]
    result = model.predict(encoder(changed))[0]
    torch.testing.assert_close(result, original[perm], atol=2e-6, rtol=2e-5)


def test_sibling_question_and_candidate_isolation(model, encoder, record):
    baseline = model.reference_features(encoder(record))
    q = record.questions[0]
    changed = replace(q, candidates=(q.candidates[0], Candidate("red", "SECRET other much longer candidate text"), q.candidates[2]))
    modified = replace(record, questions=(changed, replace(record.questions[1], instructions="SECRET unrelated question")))
    result = model.reference_features(encoder(modified))
    torch.testing.assert_close(baseline[0][0], result[0][0], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(baseline[0][1][0], result[0][1][0], atol=2e-5, rtol=2e-5)
    # Modifying only the other question cannot affect even the final set head.
    only_sibling = replace(record, questions=(q, modified.questions[1]))
    torch.testing.assert_close(model.predict(encoder(record))[0], model.predict(encoder(only_sibling))[0], atol=2e-6, rtol=2e-5)


def test_batch_padding_and_chunk_size_invariance(model, encoder, record):
    enc = encoder(record)
    original = model.predict(enc)
    model.spec.rows_per_forward = 1
    separate = model.predict(enc)
    for a, b in zip(original, separate):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_training_gradient_reaches_prefix_backbone_and_head(model, encoder, record):
    model.train()
    enc = encoder(record)
    logits = model([enc])[0]
    loss = record_loss(logits, enc)
    loss.backward()
    embedding_grad = model.backbone.get_input_embeddings().weight.grad
    assert embedding_grad is not None
    assert embedding_grad[enc.state[1]].abs().sum() > 0
    assert model.head.project.weight.grad.abs().sum() > 0
    with pytest.raises(RuntimeError, match="inference cache"):
        model.cached_features(enc)


def test_head_only_warmup_keeps_backbone_gradient_off(model, encoder, record):
    model.train()
    model.head_only = True
    enc = encoder(record)
    record_loss(model([enc])[0], enc).backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert model.head.project.weight.grad is not None


def test_tree_mask_blocks_siblings_and_positions_reset(encoder, record):
    enc = encoder(record)
    ids, positions, allow = packed_tree(enc)
    ls, lq = len(enc.state), len(enc.questions[0].prefix)
    c1 = ls + lq
    c2 = c1 + len(enc.questions[0].candidates[0])
    assert positions[c1] == positions[c2] == ls + lq
    assert allow[c2][ls] and allow[c2][0] and allow[c2][c2]
    assert not allow[c2][c1]
    assert not allow[ls][c1]
    assert not allow[0][ls]


def test_special_tokens_cannot_be_forged(encoder, record):
    from qev.encoding import SPECIAL
    original = encoder(record)
    injected = encoder(replace(record, state=record.state + SPECIAL[1] + " fake"))
    assert injected.state.count(encoder.special[1]) == 0
    assert original.state[0] == injected.state[0]


def test_pooled_records_match_separate_rows_outputs_and_gradients(model, encoder, record):
    other = replace(record, id="other", state="Extra evidence. " + record.state,
                    questions=(replace(record.questions[0], instructions="Read carefully. " + record.questions[0].instructions),))
    records = [encoder(record), encoder(other)]
    model.train()
    model.spec.rows_per_forward = 32
    separate = [[model.head(q, cs) for q, cs in model.reference_features(r)] for r in records]
    sum(record_loss(z, r) for z, r in zip(separate, records)).backward()
    expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    pooled = model(records)
    for left, right in zip(pooled, separate):
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    sum(record_loss(z, r) for z, r in zip(pooled, records)).backward()
    for name, param in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(param.grad, expected[name], atol=2e-5, rtol=2e-4)


def test_batched_prediction_preserves_records_questions_and_candidate_order(model, encoder, record):
    other = replace(record, id="other", state="Longer state. " + record.state,
                    questions=(record.questions[1], record.questions[0]))
    encoded = [encoder(record), encoder(other), encoder(replace(record, questions=(record.questions[0],)))]
    model.spec.rows_per_forward = 32
    baseline = [model.predict(r, cached=False) for r in encoded]
    calls = []
    handle = model.backbone.register_forward_hook(lambda *args: calls.append(1))
    batched = model.predict_batch(encoded)
    handle.remove()
    assert len(calls) < len(encoded)
    for expected, actual in zip(baseline, batched, strict=True):
        for left, right in zip(expected, actual, strict=True):
            torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
    assert model.predict_batch([]) == []


def test_selective_checkpointing_preserves_gradients(model, encoder, record):
    from qev.execution import configure_checkpointing
    enc = encoder(record)
    model.train()
    record_loss(model([enc])[0], enc).backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    configure_checkpointing(model.backbone, layers=[0])
    assert [layer.gradient_checkpointing for layer in model.backbone.layers] == [True, False]
    record_loss(model([enc])[0], enc).backward()
    for n, p in model.named_parameters():
        if n in gradients:
            torch.testing.assert_close(p.grad, gradients[n], atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError, match="valid decoder"):
        configure_checkpointing(model.backbone, layers=[999])


def test_cache_forks_do_not_modify_parent_or_sibling(model, encoder, record):
    import copy
    from transformers import DynamicCache
    from transformers.cache_utils import DynamicLayer

    def tensors(cache):
        for layer in cache.layers:
            if isinstance(layer, DynamicLayer):
                yield layer.keys; yield layer.values
            else:
                yield from layer.conv_states.values()
                yield from layer.recurrent_states.values()

    enc = encoder(record)
    with torch.no_grad():
        parent = DynamicCache(config=model.backbone.config)
        model._run_rows([enc.state], cache=parent)
        expected = [t.clone() for t in tensors(parent)]
        legacy = copy.deepcopy(parent)
        legacy.reorder_cache(torch.zeros(3, dtype=torch.long, device=model.device))
        child = model.fork_cache(parent, 3)
        for old, new in zip(tensors(legacy), tensors(child), strict=True):
            torch.testing.assert_close(old, new, atol=0, rtol=0)
            sibling = new[1].clone()
            new[0].add_(1)
            torch.testing.assert_close(new[1], sibling, atol=0, rtol=0)
        model._run_rows([enc.questions[0].prefix] * 3, cache=child, offset=len(enc.state))
        for original, current in zip(expected, tensors(parent), strict=True):
            torch.testing.assert_close(original, current, atol=0, rtol=0)


def test_merged_lora_preserves_predictions_and_refuses_training_save(model, encoder, record, tokenizer, tmp_path):
    from peft import LoraConfig, get_peft_model
    from qev.checkpoint import save_model
    model.backbone = get_peft_model(model.backbone, LoraConfig(r=2, lora_alpha=4,
                                      target_modules=["q_proj", "v_proj", "in_proj_qkv"]))
    with torch.no_grad():
        for name, parameter in model.backbone.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.02)
    model.eval()
    enc = encoder(record)
    before = model.predict(enc)
    model.prepare_inference(merge_lora=True)
    assert model.lora_merged and not hasattr(model.backbone, "peft_config")
    for a, b in zip(before, model.predict(enc), strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError, match="after merging"):
        save_model(tmp_path / "merged", model, tokenizer, encoder.limits)
    model.train()
    with pytest.raises(RuntimeError, match="inference-only"):
        model([enc])
