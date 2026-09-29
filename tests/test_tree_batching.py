# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Independent records share matrix operations, never attention or recurrent state."""
from dataclasses import replace

import pytest
import torch

from qev.model import lora_modules, record_loss
from qev.tree import TreeLayout, unique_tokens
from test_tree_inference import tree_model, assert_features


def records(encoder, record):
    a = encoder(record)
    b = encoder(replace(record, id='second', state='A different, longer context. ' + record.state,
                        questions=record.questions[:1]))
    c = encoder(replace(record, id='third', state='Short.'))
    return [a, b, c]


def test_batched_tree_matches_separate_trees_and_reference(tree_model, encoder, record):
    batch = records(encoder, record)
    actual = tree_model.tree_batch_features(batch)
    for r, features in zip(batch, actual, strict=True):
        assert_features(tree_model.tree_features(r), features)
        assert_features(tree_model.reference_features(r), features)
    ps = tree_model.predict_batch(batch, tree=True)
    for r, outputs in zip(batch, ps, strict=True):
        for a, b in zip(tree_model.predict(r, tree=True), outputs, strict=True):
            torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)


def test_forest_positions_and_visibility_are_block_diagonal(encoder, record):
    batch = records(encoder, record)
    packed = TreeLayout.from_records(batch, 'cpu', 8192)
    separate = [TreeLayout(r, 'cpu', 8192) for r in batch]
    assert packed.positions.tolist() == torch.cat([r.positions for r in separate], 1).tolist()
    expected = torch.block_diag(*(r.allow for r in separate))
    torch.testing.assert_close(packed.allow, expected)


def test_batched_tree_isolates_records_and_keeps_output_order(tree_model, encoder, record):
    batch = records(encoder, record)
    before = tree_model.tree_batch_features(batch)
    changed = encoder(replace(record, state='Secret payload of a different length.', questions=record.questions[:1]))
    after = tree_model.tree_batch_features([changed, *batch[1:]])
    for a, b in zip(before[1:], after[1:], strict=True):
        assert_features(a, b)
    order = [2, 0, 1]
    shuffled = tree_model.tree_batch_features([batch[i] for i in order])
    for i, actual in zip(order, shuffled, strict=True):
        assert_features(before[i], actual)


def test_tree_batch_budget_and_empty_records(tree_model, encoder, record):
    batch = records(encoder, record)
    expected = tree_model.tree_batch_features(batch)
    tree_model.spec.max_padded_tokens = max(map(unique_tokens, batch))
    actual = tree_model.tree_batch_features(batch)
    for a, b in zip(expected, actual, strict=True):
        assert_features(a, b)
    empty = replace(batch[0], questions=())
    actual = tree_model.tree_batch_features([empty, batch[0], empty])
    assert actual[0] == actual[2] == []
    assert_features(expected[0], actual[1])
    assert tree_model.tree_batch_features([]) == []
    assert tree_model.tree_batch_features([empty, empty]) == [[], []]
    tree_model.spec.max_padded_tokens = 1
    with pytest.raises(ValueError, match='unique tokens'):
        tree_model.tree_batch_features(batch)


@pytest.mark.parametrize('checkpointing', [False, True])
@pytest.mark.parametrize('lora', [False, True])
def test_batched_tree_gradients_match_sequential(tree_model, encoder, record, checkpointing, lora):
    m = tree_model
    if lora:
        from peft import LoraConfig, get_peft_model
        m.backbone = get_peft_model(m.backbone, LoraConfig(r=2, lora_alpha=4,
                                   target_modules=lora_modules(m.backbone.config)))
        with torch.no_grad():
            for n, p in m.named_parameters():
                if 'lora_B' in n:
                    p.normal_(std=0.03)
    m.train()
    m.tree_checkpointing = checkpointing
    batch = records(encoder, record)
    # Batched parents cross chunk boundaries and include a one-token root.
    batch[0] = replace(batch[0], state=batch[0].state * 4)
    batch[2] = replace(batch[2], state=batch[2].state[:1])
    m.spec.rows_per_forward = 3

    def run(mode):
        m.zero_grad(set_to_none=True)
        m.prefix_execution = mode
        logits = m(batch)
        loss = sum(record_loss(z, r) for z, r in zip(logits, batch, strict=True)) / len(batch)
        loss.backward()
        return loss.detach(), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}

    old_loss, old = run('tree')
    new_loss, new = run('tree-batched')
    torch.testing.assert_close(old_loss, new_loss, atol=4e-6, rtol=4e-5)
    assert old.keys() == new.keys()
    for name in old:
        torch.testing.assert_close(old[name], new[name], atol=5e-5, rtol=6e-4, msg=name)
    if lora:
        assert any(g.abs().sum() > 0 for n, g in new.items() if 'lora_A' in n)
    else:
        assert m.backbone.get_input_embeddings().weight.grad[batch[0].state[1]].abs().sum() > 0


def test_batch_projects_unique_tokens_together_without_sequential_fallback(tree_model, encoder, record, monkeypatch):
    batch = records(encoder, record)
    calls, handles = {}, []

    def hook(name):
        def track(module, args):
            calls.setdefault(name, []).append(args[0].shape)
        return track

    for name, module in tree_model.backbone.named_modules():
        if name.endswith(('in_proj_qkv', 'gate_proj', 'q_proj', 'o_proj')):
            handles.append(module.register_forward_pre_hook(hook(name)))

    def forbidden(*args, **kwargs):
        raise AssertionError('batched execution must not fall back to record-wise forwards')

    monkeypatch.setattr(tree_model, 'tree_features', forbidden)
    monkeypatch.setattr(tree_model, '_run_rows', forbidden)
    # Batched attention must not allocate a dense all-records token mask.
    monkeypatch.setattr(TreeLayout, 'allow', property(forbidden))
    try:
        tree_model.predict_batch(batch, tree=True)
    finally:
        for handle in handles:
            handle.remove()
    total = sum(map(unique_tokens, batch))
    readouts = sum(1 + len(q.candidates) for r in batch for q in r.questions)
    assert calls
    for name, shapes in calls.items():
        assert len(shapes) == 1, name
        count = shapes[0].numel() // shapes[0][-1]
        expected = readouts if tree_model.joint_layer == 3 and name.startswith('layers.3.') else total
        assert count == expected, name


def test_batched_head_warmup_has_no_backbone_gradient(tree_model, encoder, record):
    m = tree_model.train()
    m.prefix_execution = 'tree-batched'
    m.tree_checkpointing = True
    m.head_only = True
    batch = records(encoder, record)
    sum(record_loss(z, r) for z, r in zip(m(batch), batch, strict=True)).backward()
    assert all(p.grad is None for p in m.backbone.parameters())
    assert m.joint_layer is None or m.joint_gate.grad is None
    assert m.head.project.weight.grad is not None


def test_tree_batched_evaluation_cli_keeps_rejections_and_tail(tmp_path, monkeypatch, tree_model, encoder, record):
    import json
    import sys
    from qev import evaluate

    items = [replace(record, id=f'r{i}', state=('too long ' * 100 if i == 1 else record.state))
             for i in range(5)]
    (tmp_path / 'manifest.json').write_text('{}')
    monkeypatch.setattr(evaluate, 'load_records', lambda *args: (items, {'files': {'dev': {'role': 'development'}}}))
    monkeypatch.setattr(evaluate, 'load_model', lambda *args, **kwargs: (tree_model, None, encoder, {}))
    results = []
    for size in (1, 3):
        out = tmp_path / f'batch{size}'
        monkeypatch.setattr(sys, 'argv', ['evaluate', '--checkpoint', 'unused', '--data', str(tmp_path),
                                         '--split', 'dev', '--out', str(out), '--device', str(tree_model.device),
                                         '--tree', '--batch-size', str(size), '--progress-every', '0'])
        evaluate.main()
        report = json.loads((out / 'report.json').read_text())
        assert report['rejected_questions'] == 2
        assert report['answered_questions'] == 8
        assert report['execution'] == ('tree' if size == 1 else 'pooled_tree')
        results.append([json.loads(line) for line in (out / 'predictions.jsonl').read_text().splitlines()])
    for a, b in zip(*results, strict=True):
        for key in ('record_id', 'question_id', 'label', 'prediction', 'correct'):
            assert a[key] == b[key]
        assert a['probabilities'] == pytest.approx(b['probabilities'], abs=3e-6)
