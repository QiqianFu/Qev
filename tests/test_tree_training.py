# Adapted for Qev in 2026; see NOTICE and provenance.json.
"""Tree sharing must preserve losses, prefix gradients, updates and replay."""
from dataclasses import replace
import json
import math

import pytest
import torch

from qev.model import lora_modules, record_loss
from test_tree_inference import tree_model
from test_training import training_assets, run_training


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("lora", [False, True])
def test_tree_gradients_and_updates_match_full_paths(tree_model, encoder, record, checkpointing, lora):
    model = tree_model
    if lora:
        from peft import LoraConfig, get_peft_model
        model.backbone = get_peft_model(model.backbone, LoraConfig(
            task_type="FEATURE_EXTRACTION", r=2, lora_alpha=4,
            target_modules=lora_modules(model.backbone.config)))
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "lora_B" in name:
                    p.normal_(std=0.03)
    model.train()
    model.spec.rows_per_forward = 2
    records = [encoder(record), encoder(replace(record, id="other", state="More context. " + record.state))]
    # Cross recurrent chunk boundaries and exercise a single-token child.
    records[0] = replace(records[0], state=records[0].state * 4)
    q = records[1].questions[0]
    records[1] = replace(records[1], questions=(replace(q, candidates=(q.candidates[0][:1],)),
                                               records[1].questions[1]))
    # Targets must match the edited single-candidate question.
    q = records[1].questions[0]
    q = replace(q, question=replace(q.question, candidates=q.question.candidates[:1], target=(1.0,)))
    records[1] = replace(records[1], questions=(q, records[1].questions[1]))
    initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def run():
        model.zero_grad(set_to_none=True)
        logits = model(records)
        loss = sum(record_loss(z, r) for z, r in zip(logits, records, strict=True)) / len(records)
        loss.backward()
        gradients = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        return [[z.detach() for z in item] for item in logits], loss.detach(), gradients

    baseline, old_loss, old_grads = run()
    model.prefix_execution = "tree"
    model.tree_checkpointing = checkpointing
    actual, new_loss, grads = run()
    torch.testing.assert_close(new_loss, old_loss, atol=3e-6, rtol=3e-5)
    for a, b in zip(actual, baseline, strict=True):
        for x, y in zip(a, b, strict=True):
            torch.testing.assert_close(x, y, atol=3e-5, rtol=3e-5)
    assert grads.keys() == old_grads.keys()
    for name in grads:
        torch.testing.assert_close(grads[name], old_grads[name], atol=4e-5, rtol=5e-4, msg=name)
    if lora:
        assert any(g.abs().sum() > 0 for n, g in grads.items() if "lora_A" in n)
    else:
        embedding = model.backbone.get_input_embeddings().weight
        assert embedding.grad[records[0].state[1]].abs().sum() > 0
    if model.joint_layer is not None:
        assert model.joint_gate.grad.abs().sum() > 0
    torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01).step()
    for name, p in model.named_parameters():
        if name in grads:
            torch.testing.assert_close(p, initial[name] - 0.01 * old_grads[name], atol=3e-6, rtol=3e-5)


def test_tree_head_warmup_and_checkpoint_guard(tree_model, encoder, record):
    model = tree_model.train()
    model.prefix_execution = "tree"
    model.tree_checkpointing = True
    model.head_only = True
    e = encoder(record)
    record_loss(model([e])[0], e).backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert model.joint_layer is None or model.joint_gate.grad is None
    assert model.head.project.weight.grad is not None
    model.head_only = False
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    with pytest.raises(ValueError, match="tree_checkpointing"):
        model([e])


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("execution", ["tree", "tree-batched"])
def test_tree_train_cli_resume_and_ddp(tmp_path, tokenizer, record, distributed, execution):
    from conftest import tiny_backbone
    from safetensors.torch import load_file
    data, cfg_path = training_assets(tmp_path, tokenizer, record)
    cfg = json.loads(cfg_path.read_text())
    tiny_backbone(len(tokenizer), hybrid=True).save_pretrained(cfg['model']['base'])
    cfg['model']['candidate_interaction'] = 'last-full-attention'
    cfg['training'].update(prefix_execution=execution, gradient_checkpointing=True, joint_gate_lr=0.01)
    if execution == 'tree-batched':
        cfg['training'].update(batch_size=2, accum=1)
    cfg_path.write_text(json.dumps(cfg))
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    run_training(data, cfg_path, full, distributed=distributed)
    run_training(data, cfg_path, resumed, '--max-steps', '1', distributed=distributed)
    run_training(data, cfg_path, resumed, '--resume', str(resumed/'step-000001'), distributed=distributed)
    last = json.loads((full/'latest.json').read_text())['checkpoint']
    for name in ('head.safetensors', 'joint.safetensors', 'adapter/adapter_model.safetensors'):
        a, b = load_file(str(full/last/name)), load_file(str(resumed/last/name))
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    logs = [json.loads(line) for line in (full/'train.jsonl').read_text().splitlines()]
    assert logs[0]['head_only'] and not logs[-1]['head_only']
    assert all(row['readout_precision'] == 'fp32-head-and-joint-reductions.v1' for row in logs)
    assert json.loads((full/last/'model.json').read_text())['readout_precision'] == logs[-1]['readout_precision']


@pytest.mark.parametrize("interaction,head_layers", [("none", 1), ("last-full-attention", 0), ("none", 0),
                                                    ("last-readout-cross", 1)])
def test_interaction_ablations_train_tree_batched_and_reload(tmp_path, tokenizer, record, interaction, head_layers):
    from conftest import tiny_backbone
    from qev.checkpoint import load_model
    data, cfg_path = training_assets(tmp_path, tokenizer, record)
    cfg = json.loads(cfg_path.read_text())
    tiny_backbone(len(tokenizer), hybrid=True).save_pretrained(cfg['model']['base'])
    cfg['model'].update(candidate_interaction=interaction, head_layers=head_layers)
    cfg['training'].update(prefix_execution='tree-batched', gradient_checkpointing=True, batch_size=2, accum=1)
    if interaction != 'none':
        cfg['training']['joint_gate_lr'] = 0.01
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / 'out'
    run_training(data, cfg_path, out)
    logs = [json.loads(line) for line in (out / 'train.jsonl').read_text().splitlines()]
    assert all(math.isfinite(row['loss_global_mean']) for row in logs) and not logs[-1]['head_only']
    last = json.loads((out / 'latest.json').read_text())['checkpoint']
    model, _, encoder, _ = load_model(out / last, torch.device('cpu'))
    assert model.spec.head_layers == head_layers and model.spec.candidate_interaction == interaction
    assert len(model.head.layers) == head_layers and (model.joint_layer is None) == (interaction == 'none')
    with torch.no_grad():
        logits = model([encoder(record)])
    assert all(torch.isfinite(z).all() for group in logits for z in (group if isinstance(group, (list, tuple)) else [group]))
