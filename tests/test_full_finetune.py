# Adapted for Qev in 2026; see NOTICE and provenance.json.
"""FSDP2 full-parameter training: one sharded step equals a hand-computed single-process step.

Needs at least two CUDA GPUs (run on the cluster). FP32 end to end so the comparison is tight.
"""
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file

from qev.batching import rank_indices
from qev.checkpoint import load_model
from qev.data import file_hash, load_records
from qev.encoding import Encoder, Limits
from qev.model import QevModel, ModelSpec, load_backbone, record_loss
from qev.train import backbone_lr_scales, lr_factor
from conftest import tiny_backbone

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA GPUs")


def varied(record, i):
    """Different lengths and question counts, so ranks make different numbers of head calls."""
    return replace(record, id=f"r{i}", group_id=f"g{i}", state=record.state + " It was noted again." * (i % 3),
                   questions=record.questions if i % 2 == 0 else record.questions[:1])


def assets(tmp_path, tokenizer, record, *, checkpointing, dtype="fp32", steps_extra=None, max_padded_tokens=8192):
    base = tmp_path / "base"
    torch.manual_seed(6)
    tiny_backbone(len(tokenizer), hybrid=True).save_pretrained(base)
    tokenizer.save_pretrained(base)
    data = tmp_path / "data"
    data.mkdir()
    train = data / "train.jsonl"
    train.write_text("".join(json.dumps(varied(record, i).to_dict()) + "\n" for i in range(8)))
    dev = data / "dev.jsonl"  # odd size: one rank needs a padding batch during inline evaluation
    dev.write_text("".join(json.dumps(replace(varied(record, i), id=f"d{i}", group_id=f"dg{i}").to_dict()) + "\n"
                           for i in range(5)))
    (data / "manifest.json").write_text(json.dumps({"files": {
        "train": {"file": "train.jsonl", "sha256": file_hash(train), "records": 8, "role": "train"},
        "dev": {"file": "dev.jsonl", "sha256": file_hash(dev), "records": 5, "role": "development"}}}))
    spec = ModelSpec(str(base), head_dim=32, head_heads=4, head_layers=1, lora_rank=0, weights_dtype=dtype,
                     attention="sdpa", rows_per_forward=4, candidate_interaction="last-full-attention",
                     max_padded_tokens=max_padded_tokens)
    training = {"epochs": 1, "batch_size": 2, "accum": 1, "seed": 7, "lr": 1e-3, "head_lr": 2e-3,
                "joint_gate_lr": 0.05, "warmup_steps": 0, "head_warmup_steps": 0, "weight_decay": 0.01,
                "gradient_checkpointing": checkpointing, "save_every": 1, "autocast": dtype,
                "prefix_execution": "tree-batched", "batch_assignment": "leaf-balanced-v1", "full_finetune": True,
                "eval_every": 1, "eval_batch_size": 2, "eval_sets": [{"data": str(data), "split": "dev"}]}
    training.update(steps_extra or {})
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": asdict(spec), "limits": asdict(Limits(128, 128, 128, 384, 32)),
                                  "training": training}))
    return data, config, spec, training


def run(data, config, out, *extra):
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
           "-m", "qev.train", "--config", str(config), "--data", str(data), "--out", str(out), *extra]
    env = dict(os.environ, OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", USE_TF="0", CUDA_VISIBLE_DEVICES="0,1")
    p = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    assert p.returncode == 0, p.stdout[-6000:]


@pytest.mark.parametrize("checkpointing,offload,decay", [(False, True, 1.0), (True, True, 1.0), (False, False, 1.0),
                                                         (False, True, 0.5)],
                         ids=["plain", "checkpointed", "gpu-resident", "layer-lr-decay"])
def test_fsdp_full_step_matches_single_process(tmp_path, tokenizer, record, checkpointing, offload, decay):
    data, config, spec, training = assets(tmp_path, tokenizer, record, checkpointing=checkpointing,
                                          steps_extra={"fsdp_cpu_offload": offload, "layer_lr_decay": decay})
    run(data, config, tmp_path / "out", "--max-steps", "1")
    ckpt = tmp_path / "out" / "step-000001"
    meta = json.loads((ckpt / "model.json").read_text())
    assert meta["backbone"] == "full" and meta["adapter"] is False

    # Reference: identical initialisation order to qev.train, then one global AdamW step by hand.
    device = torch.device("cuda", 0)
    random.seed(training["seed"])
    torch.manual_seed(training["seed"])
    records, _ = load_records(data, "train", training=True)
    model = QevModel(load_backbone(spec, torch.device("cpu")), spec, tokenizer.pad_token_id).float().to(device)
    model.backbone.requires_grad_(True)
    model.prefix_execution = "tree-batched"
    model.tree_checkpointing = checkpointing
    model.train()
    encoder = Encoder(tokenizer, Limits(128, 128, 128, 384, 32), choice_none_policy=spec.choice_none_policy)
    encoded = [encoder(r) for r in records]
    costs = [r.forward_tokens for r in encoded]
    batches = [rank_indices(len(encoded), 0, training["seed"], rank, 2, batch_size=2, accum=1,
                            assignment="leaf-balanced-v1", costs=costs)[:2] for rank in (0, 1)]
    # The ranks must differ in question count, the case that deadlocked a per-call head unit.
    assert len({sum(len(encoded[i].questions) for i in ids) for ids in batches}) == 2
    # Each rank averages over its own records; FSDP averages gradients across ranks.
    loss = sum(sum(record_loss(z, encoded[i]) for z, i in zip(model([encoded[i] for i in ids]), ids)) / len(ids)
               for ids in batches) / 2
    loss.backward()
    groups = [{"params": ps, "lr": training["lr"] * scale} for ps, scale in backbone_lr_scales(model.backbone, decay)]
    groups += [{"params": list(model.head.parameters()), "lr": training["head_lr"]},
              {"params": [model.joint_gate], "lr": training["joint_gate_lr"], "weight_decay": 0.0}]
    torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
    total = math.ceil(math.ceil(8 / 2) / 2)
    for g in groups:
        g["lr"] *= lr_factor(0, total, 0)
    torch.optim.AdamW(groups, weight_decay=0.01).step()

    saved = load_file(str(ckpt / "backbone" / "model.safetensors"))
    reference = {k: v.detach().cpu() for k, v in model.backbone.state_dict().items()}
    assert set(saved) == set(reference)
    moved, mismatched = 0, {}
    initial = load_file(str(tmp_path / "base" / "model.safetensors"))
    for key, value in reference.items():
        if not torch.allclose(saved[key], value, atol=2e-5, rtol=1e-4):
            mismatched[key] = float((saved[key] - value).abs().max())
        moved += int(not torch.equal(value, initial[key]))
    assert not mismatched, mismatched
    assert moved > len(reference) // 2, "full fine-tuning must update most backbone tensors"
    for key, value in load_file(str(ckpt / "head.safetensors")).items():
        torch.testing.assert_close(value, model.head.state_dict()[key].cpu(), atol=2e-5, rtol=1e-4, msg=key)
    torch.testing.assert_close(load_file(str(ckpt / "joint.safetensors"))["gate"], model.joint_gate.detach().cpu(),
                               atol=2e-5, rtol=1e-4)

    # Inline sharded evaluation after the step equals the reference model's predictions.
    model.eval()
    inline = [json.loads(l) for l in (tmp_path / "evaluation-inline" / "step-000001" / "dev" / "predictions.jsonl").read_text().splitlines()]
    dev_records, _ = load_records(data, "dev")
    assert sorted({r["record_id"] for r in inline}) == sorted(r.id for r in dev_records)
    assert len(inline) == sum(len(r.questions) for r in dev_records)
    with torch.no_grad():
        expected = {}
        for r in dev_records:
            for q, p in zip(r.questions, model.predict_batch([encoder(r)], tree=True)[0]):
                expected[(r.id, q.id)] = p.cpu()
    for row in inline:
        torch.testing.assert_close(torch.tensor(list(row["probabilities"].values())), expected[(row["record_id"], row["question_id"])],
                                   atol=1e-4, rtol=1e-4)
    report = json.loads((tmp_path / "evaluation-inline" / "step-000001" / "dev" / "report.json").read_text())
    assert report["answered_questions"] == len(inline) and report["execution"] == "inline_fsdp_tree_batched"

    # The checkpoint reloads through the normal evaluation loader and predicts like the reference.
    loaded, _, loaded_encoder, _ = load_model(ckpt, device)
    loaded.eval()
    model.eval()
    loaded.prefix_execution = model.prefix_execution = "tree"
    with torch.no_grad():
        for a, b in zip(loaded.predict(loaded_encoder(record)), model.predict(encoder(record))):
            torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(a.sum(), torch.tensor(1.0, device=a.device))


def test_fsdp_microbatch_over_budget_raises_instead_of_splitting(tmp_path, tokenizer, record):
    data, config, spec, training = assets(tmp_path, tokenizer, record, checkpointing=True, max_padded_tokens=64)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", "-m", "qev.train",
           "--config", str(config), "--data", str(data), "--out", str(tmp_path / "out"), "--max-steps", "1"]
    p = subprocess.run(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1"), text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    assert p.returncode != 0 and "FSDP microbatch has" in p.stdout


def test_fsdp_bf16_head_warmup_then_backbone_warmup(tmp_path, tokenizer, record):
    data, config, spec, training = assets(tmp_path, tokenizer, record, checkpointing=True, dtype="bf16",
                                          steps_extra={"epochs": 2, "head_warmup_steps": 1,
                                                       "backbone_warmup_steps": 2, "save_every": 0})
    run(data, config, tmp_path / "out")
    logs = [json.loads(line) for line in (tmp_path / "out" / "train.jsonl").read_text().splitlines()]
    assert [r["head_only"] for r in logs] == [True, False, False, False]
    assert all(r["full_finetune"] and math.isfinite(r["loss_global_mean"]) for r in logs)
    final = tmp_path / "out" / json.loads((tmp_path / "out" / "latest.json").read_text())["checkpoint"]
    saved = load_file(str(final / "backbone" / "model.safetensors"))
    assert all(v.dtype == torch.bfloat16 for v in saved.values())
    base = load_file(str(tmp_path / "base" / "model.safetensors"))
    assert any(not torch.equal(saved[k].float(), base[k].to(torch.bfloat16).float()) for k in saved)
