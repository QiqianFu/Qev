# Adapted for Qev in 2026; see NOTICE and provenance.json.
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import pytest
from safetensors.torch import load_file

from qev.checkpoint import load_model
from qev.data import file_hash
from qev.encoding import Limits
from qev.model import ModelSpec
from qev.train import distributed_indices, resume_cursor
from conftest import tiny_backbone


def training_assets(tmp_path, tokenizer, record):
    base = tmp_path / "base"
    torch.manual_seed(6)
    tiny_backbone(len(tokenizer)).save_pretrained(base)
    tokenizer.save_pretrained(base)
    data = tmp_path / "data"
    data.mkdir()
    train = data / "train.jsonl"
    train.write_text("".join(json.dumps(replace(record, id=f"r{i}", group_id=f"g{i}").to_dict()) + "\n" for i in range(4)))
    (data / "manifest.json").write_text(json.dumps({"files": {
        "train": {"file": "train.jsonl", "sha256": file_hash(train), "records": 4, "role": "train"}}}))
    spec = ModelSpec(str(base), head_dim=32, head_heads=4, head_layers=1, lora_rank=2,
                     weights_dtype="fp32", attention="eager", rows_per_forward=2)
    cfg = {"model": asdict(spec), "limits": asdict(Limits(128, 128, 128, 384, 32)),
           "training": {"epochs": 2, "batch_size": 1, "accum": 2, "seed": 7,
                        "lr": 0.001, "head_lr": 0.002, "warmup_steps": 0,
                        "head_warmup_steps": 1, "gradient_checkpointing": False, "save_every": 1,
                        "autocast": "fp32"}}
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    return data, config


def run_training(data, config, out, *extra, distributed=False):
    cmd = [sys.executable]
    if distributed:
        cmd += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", "-m", "qev.train"]
    else:
        cmd += ["-m", "qev.train"]
    cmd += ["--config", str(config), "--data", str(data), "--out", str(out), "--device", "cpu", *extra]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", USE_TF="0")
    p = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
    assert p.returncode == 0, p.stdout


def test_training_resume_matches_uninterrupted(tmp_path, tokenizer, record):
    data, config = training_assets(tmp_path, tokenizer, record)
    uninterrupted, resumed = tmp_path / "full", tmp_path / "resume"
    run_training(data, config, uninterrupted)
    run_training(data, config, resumed, "--max-steps", "1")
    run_training(data, config, resumed, "--resume", str(resumed / "step-000001"))
    for rel in ["head.safetensors", "adapter/adapter_model.safetensors"]:
        a = load_file(str(uninterrupted / "step-000004" / rel))
        b = load_file(str(resumed / "step-000004" / rel))
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    model, _, encoder, _ = load_model(resumed / "step-000004")
    model.eval()
    for p in model.predict(encoder(record)):
        assert torch.isfinite(p).all()
        torch.testing.assert_close(p.sum(), torch.tensor(1.0))


def test_two_rank_ddp_head_warmup_and_update(tmp_path, tokenizer, record):
    data, config = training_assets(tmp_path, tokenizer, record)
    out = tmp_path / "ddp"
    run_training(data, config, out, distributed=True)
    state = torch.load(out / "step-000002/training.pt", weights_only=False)
    assert state["world_size"] == 2 and len(state["rng_by_rank"]) == 2
    assert state["global_step"] == 2
    logs = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    assert all(row["forward_tokens_global"] == 2 * row["forward_tokens_rank0"] for row in logs)
    assert all(row["memory_scope"] == "maximum_across_ranks" for row in logs)
    assert [row["ddp_find_unused_parameters"] for row in logs] == [True, False]


def test_sampler_equal_lengths_and_epoch_determinism():
    a = distributed_indices(5, 0, 17, 0, 2)
    b = distributed_indices(5, 0, 17, 1, 2)
    assert len(a) == len(b) == 3
    assert set(a + b) == set(range(5))
    assert a == distributed_indices(5, 0, 17, 0, 2)


def test_repartition_preserves_optimizer_batches_and_tail():
    n, epoch, seed = 10, 1, 17
    old = {"batch_size": 2, "accum": 2, "seed": seed, "epochs": 2}
    new = {**old, "accum": 1}
    state = {"training_settings": old, "world_size": 1, "epoch": epoch,
             "next_batch": 2, "global_step": 4}
    cursor, report = resume_cursor(state, new, 2, n, True)
    assert cursor == 1 and report["steps_per_epoch"] == 3
    single = distributed_indices(n, epoch, seed, 0, 1)
    ranks = [distributed_indices(n, epoch, seed, r, 2) for r in range(2)]
    for step in range(3):
        assert sorted(single[step * 4:(step + 1) * 4]) == sorted(
            i for rank in ranks for i in rank[step * 2:(step + 1) * 2])
    with pytest.raises(ValueError, match="explicit repartition"):
        resume_cursor(state, new, 2, n)
    with pytest.raises(ValueError, match="global batch"):
        resume_cursor(state, old, 2, n, True)
    with pytest.raises(ValueError, match="padding"):
        resume_cursor(state, new, 2, 11, True)
    with pytest.raises(ValueError, match="only batch_size"):
        resume_cursor(state, {**new, "seed": 18}, 2, n, True)
    with pytest.raises(ValueError, match="optimizer boundary"):
        resume_cursor({**state, "next_batch": 1}, new, 2, n, True)


def test_single_to_two_rank_resume_preserves_updates(tmp_path, tokenizer, record):
    data, config = training_assets(tmp_path, tokenizer, record)
    single, parallel = tmp_path / "single", tmp_path / "parallel"
    run_training(data, config, single)
    cfg = json.loads(config.read_text())
    cfg["training"]["accum"] = 1
    repartitioned = tmp_path / "two.json"
    repartitioned.write_text(json.dumps(cfg))
    run_training(data, repartitioned, parallel, "--resume", str(single / "step-000001"),
                 "--allow-repartition", distributed=True)
    state = torch.load(parallel / "step-000004/training.pt", weights_only=False)
    assert state["epoch"] == 2 and state["next_batch"] == 0 and state["world_size"] == 2
    migration = json.loads((parallel / "resume-migration.json").read_text())
    assert migration["global_step"] == 1 and migration["new_next_batch"] == 1
    for rel in ["head.safetensors", "adapter/adapter_model.safetensors"]:
        a = load_file(str(single / "step-000004" / rel))
        b = load_file(str(parallel / "step-000004" / rel))
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=2e-6, rtol=2e-5)
    original = torch.load(single / "step-000004/training.pt", weights_only=False)
    for key, opt in original["optimizer"]["state"].items():
        for name, tensor in opt.items():
            torch.testing.assert_close(tensor, state["optimizer"]["state"][key][name], atol=2e-6, rtol=2e-5)


def test_balanced_ddp_matches_single_rank_and_resumes(tmp_path, tokenizer, record):
    data, config = training_assets(tmp_path, tokenizer, record)
    train = data / "train.jsonl"
    # Different lengths make rank reassignment observable.
    rows = [replace(record, id=f"r{i}", group_id=f"g{i}", state=("Extra evidence. " * i) + record.state)
            for i in range(4)]
    train.write_text("".join(json.dumps(r.to_dict()) + "\n" for r in rows))
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["files"]["train"]["sha256"] = file_hash(train)
    (data / "manifest.json").write_text(json.dumps(manifest))
    cfg = json.loads(config.read_text())
    cfg["training"].update(batch_size=2, accum=2)
    config.write_text(json.dumps(cfg))
    single, balanced, resumed = tmp_path / "single", tmp_path / "balanced", tmp_path / "resumed"
    run_training(data, config, single)
    cfg["training"].update(accum=1, batch_assignment="leaf-balanced-v1")
    config.write_text(json.dumps(cfg))
    run_training(data, config, balanced, distributed=True)
    run_training(data, config, resumed, "--max-steps", "1", distributed=True)
    run_training(data, config, resumed, "--resume", str(resumed / "step-000001"), distributed=True)
    for rel in ["head.safetensors", "adapter/adapter_model.safetensors"]:
        expected = load_file(str(single / "step-000002" / rel))
        actual = load_file(str(balanced / "step-000002" / rel))
        continued = load_file(str(resumed / "step-000002" / rel))
        for key in expected:
            torch.testing.assert_close(actual[key], continued[key], atol=0, rtol=0)
            if key == "scorer.3.bias":
                # A common shift of all candidate logits also cancels in softmax.
                continue
            if key.endswith("self_attn.in_proj_bias"):
                # Key bias cancels out of softmax attention. Its true gradient is
                # zero; Adam amplifies tiny reduction-order roundoff. Check Q/V
                # weights here, and all optimizer moments and predictions below.
                d = actual[key].numel() // 3
                keep = torch.cat([torch.arange(d), torch.arange(2 * d, 3 * d)])
                torch.testing.assert_close(actual[key][keep], expected[key][keep], atol=2e-6, rtol=2e-5)
            else:
                torch.testing.assert_close(actual[key], expected[key], atol=2e-6, rtol=2e-5)
    a = torch.load(single / "step-000002/training.pt", weights_only=False)
    b = torch.load(balanced / "step-000002/training.pt", weights_only=False)
    for index, state in a["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, b["optimizer"]["state"][index][key], atol=2e-7, rtol=2e-5)
    one, _, enc, _ = load_model(single / "step-000002")
    two, _, _, _ = load_model(balanced / "step-000002")
    one.eval(); two.eval()
    for row in rows:
        for left, right in zip(one.predict(enc(row)), two.predict(enc(row)), strict=True):
            torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)


def test_assignment_migration_requires_explicit_opt_in():
    old = {"batch_size": 2, "accum": 1}
    new = {**old, "batch_assignment": "leaf-balanced-v1"}
    state = {"training_settings": old, "world_size": 2, "epoch": 0, "next_batch": 1, "global_step": 1}
    with pytest.raises(ValueError, match="explicit repartition"):
        resume_cursor(state, new, 2, 12)
    cursor, migration = resume_cursor(state, new, 2, 12, True)
    assert cursor == 1 and migration["new_batch_assignment"] == "leaf-balanced-v1"


def test_shared_prefix_ddp_checkpointing_and_resume(tmp_path, tokenizer, record):
    data, config = training_assets(tmp_path, tokenizer, record)
    cfg = json.loads(config.read_text())
    cfg["training"].update(prefix_execution="shared-prefix", gradient_checkpointing=True,
                           batch_assignment="leaf-balanced-v1")
    config.write_text(json.dumps(cfg))
    full, resume = tmp_path / "full", tmp_path / "resume"
    run_training(data, config, full, distributed=True)
    run_training(data, config, resume, "--max-steps", "1", distributed=True)
    run_training(data, config, resume, "--resume", str(resume / "step-000001"), distributed=True)
    for rel in ["head.safetensors", "adapter/adapter_model.safetensors"]:
        a = load_file(str(full / "step-000002" / rel))
        b = load_file(str(resume / "step-000002" / rel))
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)


def test_none_insert_exempt_sources_keep_records_as_provided(tmp_path, tokenizer, record):
    data, config = training_assets(tmp_path, tokenizer, record)
    cfg = json.loads(config.read_text())
    cfg["training"].update(none_insert_prob=0.9, none_insert_absent_frac=0.5)
    config.write_text(json.dumps(cfg))

    def inserted(out, exempt=None):
        if exempt is not None:
            cfg["training"]["none_insert_exempt_sources"] = exempt
            config.write_text(json.dumps(cfg))
        run_training(data, config, out)
        return sum(json.loads(line)["none_inserted_global"] for line in (out / "train.jsonl").read_text().splitlines())

    assert inserted(tmp_path / "default") > 0  # record source is "synthetic"
    assert inserted(tmp_path / "other", ["jev_distill/"]) > 0
    assert inserted(tmp_path / "exempt", ["synth"]) == 0
    cfg["training"]["none_insert_exempt_sources"] = [""]
    config.write_text(json.dumps(cfg))
    cmd = [sys.executable, "-m", "qev.train", "--config", str(config), "--data", str(data),
           "--out", str(tmp_path / "bad"), "--device", "cpu"]
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
    assert p.returncode != 0 and "none_insert_exempt_sources" in p.stdout


def test_late_split_is_mixed_into_final_steps_only(tmp_path, tokenizer, record):
    data, config = training_assets(tmp_path, tokenizer, record)
    late = data / "late.jsonl"
    late.write_text("".join(json.dumps(replace(record, id=f"late{i}", group_id=f"lg{i}").to_dict()) + "\n"
                            for i in range(2)))
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["files"]["late_train"] = {"file": "late.jsonl", "sha256": file_hash(late), "records": 2, "role": "train"}
    (data / "manifest.json").write_text(json.dumps(manifest))
    cfg = json.loads(config.read_text())
    cfg["training"].update(late_split="late_train", late_fraction=0.5)
    config.write_text(json.dumps(cfg))
    out = tmp_path / "late"
    run_training(data, config, out)
    mix = json.loads((out / "late-mix.json").read_text())
    # 4 main records, batch 1 x accum 2: 2 steps per epoch; the final epoch gains one step for the 2 late records.
    assert mix == {"split": "late_train", "fraction": 0.5, "records": 2, "start_step": 2, "epoch": 1,
                   "global_position": 0, "total_steps": 5}
    assert json.loads((out / "admission.json").read_text())["late_admitted"] == 2
    assert [json.loads(line)["step"] for line in (out / "train.jsonl").read_text().splitlines()] == [1, 2, 3, 4, 5]
    cfg["training"]["late_repeats"] = 3
    config.write_text(json.dumps(cfg))
    run_training(data, config, tmp_path / "late3")
    mix = json.loads((tmp_path / "late3" / "late-mix.json").read_text())
    assert mix["repeats"] == 3 and mix["total_steps"] == 7  # final epoch: (4 + 6) records / 2 per step


def test_layer_lr_decay_groups_cover_backbone_once(tokenizer):
    from conftest import tiny_backbone
    from qev.train import backbone_lr_scales
    backbone = tiny_backbone(len(tokenizer), hybrid=True)
    groups = backbone_lr_scales(backbone, 0.5)
    ids = [id(p) for ps, _ in groups for p in ps]
    assert sorted(ids) == sorted(id(p) for p in backbone.parameters()) and len(ids) == len(set(ids))
    scale = {id(p): s for ps, s in groups for p in ps}
    assert [s for _, s in groups] == [1.0, 0.5, 0.25]  # output side first
    assert {scale[id(p)] for p in backbone.layers[1].parameters()} == {1.0}
    assert {scale[id(p)] for p in backbone.layers[0].parameters()} == {0.5}
    assert scale[id(backbone.embed_tokens.weight)] == 0.25 and scale[id(backbone.norm.weight)] == 1.0
    assert [s for _, s in backbone_lr_scales(backbone, 1.0)] == [1.0]
    with pytest.raises(ValueError):
        backbone_lr_scales(backbone, 1.5)
