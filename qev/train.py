# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Single-GPU or torchrun DDP training; no model generation or paid service."""
import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .augment import NoneInserter, none_absent_view
from .batching import ASSIGNMENTS, rank_indices
from .checkpoint import load_model, rng_state, restore_rng, save_full_model, save_model
from .data import file_hash, load_records, write_json
from .encoding import ContextOverflow, Encoder, Limits
from .execution import configure_checkpointing
from .model import QevModel, ModelSpec, _base_model, load_backbone, record_loss


def distributed_indices(n, epoch, seed, rank, world):
    """Equal rank lengths; deterministic padded training sampler, never used for eval."""
    return rank_indices(n, epoch, seed, rank, world)


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def backbone_lr_scales(backbone, decay):
    """(params, lr scale) per trainable backbone group, output side first.

    decay = 1 keeps one group at scale 1. Otherwise decoder layer i of L gets
    decay ** (L - 1 - i); embeddings sit below layer 0 (decay ** L); the final
    norm and anything else trainable stay at the top rate.
    """
    trainable = [(n, p) for n, p in backbone.named_parameters() if p.requires_grad]
    if decay == 1.0:
        return [([p for _, p in trainable], 1.0)] if trainable else []
    if not 0 < decay < 1:
        raise ValueError("layer_lr_decay must be in (0, 1]")
    layers = _base_model(backbone).layers
    depth = {id(p): i for i, layer in enumerate(layers) for p in layer.parameters()}
    buckets = {}
    for name, p in trainable:
        if id(p) in depth:
            exponent = len(layers) - 1 - depth[id(p)]
        else:
            exponent = len(layers) if "embed" in name else 0
        buckets.setdefault(exponent, []).append(p)
    return [(ps, decay ** e) for e, ps in sorted(buckets.items())]


def resume_cursor(state, settings, world, n, allow_repartition=False):
    """Repartition complete optimizer batches only; never change their membership."""
    old, old_world = state["training_settings"], state["world_size"]
    if old_world == world and old == settings:
        return state["next_batch"], None
    if not allow_repartition:
        raise ValueError("resume requires identical world size and training settings; explicit repartition required")
    def unchanged(values):
        result = {k: v for k, v in values.items() if k not in {"batch_size", "accum", "batch_assignment"}}
        result.setdefault("prefix_execution", "leaf-rows")
        return result
    if unchanged(old) != unchanged(settings):
        raise ValueError("repartition may change only batch_size, accum and batch_assignment")
    if old.get("batch_assignment", "strided-v1") not in ASSIGNMENTS or settings.get("batch_assignment", "strided-v1") not in ASSIGNMENTS:
        raise ValueError("unknown batch assignment in checkpoint or requested settings")
    ob, oa = int(old.get("batch_size", 1)), int(old.get("accum", 8))
    nb, na = int(settings.get("batch_size", 1)), int(settings.get("accum", 8))
    global_batch = ob * oa * old_world
    if min(ob, oa, nb, na, old_world, world) < 1 or global_batch != nb * na * world:
        raise ValueError("repartition requires an unchanged positive global batch")
    if n % old_world or n % world:
        raise ValueError("repartition requires no sampler padding in either world size")
    steps = math.ceil(n / global_batch)
    cursor = int(state["next_batch"])
    if cursor % oa or cursor < 0 or cursor >= math.ceil(n / old_world / ob):
        raise ValueError("checkpoint cursor is not a normalized optimizer boundary")
    offset = cursor // oa
    if state["global_step"] != state["epoch"] * steps + offset:
        raise ValueError("checkpoint step does not match its epoch and optimizer cursor")
    return offset * na, {"old_world_size": old_world, "new_world_size": world,
                         "global_batch": global_batch, "steps_per_epoch": steps,
                         "old_next_batch": cursor, "new_next_batch": offset * na,
                         "epoch": state["epoch"], "global_step": state["global_step"],
                         "old_batch_assignment": old.get("batch_assignment", "strided-v1"),
                         "new_batch_assignment": settings.get("batch_assignment", "strided-v1"),
                         "rng": "reseeded per rank; dropout must be zero",
                         "numerics": "same samples and objective; floating-point reduction order may differ"}


def require_deterministic_repartition(model):
    if model.spec.lora_dropout or getattr(model.backbone.config, "attention_dropout", 0):
        raise ValueError("repartition requires zero LoRA and attention dropout")
    for module in model.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p:
            raise ValueError("repartition requires zero dropout in every module")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="canonical data directory created by qev.prepare")
    ap.add_argument("--out", required=True)
    initialization = ap.add_mutually_exclusive_group()
    initialization.add_argument("--resume", help="Qev step checkpoint including training.pt")
    initialization.add_argument("--init-checkpoint", help="initialize LoRA/head from a checkpoint on new data; reset optimizer and schedule")
    ap.add_argument("--allow-repartition", action="store_true", help="explicitly resume with a new rank/microbatch partition and unchanged global optimizer batches; zero dropout only")
    ap.add_argument("--max-steps", type=int, help="end this invocation after this many total optimizer steps")
    ap.add_argument("--wall-hours", type=float, help="stop at a completed optimizer step and save")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    config = json.loads(Path(a.config).read_text())
    settings = config["training"]
    seed = int(settings.get("seed", 17))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if a.device.startswith("cuda"):
        if not torch.cuda.is_available():
            ap.error("CUDA requested but unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(a.device)
    full_ft = bool(settings.get("full_finetune", False))
    if full_ft and a.init_checkpoint:
        raise ValueError("--init-checkpoint currently supports LoRA training only")
    if full_ft:
        if device.type != "cuda" or a.resume:
            raise ValueError("full fine-tuning runs on CUDA with torchrun and does not support --resume")
        # CPU-offloaded shards need a CPU collective backend next to NCCL.
        dist.init_process_group("cpu:gloo,cuda:nccl")
    elif world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    out = Path(a.out)
    creation_error = [None]
    if rank == 0:
        try:
            if out.exists() and not a.resume:
                raise FileExistsError(f"refusing to overwrite {out}; use a new directory or --resume")
            out.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            creation_error[0] = str(exc)
    if world > 1:
        dist.broadcast_object_list(creation_error, src=0)
    if creation_error[0]:
        raise FileExistsError(creation_error[0])
    random.seed(seed)
    torch.manual_seed(seed)
    records, data_manifest = load_records(a.data, "train", training=True)
    # Optional second training split mixed only into the final late_fraction of optimizer steps.
    late_split, late_fraction = settings.get("late_split"), float(settings.get("late_fraction", 0.2))
    late_records = load_records(a.data, late_split, training=True)[0] if late_split else []
    late_repeats = int(settings.get("late_repeats", 1))
    if late_split and not 0 < late_fraction < 1:
        raise ValueError("late_fraction must be in (0, 1)")
    if late_repeats < 1:
        raise ValueError("late_repeats must be positive")
    data_hash = file_hash(Path(a.data) / "manifest.json")
    data_files = {split: file_hash(Path(a.data) / entry["file"])
                  for split, entry in data_manifest["files"].items()
                  if split in {"train", late_split}}
    spec = ModelSpec(**config["model"])
    limits = Limits(**config.get("limits", {}))
    if a.resume:
        model, tokenizer, encoder, meta = load_model(a.resume, device)
        if not spec.revision:
            spec.revision = model.spec.revision
        if asdict(model.spec) != asdict(spec) or asdict(encoder.limits) != asdict(limits):
            raise ValueError("resume model/limits differ from config")
    elif a.init_checkpoint:
        from .artifacts import resolve_checkpoint
        checkpoint = resolve_checkpoint(a.init_checkpoint)
        model, tokenizer, _, meta = load_model(checkpoint, device, base=spec.base,
                                              base_revision=spec.revision, weights_dtype=spec.weights_dtype)
        if not spec.revision:
            spec.revision = model.spec.revision
            config["model"] = asdict(spec)
        if not meta.get("adapter") or asdict(model.spec) != asdict(spec):
            raise ValueError("initialization requires a compatible LoRA model configuration")
        encoder = Encoder(tokenizer, limits, choice_none_policy=spec.choice_none_policy)
        model.temperature = 1.0  # Calibration from an earlier domain is not reused.
        if rank == 0:
            write_json(out / "initialization.json", {"checkpoint": str(a.init_checkpoint),
                       "optimizer": "fresh", "global_step": 0, "temperature": 1.0})
    else:
        from transformers import AutoTokenizer
        if not Path(spec.base).exists():
            from huggingface_hub import HfApi
            resolved = HfApi().model_info(spec.base, revision=spec.revision).sha
            spec.revision = resolved
            config["model"] = asdict(spec)
        tokenizer = AutoTokenizer.from_pretrained(spec.base, revision=spec.revision, trust_remote_code=False)
        encoder = Encoder(tokenizer, limits, choice_none_policy=spec.choice_none_policy)
        if full_ft and spec.lora_rank:
            raise ValueError("full fine-tuning requires model.lora_rank = 0")
        # Full fine-tuning loads on the host; FSDP moves only shards and gathered layers to the GPU.
        model = QevModel(load_backbone(spec, torch.device("cpu") if full_ft else device),
                                    spec, tokenizer.pad_token_id)
    if device.type == "cpu" and spec.weights_dtype == "bf16":
        raise ValueError("use an fp32 config for CPU training")
    model.prefix_execution = settings.get("prefix_execution", "leaf-rows")
    if model.prefix_execution not in {"leaf-rows", "shared-prefix", "tree", "tree-batched"}:
        raise ValueError("prefix_execution must be leaf-rows, shared-prefix, tree or tree-batched")
    if model.prefix_execution in {"tree", "tree-batched"}:
        require_deterministic_repartition(model)
        if settings.get("gradient_checkpointing_layers") is not None:
            raise ValueError("tree execution uses its own layer checkpointing")
        configure_checkpointing(model.backbone, False)
        model.tree_checkpointing = bool(settings.get("gradient_checkpointing", True))
    elif model.prefix_execution == "shared-prefix":
        require_deterministic_repartition(model)
        if settings.get("gradient_checkpointing_layers") is not None:
            raise ValueError("shared prefixes use segment checkpointing, not selected HF layers")
        configure_checkpointing(model.backbone, False)
        model.shared_checkpointing = bool(settings.get("gradient_checkpointing", True))
    else:
        configure_checkpointing(model.backbone, settings.get("gradient_checkpointing", True),
                                settings.get("gradient_checkpointing_layers"))
    model.backbone.config.use_cache = False
    if full_ft:
        from .fsdp import shard_full_model
        shard_full_model(model, device, world, cpu_offload=bool(settings.get("fsdp_cpu_offload", True)))

    encoded, rejected = [], []

    def admit(rows):
        for r in rows:
            try:
                encoded.append(encoder(r))
            except ContextOverflow as exc:
                rejected.append({"id": r.id, "reason": str(exc)})
        return len(encoded)
    n_main = admit(records)  # encoded[n_main:] are the late records
    n_late = admit(late_records) - n_main
    if not n_main or (late_records and not n_late):
        raise ValueError("no records fit the declared token limits")
    if any(q.question.target is None for r in encoded for q in r.questions):
        raise ValueError("training data contain unlabelled questions")
    admission_hash = __import__("hashlib").sha256("\n".join(r.record.id for r in encoded).encode()).hexdigest()
    if rank == 0:
        write_json(out / "admission.json", {"requested": len(records), "admitted": len(encoded),
                   "rejected": rejected,
                   "by_source": dict(Counter(r.record.source for r in encoded)),
                   **({"late_split": late_split, "late_admitted": n_late,
                       "late_by_source": dict(Counter(r.record.source for r in encoded[n_main:]))}
                      if late_split else {})})
        write_json(out / "config.json", config)

    batch_size, accum = int(settings.get("batch_size", 1)), int(settings.get("accum", 8))
    epochs = int(settings.get("epochs", 2))
    if min(batch_size, accum, epochs) < 1:
        raise ValueError("batch_size, accum, epochs must be positive")
    assignment = settings.get("batch_assignment", "strided-v1")
    if assignment not in ASSIGNMENTS:
        raise ValueError(f"unknown batch assignment: {assignment}")
    if assignment == "leaf-balanced-v1":
        require_deterministic_repartition(model)
    record_costs = [r.forward_tokens for r in encoded]

    def epoch_steps(n):
        return math.ceil(math.ceil(math.ceil(n / world) / batch_size) / accum)
    steps_per_epoch = epoch_steps(n_main)
    late = None
    if late_split:
        # Start at an optimizer boundary so every earlier step matches the run without the late split.
        start_step = math.floor((1 - late_fraction) * epochs * steps_per_epoch)
        late_epoch = epochs - 1
        if start_step < late_epoch * steps_per_epoch:
            raise ValueError("late_fraction must lie within the final epoch")
        late = {"epoch": late_epoch, "start_step": start_step,
                "position": (start_step - late_epoch * steps_per_epoch) * world * batch_size * accum,
                "ids": list(range(n_main, len(encoded)))}
    total_steps = epochs * steps_per_epoch + (
        epoch_steps(n_main + late_repeats * (len(encoded) - n_main)) - steps_per_epoch if late else 0)
    if late and rank == 0:
        write_json(out / "late-mix.json", {"split": late_split, "fraction": late_fraction, "records": len(late["ids"]),
                                           **({"repeats": late_repeats} if late_repeats != 1 else {}),
                                           "start_step": late["start_step"], "epoch": late["epoch"],
                                           "global_position": late["position"], "total_steps": total_steps})
    head_warmup = int(settings.get("head_warmup_steps", 0))
    head_params = list(model.head.parameters())
    lr, head_lr = float(settings.get("lr", 5e-5)), float(settings.get("head_lr", 1e-4))
    # Optional layer-wise decay: `lr` is the rate of the layer nearest the output.
    backbone_scales = backbone_lr_scales(model.backbone, float(settings.get("layer_lr_decay", 1.0)))
    if not backbone_scales:
        backbone_scales = [([], 1.0)]
    groups = [{"params": ps, "lr": lr * scale} for ps, scale in backbone_scales]
    n_backbone = len(groups)
    groups.append({"params": head_params, "lr": head_lr})
    # A zero-initialised tanh gate moves about one lr per Adam step, so it gets
    # its own larger rate; it follows the LoRA schedule (frozen in head warmup).
    gate_lr = float(settings.get("joint_gate_lr", 1e-3))
    if model.joint_layer is not None:
        groups.append({"params": [model.joint_gate], "lr": gate_lr, "weight_decay": 0.0})
    elif "joint_gate_lr" in settings:
        raise ValueError("joint_gate_lr requires model.candidate_interaction")
    ord_w = float(settings.get("ord_w", 0.0))
    score_hl_sigma = float(settings.get("score_hl_sigma", 0.0))
    none_absent_prob = float(settings.get("none_absent_prob", 0.0))
    if not 0 <= none_absent_prob < 1:
        raise ValueError("none_absent_prob must be in [0, 1)")
    if none_absent_prob and spec.choice_none_policy not in {"always", "always-varied"}:
        raise ValueError("none_absent_prob requires an always-None Choice policy")
    none_insert_prob = float(settings.get("none_insert_prob", 0.0))
    if none_insert_prob and spec.choice_none_policy != "as-provided":
        raise ValueError("none_insert_prob requires the as-provided Choice policy")
    inserter = NoneInserter(encoder, none_insert_prob, float(settings.get("none_insert_absent_frac", 0.5)))
    # Source prefixes kept exactly as provided (e.g. paired soft/hard-target controls whose inputs must match).
    insert_exempt = settings.get("none_insert_exempt_sources", [])
    if not isinstance(insert_exempt, list) or not all(isinstance(s, str) and s for s in insert_exempt):
        raise ValueError("none_insert_exempt_sources must be a list of non-empty source prefixes")
    insert_exempt = tuple(insert_exempt)
    teacher_cache = None
    if kd := settings.get("distillation"):
        if ord_w or score_hl_sigma:
            raise ValueError("distillation requires unmodified soft-target cross entropy")
        from .distillation import TeacherCache, training_views, view_protocol
        teacher_cache = TeacherCache(kd["cache"], protocol=view_protocol(config), weight=kd.get("weight", 1.0))
        teacher_cache.verify_coverage(training_views(encoded, n_main, encoder, config))
        data_files["teacher_manifest"] = file_hash(teacher_cache.path / "manifest.json")
        for entry in teacher_cache.manifest["shards"]:
            data_files["teacher/" + entry["file"]] = file_hash(teacher_cache.path / entry["file"])
    if ord_w < 0:
        raise ValueError("ord_w must be non-negative")
    if score_hl_sigma < 0:
        raise ValueError("score_hl_sigma must be non-negative")
    optimizer = torch.optim.AdamW(groups, weight_decay=float(settings.get("weight_decay", 0.01)))
    backbone_warmup = int(settings.get("backbone_warmup_steps", 0))
    eval_every = int(settings.get("eval_every", 0))
    eval_sets = settings.get("eval_sets", [])
    if eval_every and not (full_ft and eval_sets):
        raise ValueError("eval_every requires full_finetune and a non-empty eval_sets list")
    start_epoch, next_batch, global_step = 0, 0, 0
    if a.resume:
        state = torch.load(Path(a.resume) / "training.pt", map_location="cpu", weights_only=False)
        if (state["data_hash"] != data_hash or state["admission_hash"] != admission_hash
                or state.get("data_files", data_files) != data_files):
            raise ValueError("resume data or admitted sample set changed")
        if late and a.allow_repartition:
            raise ValueError("repartition is not supported with a late split")
        next_batch, migration = resume_cursor(state, settings, world, len(encoded), a.allow_repartition)
        if migration:
            require_deterministic_repartition(model)
            if rank == 0:
                write_json(out / "resume-migration.json", {"checkpoint": str(a.resume), **migration})
        optimizer.load_state_dict(state["optimizer"])
        for opt_state in optimizer.state.values():
            for key, value in opt_state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    opt_state[key] = value.to(device)
        start_epoch, global_step = state["epoch"], state["global_step"]
        if migration:
            torch.manual_seed(seed + rank)
            random.seed(seed + rank)
        else:
            restore_rng(state["rng_by_rank"][rank])
    else:
        # Initialization is shared across ranks; dropout streams are rank-specific.
        torch.manual_seed(seed + rank)
        random.seed(seed + rank)
    ddp_warmup = global_step < head_warmup

    def wrap_model():
        if full_ft:
            return model
        return DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None,
                                       find_unused_parameters=ddp_warmup,
                                       gradient_as_bucket_view=True) if world > 1 else model

    wrapped = wrap_model()
    model.train()
    precision = settings.get("autocast", spec.weights_dtype)
    if precision not in {"fp32", "bf16"}:
        raise ValueError("autocast must be fp32 or bf16")
    if spec.weights_dtype == "fp32" and precision != "fp32":
        raise ValueError("FP32 weights require FP32 execution; mixed precision is disabled")
    use_amp = device.type == "cuda" and precision == "bf16"
    t0 = time.monotonic()
    wall_limit = a.wall_hours or settings.get("wall_hours")
    save_every = int(settings.get("save_every", 100))
    last_saved = global_step if a.resume and (out / f"step-{global_step:06d}").exists() else -1

    def save(epoch, batch):
        nonlocal last_saved
        if last_saved == global_step:
            return
        states = [None] * world if rank == 0 else None
        if world > 1:
            dist.gather_object(rng_state(), states, dst=0)
        else:
            states = [rng_state()]
        if full_ft:
            from .fsdp import full_state_dict
            state = full_state_dict(model)  # collective on every rank
        if rank == 0:
            destination = out / f"step-{global_step:06d}"
            temporary = out / f".step-{global_step:06d}.incomplete"
            extra = {"training_execution": model.prefix_execution}
            if full_ft:
                # Weights only: the CPU-offloaded optimizer state (~12 bytes/parameter) is not saved.
                save_full_model(temporary, state, model, tokenizer, limits, {**extra, "full_finetune": True})
            else:
                save_model(temporary, model, tokenizer, limits, extra)
            torch.save({"optimizer": None if full_ft else optimizer.state_dict(), "epoch": epoch, "next_batch": batch,
                        "global_step": global_step, "world_size": world, "rng_by_rank": states,
                        "data_hash": data_hash, "admission_hash": admission_hash, "data_files": data_files,
                        "training_settings": settings}, temporary / "training.pt")
            temporary.rename(destination)

            write_json(out / "latest.json", {"checkpoint": destination.name, "step": global_step})
        if world > 1:
            dist.barrier()
        last_saved = global_step

    stop = False
    for epoch in range(start_epoch, epochs):
        ids = rank_indices(n_main, epoch, seed, rank, world, batch_size=batch_size,
                           accum=accum, assignment=assignment, costs=record_costs,
                           late=(late["position"], late["ids"]) if late and epoch == late["epoch"] else None,
                           late_repeats=late_repeats)
        batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
        cursor = next_batch if epoch == start_epoch else 0
        while cursor < len(batches):
            if a.max_steps is not None and global_step >= a.max_steps:
                stop = True
                break
            group = batches[cursor:cursor + accum]
            n_records = sum(map(len, group))
            model.head_only = global_step < head_warmup
            if world > 1 and not full_ft and ddp_warmup != model.head_only:
                # Rebuild at a completed optimizer boundary, on every rank.
                # The warmup really has unused LoRA parameters; the full phase does not.
                del wrapped
                ddp_warmup = model.head_only
                wrapped = wrap_model()
            factor = lr_factor(global_step, total_steps, int(settings.get("warmup_steps", 20)))
            # Optional separate backbone warmup that starts when head-only training ends.
            ramp = min(1.0, (global_step - head_warmup + 1) / backbone_warmup) if backbone_warmup else 1.0
            for gi, (_, scale) in enumerate(backbone_scales):
                optimizer.param_groups[gi]["lr"] = 0.0 if model.head_only else lr * scale * factor * ramp
            optimizer.param_groups[n_backbone]["lr"] = head_lr * factor
            if model.joint_layer is not None:
                optimizer.param_groups[n_backbone + 1]["lr"] = 0.0 if model.head_only else gate_lr * factor
            optimizer.zero_grad(set_to_none=True)
            observed_loss, tokens, relabelled, inserted, inserted_absent = 0.0, 0, 0, 0, 0
            for mi, group_ids in enumerate(group):
                chunk = []
                for i in group_ids:
                    view, count = none_absent_view(encoded[i], prob=none_absent_prob, seed=seed, epoch=epoch)
                    added = absent = 0
                    if not view.record.source.startswith(insert_exempt):
                        view, added, absent = inserter(view, seed=seed, epoch=epoch)
                    if teacher_cache is not None:
                        view = teacher_cache.apply(view)
                    chunk.append(view)
                    relabelled += count
                    inserted += added
                    inserted_absent += absent
                sync = wrapped.no_sync() if world > 1 and not full_ft and mi != len(group) - 1 else nullcontext()
                with sync:
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                        logits = wrapped(chunk)
                        loss = sum(record_loss(z, r, ord_w, score_hl_sigma) for z, r in zip(logits, chunk, strict=True)) / n_records
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite training loss")
                    loss.backward()
                observed_loss += float(loss.detach())
                tokens += sum(r.forward_tokens for r in chunk)
            norm = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0, error_if_nonfinite=True)
            gate_values = None
            if full_ft:
                from .fsdp import gathered
                norm = gathered(norm)  # collective on every rank
                if model.joint_layer is not None:
                    gate_values = gathered(model.joint_gate.detach())
            elif model.joint_layer is not None:
                gate_values = model.joint_gate.detach()
            optimizer.step()
            global_step += 1
            cursor += len(group)
            totals = torch.tensor([observed_loss, tokens, relabelled, inserted, inserted_absent],
                                  dtype=torch.float64, device=device)
            peaks = torch.tensor([
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
            ], dtype=torch.int64, device=device)
            if world > 1:
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
            if rank == 0:
                log = {"step": global_step, "epoch": epoch, "loss_rank0": observed_loss,
                       "prefix_execution": model.prefix_execution,
                       "readout_precision": model.readout_precision,
                       "forward_tokens_scope": "reference_leaf_work",
                       "loss_global_mean": float(totals[0]) / world,
                       "grad_norm": float(norm), "forward_tokens_rank0": tokens,
                       "lr_backbone_top_bottom": [optimizer.param_groups[0]["lr"],
                                                  optimizer.param_groups[n_backbone - 1]["lr"]],
                       "lr_head": optimizer.param_groups[n_backbone]["lr"],
                       "none_absent_views_global": int(totals[2]),
                       "none_inserted_global": int(totals[3]), "none_inserted_absent_global": int(totals[4]),
                       "forward_tokens_global": int(totals[1]),
                       "seconds": time.monotonic() - t0, "head_only": model.head_only,
                       "ddp_find_unused_parameters": bool(world > 1 and ddp_warmup),
                       "joint_gate_tanh": ([round(float(g), 6) for g in torch.tanh(gate_values.float())]
                                           if gate_values is not None else None),
                       "full_finetune": full_ft,
                       "peak_gpu_bytes": int(peaks[0]), "peak_gpu_reserved_bytes": int(peaks[1]),
                       "memory_scope": "maximum_across_ranks"}
                with (out / "train.jsonl").open("a") as stream:
                    stream.write(json.dumps(log) + "\n")
                print(json.dumps(log), flush=True)
            if eval_every and global_step % eval_every == 0:
                from .inline_eval import evaluate_sharded
                summary = evaluate_sharded(
                    model, encoder, eval_sets, out.parent / "evaluation-inline", step=global_step, rank=rank,
                    world=world, batch_size=int(settings.get("eval_batch_size", batch_size)),
                    autocast=lambda: torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp))
                if rank == 0:
                    with (out / "eval.jsonl").open("a") as stream:
                        stream.write(json.dumps({"step": global_step, "accuracy": summary}) + "\n")
                    print(json.dumps({"stage": "inline_evaluation", "step": global_step, "accuracy": summary}), flush=True)
            save_epoch, save_batch = (epoch + 1, 0) if cursor == len(batches) else (epoch, cursor)
            if save_every and global_step % save_every == 0:
                save(save_epoch, save_batch)
            timeout = bool(wall_limit and time.monotonic() - t0 >= wall_limit * 3600)
            signal = torch.tensor(int(timeout), device=device)
            if world > 1:
                dist.all_reduce(signal, op=dist.ReduceOp.MAX)
            if signal.item():
                stop = True
                break
        next_batch = cursor
        if stop:
            save(epoch + (cursor == len(batches)), 0 if cursor == len(batches) else cursor)
            break
        next_batch = 0
    if not stop:
        save(epochs, 0)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
