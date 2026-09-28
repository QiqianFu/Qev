# Adapted for Qev in 2026; see NOTICE and provenance.json.
"""Versioned checkpoints; never treat an original Kev pointer head as a set head."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch
from safetensors.torch import load_file, save_file

from .data import write_json
from .encoding import Encoder, Limits, LAYOUT
from .model import QevModel, ModelSpec, load_backbone

FORMAT = "qev.checkpoint.v1"


def wait_for_archive(checkpoint, ack_dir, timeout=600):
    """Staged checkpoints are committed only after the host verifies shared storage."""
    ack = Path(ack_dir) / (checkpoint + ".json")
    started = time.monotonic()
    while True:
        if ack.is_file():
            receipt = json.loads(ack.read_text())
            if receipt.get("checkpoint") != checkpoint or receipt.get("verified") is not True:
                raise ValueError(f"invalid archive acknowledgement: {ack}")
            return receipt
        if time.monotonic() - started >= timeout:
            raise TimeoutError(f"checkpoint {checkpoint} was written locally but shared archive was not acknowledged")
        time.sleep(0.25)


def rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("resume requires the same visible CUDA device count")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_model(path, model, tokenizer, limits, extra=None):
    if model.lora_merged:
        raise ValueError("cannot save an adapter checkpoint after merging LoRA for inference; retain the original checkpoint")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    adapted = hasattr(model.backbone, "peft_config")
    if adapted:
        model.backbone.save_pretrained(path / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(path / "tokenizer")
    save_file({k: v.detach().cpu().contiguous() for k, v in model.head.state_dict().items()}, str(path / "head.safetensors"))
    if model.joint_layer is not None:
        save_file({"gate": model.joint_gate.detach().cpu().contiguous()}, str(path / "joint.safetensors"))
    write_json(path / "model.json", {"format": FORMAT, "layout": LAYOUT, "spec": asdict(model.spec),
                                    "limits": asdict(limits), "adapter": adapted,
                                    "readout_precision": model.readout_precision,
                                    "temperature": model.temperature, "extra": extra or {}})


def save_full_model(path, state, model, tokenizer, limits, extra=None):
    """Full-parameter checkpoint from a gathered state dict: BF16 backbone weights loadable by from_pretrained."""
    from .model import _base_model
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    dtype = torch.float32 if model.spec.weights_dtype == "fp32" else torch.bfloat16
    backbone = {k[len("backbone."):]: v.to(dtype).contiguous() for k, v in state.items() if k.startswith("backbone.")}
    head = {k[len("head."):]: v.float().contiguous() for k, v in state.items() if k.startswith("head.")}
    if not backbone or not head:
        raise ValueError("full state dict is missing backbone or head tensors")
    (path / "backbone").mkdir()
    config = _base_model(model.backbone).config
    config.save_pretrained(path / "backbone")
    save_file(backbone, str(path / "backbone" / "model.safetensors"), metadata={"format": "pt"})
    tokenizer.save_pretrained(path / "tokenizer")
    save_file(head, str(path / "head.safetensors"))
    if model.joint_layer is not None:
        save_file({"gate": state["joint_gate"].float().contiguous()}, str(path / "joint.safetensors"))
    write_json(path / "model.json", {"format": FORMAT, "layout": LAYOUT, "spec": asdict(model.spec),
                                    "limits": asdict(limits), "adapter": False, "backbone": "full",
                                    "readout_precision": model.readout_precision,
                                    "temperature": model.temperature, "extra": extra or {}})


def load_model(path, device="cpu", *, merge_lora=False, weights_dtype=None,
               revision=None, base=None, base_revision=None):
    from transformers import AutoTokenizer
    from .artifacts import resolve_checkpoint
    path = resolve_checkpoint(path, revision)
    meta = json.loads((path / "model.json").read_text())
    if meta.get("format") not in {FORMAT, "branchkev.checkpoint.v1"} or meta.get("layout") != LAYOUT:
        raise ValueError("not a compatible Qev checkpoint")
    spec_values = dict(meta["spec"])
    if meta.get("backbone") == "full":
        if base is not None or base_revision is not None:
            raise ValueError("a full checkpoint already contains its backbone")
        # Fully fine-tuned weights live inside the checkpoint; the original base is only provenance.
        spec_values["base"], spec_values["revision"] = str(path / "backbone"), None
    elif base is not None:
        spec_values["base"] = str(base)
        spec_values["revision"] = None if Path(base).exists() else (base_revision or spec_values.get("revision"))
    elif base_revision is not None:
        raise ValueError("base_revision requires a base override")
    if weights_dtype is not None:
        spec_values["weights_dtype"] = weights_dtype
    spec = ModelSpec(**spec_values)
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
    backbone = load_backbone(spec, device, str(path / "adapter") if meta["adapter"] else None)
    model = QevModel(backbone, spec, tokenizer.pad_token_id)
    model.head.load_state_dict(load_file(str(path / "head.safetensors")), strict=True)
    if model.joint_layer is not None:
        with torch.no_grad():
            model.joint_gate.copy_(load_file(str(path / "joint.safetensors"))["gate"])
    model.temperature = float(meta["temperature"])
    if not 0 < model.temperature < float("inf"):
        raise ValueError("invalid checkpoint temperature")
    encoder = Encoder(tokenizer, Limits(**meta["limits"]), choice_none_policy=spec.choice_none_policy)
    if merge_lora:
        model.prepare_inference(merge_lora=True)
    return model, tokenizer, encoder, meta
