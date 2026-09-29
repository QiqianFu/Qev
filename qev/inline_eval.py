# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Periodic evaluation inside FSDP full-parameter training (no intermediate checkpoints needed).

Every rank evaluates a strided shard of each split through the model's root call, so FSDP gathers
root and layer shards. FSDP collectives require equal call counts, so shorter ranks repeat a padding
batch whose outputs are discarded. Rows and the report use evaluate.py's format; execution is
tree-batched under the training autocast, which can differ numerically from standalone --reference.
"""
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import torch
import torch.distributed as dist

from .choice_policy import none_candidate
from .data import load_records, write_json
from .encoding import ContextOverflow
from .evaluate import checked_probabilities, summarize


def _rows(encoded, logits, temperature, seconds, batch_records):
    rows = []
    record = encoded.record
    for q, z in zip(record.questions, logits, strict=True):
        ids = [c.id for c in q.candidates]
        p = checked_probabilities(torch.softmax(z.float() / temperature, -1), len(ids))
        pred = ids[int(p.argmax())]
        none = none_candidate(q)
        target = torch.tensor(q.target) if q.target is not None else None
        rows.append({"record_id": record.id, "group_id": record.group_id, "question_id": q.id, "source": record.source,
                     "type": q.type, "prediction": pred, "probabilities": dict(zip(ids, p.tolist())),
                     "none_candidate_id": none.id if none else None, "predicted_none": bool(none and pred == none.id),
                     "label": q.label, "correct": pred == q.label if q.label is not None else None,
                     "confidence": float(p.max()), "record_seconds": seconds / batch_records,
                     "batch_seconds": seconds, "batch_records": batch_records, "timing_scope": "amortized_batch",
                     "brier": float((p - target).square().sum()) if target is not None else None,
                     "nll": float(-(target * p.clamp_min(1e-12).log()).sum()) if target is not None else None})
    return rows


def evaluate_sharded(model, encoder, eval_sets, out, *, step, rank, world, batch_size, autocast):
    """eval_sets: [{"data": dir, "split": name}]. Collective: every rank must call. Rank 0 writes reports."""
    was_training = model.training
    model.eval()
    summaries = {}
    try:
        for item in eval_sets:
            records, manifest = load_records(item["data"], item["split"])
            if manifest["files"][item["split"]]["role"] == "test":
                raise ValueError("inline evaluation never opens test splits")
            encoded, rejected, rejected_by_source = [], [], defaultdict(int)
            for record in records:
                try:
                    encoded.append(encoder(record))
                except ContextOverflow as exc:
                    rejected.append({"id": record.id, "source": record.source, "reason": str(exc)})
                    rejected_by_source[record.source] += len(record.questions)
            mine = encoded[rank::world]
            calls = math.ceil(math.ceil(len(encoded) / world) / batch_size)
            rows, started = [], time.perf_counter()
            with torch.no_grad(), autocast():
                for c in range(calls):
                    batch = mine[c * batch_size:(c + 1) * batch_size]
                    padding = not batch
                    t0 = time.perf_counter()
                    logits = model(batch if batch else encoded[:1])  # equal collective count on every rank
                    seconds = time.perf_counter() - t0
                    if not padding:
                        for e, z in zip(batch, logits, strict=True):
                            rows.extend(_rows(e, z, model.temperature, seconds, len(batch)))
            gathered = [None] * world if rank == 0 else None
            dist.gather_object(rows, gathered, dst=0)
            if rank == 0:
                rows = [r for part in gathered for r in part]
                target = Path(out) / f"step-{step:06d}" / item["split"]
                target.mkdir(parents=True, exist_ok=False)
                with (target / "predictions.jsonl").open("w") as stream:
                    for row in rows:
                        stream.write(json.dumps(row) + "\n")
                report = summarize(rows, sum(rejected_by_source.values()), rejected_by_source)
                report.update({"split": item["split"], "step": step,
                               "execution": "inline_fsdp_tree_batched", "autocast": "training",
                               "readout_precision": model.readout_precision, "world_size": world,
                               "seconds": time.perf_counter() - started, "rejected": rejected})
                write_json(target / "report.json", report)
                summaries[item["split"]] = report["accuracy_answered"]
    finally:
        model.train(was_training)
    return summaries
