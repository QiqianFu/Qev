# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Save per-question probabilities and matched-denominator accuracy metrics."""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import torch

from .checkpoint import load_model
from .data import load_records, write_json
from .encoding import ContextOverflow, Encoder
from .choice_policy import none_candidate


def checked_probabilities(p, n):
    p = p.detach().float().cpu()
    if p.shape != (n,) or not torch.isfinite(p).all() or (p < 0).any():
        raise ValueError("invalid prediction probability vector")
    if not math.isclose(float(p.sum()), 1.0, abs_tol=1e-5):
        raise ValueError("prediction probabilities do not sum to one")
    return p


def summarize(rows, rejected_questions=0, rejected_by_source=None):
    labelled = [r for r in rows if r["correct"] is not None]
    groups = defaultdict(list)
    for row in labelled:
        groups[row["source"]].append(row["correct"])
    by_source = {source: {"n": len(values), "accuracy": sum(values) / len(values)} for source, values in groups.items()}
    rejected_by_source = rejected_by_source or {}
    all_sources = set(groups) | set(rejected_by_source)
    source_all = {s: sum(groups[s]) / (len(groups[s]) + rejected_by_source.get(s, 0)) for s in all_sources}
    brier = [r["brier"] for r in rows if r["brier"] is not None]
    nll = [r["nll"] for r in rows if r["nll"] is not None]
    n = len(labelled)
    ece = 0.0
    for i in range(10):
        part = [r for r in labelled if min(int(r["confidence"] * 10), 9) == i]
        if part:
            ece += len(part) / max(1, n) * abs(sum(r["confidence"] for r in part) / len(part) - sum(r["correct"] for r in part) / len(part))
    return {"answered_questions": len(rows), "labelled_questions": n, "rejected_questions": rejected_questions,
            "accuracy_answered": sum(r["correct"] for r in labelled) / n if n else None,
            "accuracy_counting_rejections_wrong": sum(r["correct"] for r in labelled) / (n + rejected_questions) if n + rejected_questions else None,
            "macro_accuracy_answered": sum(v["accuracy"] for v in by_source.values()) / len(by_source) if by_source else None,
            "macro_accuracy_counting_rejections_wrong": sum(source_all.values()) / len(source_all) if source_all else None,
            "brier": sum(brier) / len(brier) if brier else None, "nll": sum(nll) / len(nll) if nll else None,
            "ece_10_bins": ece if n else None, "by_source": by_source}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--revision", help="pinned Hub checkpoint revision")
    ap.add_argument("--base", help="local base-model cache override")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="transfer_dev")
    ap.add_argument("--allow-test", action="store_true", help="explicitly evaluate a final test split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    execution = ap.add_mutually_exclusive_group()
    execution.add_argument("--reference", action="store_true", help="use full leaf rows instead of prefix cache")
    execution.add_argument("--tree", action="store_true", help="one decoder traversal with tree masks and branched DeltaNet states")
    ap.add_argument("--batch-size", type=int, default=1, help="pool this many records; >1 requires --reference or --tree")
    ap.add_argument("--merge-lora", action="store_true", help="merge adapters in this inference process only")
    ap.add_argument("--weights-dtype", choices=["fp32", "checkpoint"], default="fp32",
                    help="load actual backbone/adapter weights in FP32; checkpoint retains historical precision")
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--max-state", type=int, help="inference-only override of the checkpoint's state token limit")
    ap.add_argument("--max-path", type=int, help="inference-only override of the checkpoint's path token limit")
    a = ap.parse_args()
    if a.batch_size < 1 or (a.batch_size > 1 and not (a.reference or a.tree)):
        ap.error("--batch-size must be positive; values >1 require --reference or --tree")
    records, manifest = load_records(a.data, a.split)
    if manifest["files"][a.split]["role"] == "test" and not a.allow_test:
        ap.error("test split requires --allow-test")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    model, tokenizer, encoder, _ = load_model(a.checkpoint, a.device,
                                            revision=a.revision, base=a.base,
                                            weights_dtype="fp32" if a.weights_dtype == "fp32" else None)
    trained_limits = encoder.limits
    if a.max_state or a.max_path:
        # Longer inputs than the checkpoint was trained on; reported separately from the trained-limit numbers.
        limits = replace(trained_limits, max_state=a.max_state or trained_limits.max_state,
                         max_path=a.max_path or trained_limits.max_path)
        encoder = Encoder(tokenizer, limits, choice_none_policy=encoder.choice_none_policy)
    model.prepare_inference(merge_lora=a.merge_lora)
    predictions, rejected = [], []
    rejected_questions = 0
    rejected_by_source = defaultdict(int)
    record_times = []
    started = time.perf_counter()
    with torch.no_grad(), (out / "predictions.jsonl").open("w") as stream:
        for start in range(0, len(records), a.batch_size):
            batch = []
            for record in records[start:start + a.batch_size]:
                try:
                    batch.append(encoder(record))
                except ContextOverflow as exc:
                    rejected.append({"id": record.id, "source": record.source, "questions": len(record.questions), "reason": str(exc)})
                    rejected_questions += len(record.questions)
                    rejected_by_source[record.source] += len(record.questions)
            if not batch:
                continue
            if model.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = (model.predict_batch(batch, tree=a.tree) if a.batch_size > 1 else
                       [model.predict(batch[0], cached=not a.reference, tree=a.tree)])
            if model.device.type == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            record_times.append(seconds)
            for encoded, ps in zip(batch, outputs, strict=True):
                record = encoded.record
                for q, p in zip(record.questions, ps, strict=True):
                    ids = [c.id for c in q.candidates]
                    p = checked_probabilities(p, len(ids))
                    pred = ids[int(p.argmax())]
                    none = none_candidate(q)
                    target = torch.tensor(q.target) if q.target is not None else None
                    row = {"record_id": record.id, "group_id": record.group_id, "question_id": q.id, "source": record.source,
                           "type": q.type, "prediction": pred, "probabilities": dict(zip(ids, p.tolist())),
                           "none_candidate_id": none.id if none else None,
                           "predicted_none": bool(none and pred == none.id),
                           "label": q.label, "correct": pred == q.label if q.label is not None else None,
                           "confidence": float(p.max()), "record_seconds": seconds / len(batch),
                           "batch_seconds": seconds, "batch_records": len(batch),
                           "timing_scope": "amortized_batch" if a.batch_size > 1 else "record",
                           "brier": float((p - target).square().sum()) if target is not None else None,
                           "nll": float(-(target * p.clamp_min(1e-12).log()).sum()) if target is not None else None}
                    predictions.append(row)
                    stream.write(json.dumps(row) + "\n")
            processed = min(start + a.batch_size, len(records))
            if a.progress_every and (processed // a.progress_every > start // a.progress_every or processed == len(records)):
                stream.flush()
                print(json.dumps({"stage": "evaluation", "split": a.split, "cache": not (a.reference or a.tree),
                                  "records": processed, "total": len(records),
                                  "questions": len(predictions), "seconds": time.perf_counter() - started}), flush=True)
    report = summarize(predictions, rejected_questions, rejected_by_source)
    execution_name = "tree" if a.tree else "reference" if a.reference else "cached"
    if a.batch_size > 1:
        execution_name = "pooled_" + execution_name
    report.update({"split": a.split,
                   "checkpoint": str(a.checkpoint), "cache": not (a.reference or a.tree),
                   "batch_size": a.batch_size,
                   "lora_merged": model.lora_merged, "head_precision": model.head_precision,
                   "readout_precision": model.readout_precision,
                   "weights_dtype": model.spec.weights_dtype,
                   "choice_none_policy": model.spec.choice_none_policy,
                   "candidate_interaction": model.spec.candidate_interaction,
                   "execution": execution_name,
                   "inference_seconds": sum(record_times),
                   "peak_gpu_bytes": torch.cuda.max_memory_allocated(model.device) if model.device.type == "cuda" else 0,
                   "limits": asdict(encoder.limits), "trained_limits": asdict(trained_limits),
                   "rejected": rejected})
    write_json(out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"by_source", "rejected"}}, indent=2))


if __name__ == "__main__":
    main()
