# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Compare execution paths on the same checkpoint and fixed development inputs."""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import statistics
import sys
import time
from types import MethodType

import torch

from .checkpoint import load_model
from .data import load_records, write_json
from .encoding import ContextOverflow
from .execution import configure_checkpointing, configure_fp32
from .model import record_loss


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="decision_dev")
    ap.add_argument("--records", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 16])
    ap.add_argument("--baseline-model", help="model.py from the pre-optimization Git snapshot")
    ap.add_argument("--training-gate", action="store_true", help="compare leaf/shared gradients on two short records")
    ap.add_argument("--training-only", action="store_true", help="only compare warmed FP32 forward/backward execution")
    ap.add_argument("--training-repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if min(a.records, a.repeats, a.training_repeats, *a.batch_sizes) < 1:
        ap.error("records, repeats and batch sizes must be positive")
    if not torch.cuda.is_available():
        ap.error("benchmark requires CUDA")
    records, manifest = load_records(a.data, a.split)
    if manifest["files"][a.split]["role"] != "development":
        ap.error("benchmark accepts development data only")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    precision = configure_fp32()
    model, _, encoder, _ = load_model(a.checkpoint, "cuda", weights_dtype="fp32")
    assert all(p.dtype == torch.float32 for p in model.parameters())
    model.eval()
    rng = random.Random(17)
    rng.shuffle(records)
    encoded = []
    for record in records:
        try:
            encoded.append(encoder(record))
        except ContextOverflow:
            continue
        if len(encoded) == a.records:
            break
    if not encoded:
        ap.error("no admitted benchmark records")
    report = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "precision": precision,
              "checkpoint": a.checkpoint, "split": a.split, "seed": 17,
              "record_ids": [e.record.id for e in encoded], "repeats": a.repeats, "paths": {}}
    baseline = None
    current_rows, current_fork = model._run_rows, model.fork_cache
    if a.baseline_model:
        name = "qev._benchmark_baseline"
        spec = importlib.util.spec_from_file_location(name, a.baseline_model)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[name] = baseline
        spec.loader.exec_module(baseline)

    reference = None

    def measure(name, predict, size=1):
        nonlocal reference
        groups = [encoded[i:i + size] for i in range(0, len(encoded), size)]
        with torch.no_grad():
            for group in groups:
                predict(group)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            totals, latencies, output = [], [], []
            calls = []
            hook = model.backbone.register_forward_hook(lambda *args: calls.append(1))
            try:
                for repetition in range(a.repeats):
                    elapsed = 0
                    output = []
                    for group in groups:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        ps = predict(group)
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - start
                        elapsed += seconds
                        latencies.append(seconds)
                        output.extend(p.detach().float().cpu() for questions in ps for p in questions)
                    totals.append(elapsed)
            finally:
                hook.remove()
        assert all(torch.isfinite(p).all() and (p >= 0).all() and abs(float(p.sum()) - 1) < 1e-5 for p in output)
        if reference is None:
            reference = output
        pairs = list(zip(reference, output, strict=True))
        sorted_times = sorted(latencies)
        result = {"median_total_seconds": statistics.median(totals), "total_seconds": totals,
                  "questions_per_second": len(output) / statistics.median(totals),
                  "latency_unit": "record" if size == 1 else "batch",
                  "latency_p50_seconds": statistics.median(latencies),
                  "latency_p95_seconds": sorted_times[min(len(sorted_times) - 1, int(.95 * len(sorted_times)))],
                  "batch_size": size, "backbone_calls_per_repeat": len(calls) / a.repeats,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "probability_max_abs_vs_reference": max(float((x - y).abs().max()) for x, y in pairs),
                  "argmax_flips_vs_reference": sum(int(x.argmax()) != int(y.argmax()) for x, y in pairs)}
        report["paths"][name] = result
        write_json(out / "report.json", report)
        print(json.dumps({"path": name, **result}), flush=True)

    serial_reference = lambda group: [model.predict(e, cached=False) for e in group]
    if not a.training_only:
        if baseline:
            model._run_rows = MethodType(baseline.QevModel._run_rows, model)
            measure("baseline_reference", serial_reference)
            model._run_rows = current_rows
        measure("reference", serial_reference)
        for size in a.batch_sizes:
            measure(f"pooled_{size}", model.predict_batch, size)
        if baseline:
            model.fork_cache = MethodType(baseline.QevModel.fork_cache, model)
            measure("legacy_cache_fork", lambda group: [model.predict(e) for e in group])
            model.fork_cache = current_fork
        measure("cached", lambda group: [model.predict(e) for e in group])

    if a.training_gate or a.training_only:
        # Bound the memory footprint independently of the inference benchmark sample.
        short = []
        for record in records:
            try:
                e = encoder(record)
            except ContextOverflow:
                continue
            if len(e.questions) == 1 and len(e.questions[0].candidates) <= 4 and e.forward_tokens <= 512:
                short.append(e)
            if len(short) == 2:
                break
        if len(short) < 2:
            raise ValueError("not enough short development records for the training gate")
        model.train()
        expected = None
        gates = {}
        for path in ("leaf-rows", "shared-prefix"):
            configure_checkpointing(model.backbone, path == "leaf-rows")
            model.prefix_execution = path
            model.shared_checkpointing = True
            torch.cuda.reset_peak_memory_stats()
            times = []
            for repetition in range(a.training_repeats + 1):
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.autocast("cuda", enabled=False):
                    logits = model(short)
                    loss = sum(record_loss(z, r) for z, r in zip(logits, short, strict=True)) / len(short)
                loss.backward()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
            result = {"seconds_cold": times[0], "warm_seconds": times[1:],
                      "median_warm_seconds": statistics.median(times[1:]), "loss": float(loss.detach()),
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
            grads = {n: p.grad.detach().float().cpu() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
            if expected is None:
                expected = grads
            else:
                assert expected.keys() == grads.keys()
                numerator = sum(float((grads[n] - g).square().sum()) for n, g in expected.items())
                denominator = sum(float(g.square().sum()) for g in expected.values())
                result["gradient_relative_l2"] = (numerator / max(denominator, 1e-30)) ** .5
                result["passed"] = result["gradient_relative_l2"] < .001 and torch.isfinite(loss).item()
            gates[path] = result
            print(json.dumps({"training_path": path, **result}), flush=True)
        report["training_gate"] = {"record_ids": [r.record.id for r in short], "paths": gates,
                                   "note": "One warmup followed by repeated forward/backward on two short records; no optimizer updates or DDP; not a full-run speed estimate."}
        write_json(out / "report.json", report)
        if not gates["shared-prefix"]["passed"]:
            raise RuntimeError("shared-prefix FP32 gradient gate failed")
        del expected, grads
        model.zero_grad(set_to_none=True)
        configure_checkpointing(model.backbone, False)
        model.eval()

    if not a.training_only:
        model.prepare_inference(merge_lora=True)
        measure("merged_pooled", model.predict_batch, max(a.batch_sizes))
    write_json(out / "report.json", report)


if __name__ == "__main__":
    main()
