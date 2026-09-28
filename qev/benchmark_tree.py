# Adapted for Qev in 2026; see NOTICE and provenance.json.
"""Paired same-checkpoint cached/tree latency and numerical quality audit."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from .benchmark_speed import prepare
from .checkpoint import load_model
from .data import file_hash, write_json
from .evaluate import checked_probabilities


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--records', type=int, default=32)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--quality-records', type=int, default=1000)
    p.add_argument('--merge-lora', action='store_true')
    a = p.parse_args()
    if min(a.records, a.repeats, a.quality_records) < 1:
        p.error('record and repeat counts must be positive')
    a.out.mkdir(parents=True, exist_ok=False)
    _, records, groups, _, descriptions = prepare(a.checkpoint, a.data, a.records)
    model, _, _, _ = load_model(a.checkpoint, 'cuda')
    model.prepare_inference(merge_lora=a.merge_lora)
    report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'checkpoint': str(a.checkpoint), 'manifest_sha256': file_hash(a.data / 'manifest.json'),
              'merged_lora': model.lora_merged, 'weights_dtype': model.spec.weights_dtype,
              'sample': descriptions, 'repeats': a.repeats, 'timings': {}}
    for split, sample in groups.items():
        rows = []
        for idx, e in enumerate(sample):
            row = {'record_id': e.record.id}
            outputs = {}
            # Alternate arm order, with per-input warmup on both arms.
            for tree in (False, True) if idx % 2 == 0 else (True, False):
                name = 'tree' if tree else 'cached'
                for _ in range(2):
                    model.predict(e, tree=tree)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                times = []
                for _ in range(a.repeats):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    ps = [v.float().cpu() for v in model.predict(e, tree=tree)]
                    torch.cuda.synchronize()
                    times.append(1000 * (time.perf_counter() - started))
                row[name] = {'median_ms': statistics.median(times), 'raw_ms': times,
                             'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
                outputs[name] = ps
            row['probability_max_abs'] = max(float((x - y).abs().max()) for x, y in
                                             zip(outputs['cached'], outputs['tree'], strict=True))
            rows.append(row)
        write_json(a.out / f'{split}-timings.json', rows)
        report['timings'][split] = {name: statistics.median(r[name]['median_ms'] for r in rows)
                                   for name in ('cached', 'tree')}
        report['timings'][split]['max_probability_difference'] = max(r['probability_max_abs'] for r in rows)
        print(json.dumps({'stage': 'timing', 'split': split, **report['timings'][split]}), flush=True)
        write_json(a.out / 'report.json', report)

    quality = {'n': 0, 'cached_correct': 0, 'tree_correct': 0, 'flips': 0,
               'right_to_wrong': 0, 'wrong_to_right': 0, 'probability_max_abs': 0.0}
    with (a.out / 'paired-predictions.jsonl').open('w') as stream:
        for i, e in enumerate(records[:a.quality_records]):
            cached = model.predict(e)
            tree = model.predict(e, tree=True)
            for q, x, y in zip(e.record.questions, cached, tree, strict=True):
                ids = [c.id for c in q.candidates]
                x, y = (checked_probabilities(v, len(ids)) for v in (x, y))
                a_label, b_label = ids[int(x.argmax())], ids[int(y.argmax())]
                ac, bc = a_label == q.label, b_label == q.label
                diff = float((x - y).abs().max())
                quality['n'] += 1
                quality['cached_correct'] += ac
                quality['tree_correct'] += bc
                quality['flips'] += a_label != b_label
                quality['right_to_wrong'] += ac and not bc
                quality['wrong_to_right'] += bc and not ac
                quality['probability_max_abs'] = max(quality['probability_max_abs'], diff)
                stream.write(json.dumps({'record_id': e.record.id, 'question_id': q.id,
                    'label': q.label, 'cached': dict(zip(ids, x.tolist())), 'tree': dict(zip(ids, y.tolist())),
                    'cached_prediction': a_label, 'tree_prediction': b_label}) + '\n')
            if (i + 1) % 100 == 0:
                stream.flush()
                print(json.dumps({'stage': 'quality', **quality}), flush=True)
    report.update(quality=quality, completed=True)
    write_json(a.out / 'report.json', report)


if __name__ == '__main__':
    main()
