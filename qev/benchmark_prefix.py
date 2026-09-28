# Adapted for Qev in 2026; see NOTICE and provenance.json.
"""Validate and time shared-prefix inference on a fixed checkpoint (no training)."""
import argparse
import json
from pathlib import Path
import random
import statistics
import time

import torch

from .checkpoint import load_model
from .data import file_hash, json_rows, load_records, write_json


def stats(values):
    values = sorted(values)
    return {"median_ms": 1000 * statistics.median(values), "mean_ms": 1000 * statistics.mean(values),
            "p90_ms": 1000 * values[int(.9 * len(values))]}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--split', default='mmlupro')
    p.add_argument('--extra-split', default='scienthoon')
    p.add_argument('--reference-predictions', required=True)
    p.add_argument('--records', type=int, default=32)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if min(a.records, a.repeats) < 1:
        p.error('records and repeats must be positive')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    records, manifest = load_records(a.data, a.split)
    extra, _ = load_records(a.data, a.extra_split)
    if any(manifest['files'][s]['role'] == 'test' for s in (a.split, a.extra_split)):
        p.error('benchmark must not use a final test split')
    model, _, encoder, _ = load_model(a.checkpoint, 'cuda')
    model.prepare_inference()
    encoded = [encoder(r) for r in records]
    other = [encoder(r) for r in extra]
    rng = random.Random(17)
    indices = set(rng.sample(range(len(encoded)), min(a.records, len(encoded))))
    indices.update(max(range(len(encoded)), key=key) for key in (
        lambda i: len(encoded[i].state), lambda i: encoded[i].forward_tokens))
    groups = {a.split: [encoded[i] for i in sorted(indices)],
              a.extra_split: sorted(other, key=lambda e: e.forward_tokens, reverse=True)[:8]}
    report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'checkpoint': a.checkpoint, 'weights_dtype': model.spec.weights_dtype,
              'lora_merged': model.lora_merged, 'candidate_interaction': model.spec.candidate_interaction,
              'manifest_sha256': file_hash(Path(a.data) / 'manifest.json'),
              'reference_predictions_sha256': file_hash(a.reference_predictions),
              'timing_scope': 'synchronized wall time, input tensors and output probabilities included, tokenization excluded',
              'sample_seed': 17, 'repeats': a.repeats, 'groups': {}}

    def predict(e, cached):
        return model.predict(e, cached=cached)

    for split, sample in groups.items():
        results, outputs = {}, {}
        for name, cached in (('reference', False), ('shared_cached', True)):
            for e in sample:
                predict(e, cached)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            totals, elapsed, ps = [], [], []
            for repetition in range(a.repeats):
                total = 0
                ps = []
                for e in sample:
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    prediction = predict(e, cached)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - t
                    elapsed.append(seconds)
                    total += seconds
                    ps.extend(p.cpu() for p in prediction)
                totals.append(total)
            outputs[name] = ps
            results[name] = {**stats(elapsed), 'total_seconds': totals,
                             'median_total_seconds': statistics.median(totals),
                             'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
            print(json.dumps({'split': split, 'path': name, **results[name]}), flush=True)
        pairs = list(zip(outputs['reference'], outputs['shared_cached'], strict=True))
        results['probability_max_abs'] = max(float((x-y).abs().max()) for x, y in pairs)
        results['prediction_flips'] = sum(int(x.argmax()) != int(y.argmax()) for x, y in pairs)
        # Verify physical execution separately, so tracing does not perturb timings.
        calls = []
        original = model._run_rows

        def traced(rows, **kwargs):
            calls.append({'tokens': sum(map(len, rows)), 'padded_tokens': len(rows)*max(map(len, rows)),
                          'offset': kwargs.get('offset', 0)})
            return original(rows, **kwargs)

        model._run_rows = traced
        try:
            for e in sample:
                predict(e, True)
        finally:
            model._run_rows = original
        expected = sum(len(e.state) + sum(len(q.prefix) + sum(map(len, q.candidates)) for q in e.questions)
                       for e in sample)
        assert sum(c['tokens'] for c in calls) == expected
        results.update({'record_ids': [e.record.id for e in sample], 'questions': len(pairs),
                        'shared_computed_tokens': expected, 'full_leaf_tokens': sum(e.forward_tokens for e in sample),
                        'shared_backbone_calls': len(calls)})
        report['groups'][split] = results
        write_json(out / 'report.json', report)

    # Full-set accuracy against the already saved reference-path evaluation.
    reference = {(r['record_id'], r['question_id']): r for r in json_rows(a.reference_predictions)}
    expected_keys = {(e.record.id, q.question.id) for e in encoded for q in e.questions}
    assert reference.keys() == expected_keys
    flips, correct, reference_correct, max_diff = 0, 0, 0, 0.0
    with (out / 'predictions.jsonl').open('w') as stream:
        for i, e in enumerate(encoded):
            for q, probability in zip(e.record.questions, predict(e, True), strict=True):
                values = probability.float().cpu()
                assert torch.isfinite(values).all() and abs(float(values.sum())-1) < 1e-5
                ids = [c.id for c in q.candidates]
                old = reference[(e.record.id, q.id)]
                assert list(old['probabilities']) == ids and old['label'] == q.label
                prediction = ids[int(values.argmax())]
                diff = float((values - torch.tensor(list(old['probabilities'].values()))).abs().max())
                max_diff = max(max_diff, diff)
                flips += prediction != old['prediction']
                correct += prediction == q.label
                reference_correct += old['correct']
                stream.write(json.dumps({'record_id': e.record.id, 'question_id': q.id, 'label': q.label,
                    'prediction': prediction, 'probabilities': dict(zip(ids, values.tolist())),
                    'correct': prediction == q.label, 'reference_prediction': old['prediction'],
                    'probability_max_abs': diff}) + '\n')
            if (i+1) % 100 == 0:
                stream.flush()
                print(json.dumps({'full_cached_records': i+1, 'correct': correct, 'flips': flips}), flush=True)
    n = len(reference)
    report['full_quality'] = {'questions': n, 'correct': correct, 'reference_correct': reference_correct,
                              'accuracy': correct/n, 'reference_accuracy': reference_correct/n,
                              'prediction_flips': flips, 'probability_max_abs': max_diff}
    write_json(out / 'report.json', report)
    print(json.dumps(report['full_quality']), flush=True)


if __name__ == '__main__':
    main()
