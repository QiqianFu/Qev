# Adapted for Qev in 2026; see NOTICE and provenance.json.
"""Same-GPU native/reference/shared/merged inference timing and paired quality."""
import argparse
import gc
import json
from pathlib import Path
import random
import statistics
import time

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from .base_evaluate import answer_codes, make_prompt, score_codes
from .checkpoint import load_model
from .data import file_hash, json_rows, load_records, write_json
from .encoding import Encoder, Limits
from .evaluate import checked_probabilities


def prepare(checkpoint, data, sample_size):
    meta = json.loads((checkpoint / 'model.json').read_text())
    tok = AutoTokenizer.from_pretrained(checkpoint / 'tokenizer', local_files_only=True)
    encoder = Encoder(tok, Limits(**meta['limits']),
                      choice_none_policy=meta['spec'].get('choice_none_policy', 'as-provided'))
    records, manifest = load_records(data, 'mmlupro')
    other, _ = load_records(data, 'scienthoon')
    assert all(manifest['files'][s]['role'] != 'test' for s in ('mmlupro', 'scienthoon'))
    encoded = [encoder(r) for r in records]
    extra = [encoder(r) for r in other]
    selected = set(random.Random(17).sample(range(len(encoded)), min(sample_size, len(encoded))))
    selected.update(max(range(len(encoded)), key=key) for key in (
        lambda i: len(encoded[i].state), lambda i: encoded[i].forward_tokens))
    groups = {'mmlupro': [encoded[i] for i in sorted(selected)],
              'scienthoon': sorted(extra, key=lambda e: e.forward_tokens, reverse=True)[:8]}
    native_inputs = {}
    descriptions = {}
    for split, sample in groups.items():
        descriptions[split] = []
        for e in sample:
            prompts = []
            for q in e.record.questions:
                assert q.label in [c.id for c in q.candidates]
                labels, texts, codes, suffix = answer_codes(tok, len(q.candidates))
                prompt = make_prompt(e.record, q, labels, suffix)
                ids = tok.encode(prompt, add_special_tokens=False)
                assert len(ids) < 8192 and all(len(c) == 1 for c in codes)
                for i in (0, len(codes)-1):
                    assert tok.encode(prompt+texts[i], add_special_tokens=False) == ids + list(codes[i])
                prompts.append((ids, codes))
            native_inputs[e.record.id] = prompts
            descriptions[split].append({'record_id': e.record.id, 'questions': len(e.questions),
                'candidate_counts': [len(q.candidates) for q in e.questions],
                'native_prompt_tokens': sum(len(ids) for ids, _ in prompts),
                'full_leaf_tokens': e.forward_tokens,
                'shared_tokens': len(e.state) + sum(len(q.prefix)+sum(map(len, q.candidates)) for q in e.questions)})
    return meta, encoded, groups, native_inputs, descriptions


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--reference-predictions', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--records', type=int, default=32)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    if min(a.records, a.repeats) < 1:
        p.error('records and repeats must be positive')
    a.out.mkdir(parents=True, exist_ok=False)
    meta, all_records, groups, native_inputs, descriptions = prepare(a.checkpoint, a.data, a.records)
    assert meta['spec']['weights_dtype'] == 'bf16'
    report = {'checkpoint': str(a.checkpoint), 'candidate_interaction': meta['spec'].get('candidate_interaction', 'none'),
        'weights_dtype': 'bf16', 'head_precision': 'fp32', 'seed': 17, 'repeats': a.repeats,
        'manifest_sha256': file_hash(a.data/'manifest.json'),
        'checkpoint_metadata_sha256': file_hash(a.checkpoint/'model.json'),
        'reference_predictions_sha256': file_hash(a.reference_predictions),
        'timing_scope': 'synchronized per-record wall time, tensor preparation and GPU probabilities included; CPU transfer/checks, loading, tokenization and compilation excluded',
        'native_protocol': 'Original BF16 LM head, one forward per question, single-token answer letters; multi-question records execute their questions serially.',
        'samples': descriptions, 'paths': {}, 'quality': {}}
    write_json(a.out/'report.json', report)
    if a.prepare_only:
        print(json.dumps({'stage': 'prepared', 'records': {k: len(v) for k,v in groups.items()}}))
        return
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    report.update({'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__})
    outputs = {}

    def measure(name, predict):
        result, path_outputs, rows = {}, {}, []
        for split, sample in groups.items():
            for e in sample:
                predict(e)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            totals, latencies, ps = [], [], []
            for rep in range(a.repeats):
                total = 0.0
                ps = []
                for e in sample:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    predicted = predict(e)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-start
                    total += elapsed
                    latencies.append(elapsed)
                    values = [checked_probabilities(v, len(q.candidates))
                              for q,v in zip(e.record.questions, predicted, strict=True)]
                    ps.extend(values)
                    rows.append({'path': name, 'split': split, 'record_id': e.record.id, 'repetition': rep,
                                 'seconds': elapsed, 'probabilities': [v.tolist() for v in values]})
                totals.append(total)
            ordered = sorted(latencies)
            labels = [next(i for i,c in enumerate(q.candidates) if c.id == q.label)
                      for e in sample for q in e.record.questions]
            result[split] = {'records': len(sample), 'questions': len(labels),
                'median_ms': 1000*statistics.median(ordered), 'mean_ms': 1000*statistics.mean(ordered),
                'p90_ms': 1000*ordered[int(.9*len(ordered))], 'total_seconds': totals,
                'median_total_seconds': statistics.median(totals),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'sample_correct': sum(int(v.argmax()) == label for v,label in zip(ps,labels,strict=True))}
            path_outputs[split] = ps
            print(json.dumps({'stage': 'timing', 'path': name, 'split': split, **result[split]}), flush=True)
        outputs[name] = path_outputs
        report['paths'][name] = result
        write_json(a.out/'report.json', report)
        with (a.out/f'{name}-timings.jsonl').open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row)+'\n')

    # Load one model at a time so memory figures refer to the measured arm.
    native = AutoModelForImageTextToText.from_pretrained(meta['spec']['base'], revision=meta['spec']['revision'],
        dtype=torch.bfloat16, attn_implementation='sdpa', trust_remote_code=False, local_files_only=True)
    native.model.visual = None
    native.requires_grad_(False).eval().to('cuda')
    measure('native', lambda e: [score_codes(native, ids, codes, 'cuda').softmax(-1)
                                 for ids,codes in native_inputs[e.record.id]])
    del native
    gc.collect()
    torch.cuda.empty_cache()
    model, _, _, _ = load_model(a.checkpoint, 'cuda')
    model.prepare_inference()
    measure('reference', lambda e: model.predict(e, cached=False))
    measure('shared_cached', lambda e: model.predict(e, cached=True))

    saved = {(r['record_id'],r['question_id']): r for r in json_rows(a.reference_predictions)}
    assert saved.keys() == {(e.record.id,q.id) for e in all_records for q in e.record.questions}

    def quality(name, comparator):
        correct = flips = 0
        max_diff = 0.0
        results = {}
        torch.cuda.reset_peak_memory_stats()
        with (a.out/f'{name}-predictions.jsonl').open('w') as stream:
            for i,e in enumerate(all_records):
                for q,prob in zip(e.record.questions, model.predict(e), strict=True):
                    ids = [c.id for c in q.candidates]
                    values = checked_probabilities(prob, len(ids))
                    old = comparator[(e.record.id,q.id)]
                    assert list(old['probabilities']) == ids and old['label'] == q.label
                    label = ids[int(values.argmax())]
                    diff = float((values-torch.tensor(list(old['probabilities'].values()))).abs().max())
                    max_diff = max(max_diff,diff)
                    flips += label != old['prediction']
                    correct += label == q.label
                    row = {'record_id': e.record.id, 'question_id': q.id, 'label': q.label, 'prediction': label,
                           'correct': label == q.label, 'probabilities': dict(zip(ids,values.tolist()))}
                    results[(e.record.id,q.id)] = row
                    stream.write(json.dumps(row)+'\n')
                if (i+1) % 200 == 0:
                    stream.flush()
                    print(json.dumps({'stage': 'quality', 'path': name, 'records': i+1, 'correct': correct,
                                      'allocated_bytes': torch.cuda.memory_allocated()}), flush=True)
        report['quality'][name] = {'n': len(results), 'correct': correct, 'accuracy': correct/len(results),
            'comparator': 'saved_reference' if name == 'shared_cached' else 'same_gpu_unmerged_cached',
            'prediction_flips': flips, 'probability_max_abs': max_diff,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        write_json(a.out/'report.json', report)
        return results

    unmerged = quality('shared_cached', saved)
    model.prepare_inference(merge_lora=True)
    assert model.lora_merged
    measure('merged_cached', lambda e: model.predict(e, cached=True))
    quality('merged_cached', unmerged)
    report['sample_differences'] = {}
    for left,right in (('reference','shared_cached'), ('shared_cached','merged_cached')):
        report['sample_differences'][f'{left}_vs_{right}'] = {
            split: {'probability_max_abs': max(float((x-y).abs().max()) for x,y in zip(outputs[left][split],outputs[right][split],strict=True)),
                    'prediction_flips': sum(int(x.argmax()) != int(y.argmax()) for x,y in zip(outputs[left][split],outputs[right][split],strict=True))}
            for split in groups}
    write_json(a.out/'report.json', report)
    print(json.dumps({'stage': 'complete', 'quality': report['quality']}), flush=True)


if __name__ == '__main__':
    main()
