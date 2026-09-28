# Adapted for Qev in 2026; see NOTICE and provenance.json.
"""Checkpoint-driven inference for TypeSafe-shaped JSONL requests."""
import argparse
import json
from pathlib import Path
from dataclasses import replace

from .checkpoint import load_model
from .choice_policy import none_candidate
from .data import json_rows
from .evaluate import checked_probabilities
from .schema import typed_record
from .encoding import Encoder


def predict_request(model, encoder, request, *, request_id='request', cached=True, tree=False):
    encoded = encoder(typed_record(request, source='inference', record_id=request_id))
    predictions = model.predict(encoded, cached=cached, tree=tree)
    output = {}
    for q, p in zip(encoded.record.questions, predictions, strict=True):
        ids = [c.id for c in q.candidates]
        values = checked_probabilities(p, len(ids))
        label = ids[int(values.argmax())]
        none = none_candidate(q)
        output[q.id] = {'type': q.type, 'prediction': label, 'probabilities': dict(zip(ids, values.tolist())),
                        'none_candidate_id': none.id if none else None,
                        'predicted_none': bool(none and label == none.id)}
        if q.type == 'choice':
            output[q.id]['choice'] = label
        elif q.type == 'noul':
            output[q.id]['noul'] = output[q.id]['probabilities']['true']
        elif q.type == 'score':
            output[q.id]['score'] = sum(i * value for i, value in enumerate(values.tolist()))
    return output


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--revision', help='pinned Hub checkpoint revision')
    p.add_argument('--base', help='local base-model cache override')
    p.add_argument('--weights-dtype', choices=['fp32', 'checkpoint'], default='checkpoint')
    p.add_argument('--max-state', type=int)
    p.add_argument('--max-path', type=int)
    execution = p.add_mutually_exclusive_group()
    execution.add_argument('--reference', action='store_true')
    execution.add_argument('--tree', action='store_true')
    a = p.parse_args()
    model, tokenizer, encoder, _ = load_model(a.checkpoint, a.device, revision=a.revision, base=a.base,
                        weights_dtype='fp32' if a.weights_dtype == 'fp32' else None)
    if a.max_state or a.max_path:
        encoder = Encoder(tokenizer, replace(encoder.limits,
                          max_state=a.max_state or encoder.limits.max_state,
                          max_path=a.max_path or encoder.limits.max_path),
                          choice_none_policy=encoder.choice_none_policy)
    model.prepare_inference()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(a.out).open('x', encoding='utf-8') as stream:
        for i, request in enumerate(json_rows(a.input)):
            rid = str(request.get('id', i))
            result = predict_request(model, encoder, request, request_id=rid, cached=not a.reference, tree=a.tree)
            stream.write(json.dumps({'id': rid, 'questions': result}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
