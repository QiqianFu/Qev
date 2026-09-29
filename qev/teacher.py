# SPDX-License-Identifier: Apache-2.0
"""Prepare teacher probabilities and responses for Qev distillation."""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import tempfile

from .counterfactual import blind_record, digest, edits
from .data import load_records, write_json
from .distillation import SCHEMA, probabilities, question_key, training_views, view_protocol
from .schema import Record


def write_rows(path, rows):
    with Path(path).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def teacher_output(model, encoder, record, capture=False):
    """Use the same reference forward for probabilities and head representations."""
    import torch
    from contextlib import nullcontext
    from .representation_response import CaptureTaskStates
    if len(record.questions) != 1 or any(q.label is not None or q.target is not None for q in record.questions):
        raise ValueError('teacher input must be one question without training targets')
    context = CaptureTaskStates(model.head) if capture else nullcontext()
    with torch.inference_mode(), context:
        logits = model.head(*model.reference_features(encoder(record))[0]).float().cpu().tolist()
    probabilities(logits)  # Reject nonfinite values before writing a cache.
    features = context.features[0].detach().cpu() if capture else None
    return logits, features


def training_records(data, config):
    main = load_records(data, 'train', training=True)[0]
    split = config['training'].get('late_split')
    late = load_records(data, split, training=True)[0] if split else []
    return main, late


def cache_logits(data, out, config, student_encoder, teacher, teacher_encoder):
    """Cache every original and augmented input the student trainer can use."""
    main, late = training_records(data, config)
    encoded = [student_encoder(record) for record in main + late]
    seen, rows = set(), []
    for view in training_views(encoded, len(main), student_encoder, config):
        for question in (q.question for q in view.questions):
            key = question_key(view.record.state, question)
            if key in seen:
                continue
            seen.add(key)
            record = blind_record(view.record, question)
            logits, _ = teacher_output(teacher, teacher_encoder, record)
            rows.append({'key': key, 'candidate_ids': [c.id for c in question.candidates], 'logits': logits})
    write_rows(out / 'logits.jsonl', rows)
    write_json(out / 'manifest.json', {
        'schema': SCHEMA, 'complete': True, 'temperature': 1,
        'view_protocol': view_protocol(config), 'questions': len(rows),
        'shards': [{'file': 'logits.jsonl', 'questions': len(rows)}],
    })
    return {'questions': len(rows)}


def split_response_data(replay, pairs, seed):
    """Keep all views of a parent in one split and remove input collisions."""
    probe = [row for row in pairs if int(digest([seed, row['group_id']])[:8], 16) % 5 == 0]
    groups = {row['group_id'] for row in probe}
    keys = {question_key(Record.from_dict(row[k]).state, Record.from_dict(row[k]).questions[0])
            for row in probe for k in ('original', 'edited')}
    train = [row for row in pairs if row['group_id'] not in groups and not any(
        question_key(Record.from_dict(row[k]).state, Record.from_dict(row[k]).questions[0]) in keys
        for k in ('original', 'edited'))]
    replay = [row for row in replay if not groups.intersection(row['parent_groups']) and row['id'] not in keys]
    return replay, train, probe


def cache_responses(data, out, config, student_encoder, teacher, teacher_encoder, *,
                    seed=17, confidence=.8, exclude_sources=('world_knowledge/',), max_candidates=8):
    """Create label-blind local edits and collect soft and representation targets."""
    from .encoding import ContextOverflow
    from .representation_response import response_geometry
    if not 0 <= confidence < 1:
        raise ValueError('confidence must lie in [0, 1)')
    main, late = training_records(data, config)
    originals, parent_groups = {}, defaultdict(set)
    for record in main + late:
        if record.source.startswith(tuple(exclude_sources)):
            continue
        for question in record.questions:
            blind = blind_record(record, question)
            originals.setdefault(blind.id, blind)
            parent_groups[blind.id].add(record.group_id)
    replay_rows, pair_rows, seen_pairs = [], [], set()
    for original in originals.values():
        try:
            student_encoder(original)
            original_logits, original_features = teacher_output(teacher, teacher_encoder, original, capture=True)
        except ContextOverflow:
            continue
        replay_rows.append({'id': original.id, 'parent_groups': sorted(parent_groups[original.id]),
                            'record': original.to_dict(), 'teacher_logits': original_logits})
        question = original.questions[0]
        if question.type not in {'choice', 'noul'} or not 2 <= len(question.candidates) <= max_candidates:
            continue
        if max(probabilities(original_logits)) <= confidence:
            continue
        for patch, state in edits(original.state, seed=seed, key=original.id):
            edited = replace(original, state=state, id=question_key(state, question))
            pair_id = digest([original.id, edited.id])
            if pair_id in seen_pairs:
                continue
            seen_pairs.add(pair_id)
            try:
                student_encoder(edited)
                logits, features = teacher_output(teacher, teacher_encoder, edited, capture=True)
            except ContextOverflow:
                continue
            if max(probabilities(logits)) <= confidence:
                continue
            geometry = response_geometry(original_features.double(), features.double())['anchored_response']
            # The selected continuation used log probabilities for paired targets.
            # Keep its small-probability floor and temperature convention.
            original_target = [math.log(max(p, 1e-30)) for p in probabilities(original_logits)]
            edited_target = [math.log(max(p, 1e-30)) for p in probabilities(logits)]
            pair_rows.append({'id': pair_id, 'group_id': original.group_id,
                              'original': original.to_dict(), 'edited': edited.to_dict(),
                              'teacher_original_logits': original_target, 'teacher_edited_logits': edited_target,
                              'response': geometry.tolist(), 'edit': patch['operator']})
    replay_rows, train, probe = split_response_data(replay_rows, pair_rows, seed)
    if not replay_rows or not train:
        raise ValueError('not enough accepted examples; add more training groups or adjust the confidence threshold')
    scale = sum(sum(x*x for row in pair['response'] for x in row) / len(pair['response'])**2
                for pair in train) / len(train)
    files = {}
    for name, rows, role in [('replay', replay_rows, 'train'), ('pairs', train, 'train'),
                              ('probe', probe, 'teacher_response_diagnostic')]:
        write_rows(out / f'{name}.jsonl', rows)
        files[name] = {'file': f'{name}.jsonl', 'records': len(rows), 'role': role}
    write_json(out / 'manifest.json', {'schema': 'qev.response-data.v1', 'files': files,
                                      'response_scale': max(scale, 1e-6), 'seed': seed,
                                      'confidence': confidence, 'excluded_sources': list(exclude_sources)})
    return {name: entry['records'] for name, entry in files.items()}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=['logits', 'responses'])
    p.add_argument('--teacher', default='AustinFu/Qev-9B')
    p.add_argument('--data', required=True)
    p.add_argument('--config', required=True, help='student training configuration')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--confidence', type=float, default=.8)
    p.add_argument('--exclude-source', action='append', default=None)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    from transformers import AutoTokenizer
    from .checkpoint import load_model
    from .encoding import Encoder, Limits
    config = json.loads(Path(a.config).read_text())
    spec = config['model']
    tokenizer = AutoTokenizer.from_pretrained(spec['base'], revision=spec.get('revision'))
    student_encoder = Encoder(tokenizer, Limits(**config.get('limits', {})),
                              choice_none_policy=spec.get('choice_none_policy', 'as-provided'))
    model, _, teacher_encoder, _ = load_model(a.teacher, a.device)
    model.eval()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + a.out.name + '-', dir=a.out.parent))
    try:
        if a.mode == 'logits':
            summary = cache_logits(a.data, stage, config, student_encoder, model, teacher_encoder)
        else:
            summary = cache_responses(a.data, stage, config, student_encoder, model, teacher_encoder,
                                      seed=config['training'].get('seed', 17), confidence=a.confidence,
                                      exclude_sources=a.exclude_source if a.exclude_source is not None else ['world_knowledge/'])
        stage.rename(a.out)
        print(json.dumps(summary))
    finally:
        if stage.exists():
            shutil.rmtree(stage)


if __name__ == '__main__':
    main()
