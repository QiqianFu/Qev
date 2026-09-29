# SPDX-License-Identifier: Apache-2.0
"""Read aligned teacher examples for response distillation."""
from dataclasses import replace
import json
import math
from pathlib import Path

from .data import json_rows
from .distillation import probabilities, question_key
from .schema import Record


def read_teacher_record(value, logits):
    record = Record.from_dict(value)
    if len(record.questions) != 1:
        raise ValueError('distillation examples must contain one question')
    question = record.questions[0]
    if question.label is not None or question.target is not None:
        raise ValueError('response distillation uses teacher targets, not original labels')
    if len(probabilities(logits)) != len(question.candidates):
        raise ValueError('teacher logits do not match the candidate count')
    return record


def load_response_data(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('schema') != 'qev.response-data.v1':
        raise ValueError('expected a dataset prepared by qev.teacher responses')
    rows = {}
    seen = set()
    keys, groups = {}, {}
    for split in ('replay', 'pairs', 'probe'):
        info = manifest['files'][split]
        role = 'teacher_response_diagnostic' if split == 'probe' else 'train'
        if info['role'] != role or Path(info['file']).name != info['file']:
            raise ValueError('invalid response data split')
        rows[split] = list(json_rows(directory / info['file']))
        if len(rows[split]) != info['records']:
            raise ValueError('response data record count mismatch')
        keys[split], groups[split] = set(), set()
        for row in rows[split]:
            if row['id'] in seen:
                raise ValueError('duplicate teacher example')
            seen.add(row['id'])
            if split == 'replay':
                record = read_teacher_record(row['record'], row['teacher_logits'])
                key = question_key(record.state, record.questions[0])
                if key != row['id'] or not row['parent_groups']:
                    raise ValueError('replay input or parent group mismatch')
                keys[split].add(key)
                groups[split].update(row['parent_groups'])
            else:
                original = read_teacher_record(row['original'], row['teacher_original_logits'])
                edited = read_teacher_record(row['edited'], row['teacher_edited_logits'])
                if (replace(original.questions[0], id='') != replace(edited.questions[0], id='')
                        or original.state == edited.state):
                    raise ValueError('pair must change context and preserve question and candidate order')
                if original.group_id != row['group_id'] or edited.group_id != row['group_id']:
                    raise ValueError('pair parent group mismatch')
                size = len(original.questions[0].candidates) + 1
                response = row['response']
                if (len(response) != size or any(len(line) != size for line in response)
                        or any(not math.isfinite(v) for line in response for v in line)):
                    raise ValueError('invalid teacher response matrix')
                keys[split].update(question_key(r.state, r.questions[0]) for r in (original, edited))
                groups[split].add(row['group_id'])
    if (keys['probe'] & (keys['pairs'] | keys['replay'])
            or groups['probe'] & (groups['pairs'] | groups['replay'])):
        raise ValueError('training and probe examples overlap')
    if not rows['replay'] or not rows['pairs']:
        raise ValueError('response distillation needs replay examples and edited pairs')
    scale = manifest['response_scale']
    if not math.isfinite(scale) or scale < 1e-6:
        raise ValueError('invalid response normalization scale')
    return rows, manifest
