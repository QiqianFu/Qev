"""Convert labelled request JSONL to group-disjoint, hash-verified train/dev data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile

from .data import file_hash, json_rows, write_json
from .schema import Record, typed_record


def read_requests(path, source):
    result = []
    for i, row in enumerate(json_rows(path)):
        if row.get('schema') in {'qev.record.v1', 'branchkev.record.v1'}:
            record = Record.from_dict(row)
        else:
            record = typed_record(row, source=source, record_id=str(row.get('id', f'{source}:{i}')),
                                  group_id=row.get('group_id'))
        if any(q.target is None for q in record.questions):
            raise ValueError(f'{record.id}: every training/validation question needs a label or target')
        result.append(record)
    if not result:
        raise ValueError(f'empty input: {path}')
    if len({r.id for r in result}) != len(result):
        raise ValueError(f'duplicate record IDs in {path}')
    return result


def input_key(record):
    payload = [record.state, [[q.type, q.instructions, sorted((c.id, c.text) for c in q.candidates)]
                             for q in record.questions]]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def build_dataset(input_path, out, *, validation=None, validation_fraction=0.1, seed=17):
    rows = read_requests(input_path, 'user-train')
    if validation:
        train, dev = rows, read_requests(validation, 'user-dev')
    else:
        if not 0 < validation_fraction < 1:
            raise ValueError('validation_fraction must lie between 0 and 1')
        groups = sorted({r.group_id for r in rows})
        if len(groups) < 2:
            raise ValueError('need at least two groups or an explicit validation file')
        random.Random(seed).shuffle(groups)
        held_out = set(groups[:min(len(groups)-1, max(1, round(len(groups)*validation_fraction)))])
        train, dev = [r for r in rows if r.group_id not in held_out], [r for r in rows if r.group_id in held_out]
    for key, name in ((lambda r: r.id, 'record IDs'), (lambda r: r.group_id, 'groups'), (input_key, 'inputs')):
        if {key(r) for r in train} & {key(r) for r in dev}:
            raise ValueError(f'train/dev overlap in {name}; deduplicate or assign related records one group_id')
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + out.name + '-', dir=out.parent))
    try:
        files = {}
        for split, records, role in [('train', train, 'train'), ('dev', dev, 'development')]:
            file = stage / f'{split}.jsonl'
            file.write_text(''.join(json.dumps(r.to_dict(), ensure_ascii=False) + '\n' for r in records))
            files[split] = {'file':file.name, 'role':role, 'sha256':file_hash(file),
                            'records':len(records), 'questions':sum(len(r.questions) for r in records),
                            'by_source':dict(Counter(r.source for r in records))}
        write_json(stage / 'manifest.json', {'schema':'qev.data.v1', 'files':files, 'seed':seed,
                   'input_sha256':file_hash(input_path),
                   'validation_sha256':file_hash(validation) if validation else None,
                   'split_policy':'explicit disjoint validation' if validation else 'seeded group split'})
        stage.rename(out)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return out


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--validation')
    p.add_argument('--validation-fraction', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=17)
    a = p.parse_args()
    print(build_dataset(a.input, a.out, validation=a.validation,
                        validation_fraction=a.validation_fraction, seed=a.seed))


if __name__ == '__main__':
    main()
