# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
from dataclasses import replace
import json

import pytest

from qev.data import file_hash, json_rows, load_records
from qev.encoding import leaf_rows
from qev.none_data import build_dataset, eligible, make_pair
from qev.schema import Candidate


def test_pair_is_counterfactual_without_changing_other_paths(record, encoder):
    q = record.questions[0]
    original = record.to_dict()
    (present, absent), lineage = make_pair(record, q, 'test', seed=17, wording=0)
    assert record.to_dict() == original
    p, a = present.questions[0], absent.questions[0]
    assert present.state == absent.state == record.state
    assert present.group_id == absent.group_id == record.group_id
    assert p.id == a.id == q.id and p.instructions == a.instructions == q.instructions
    assert p.label == q.label and a.label == lineage['none_id']
    assert tuple(c for c in p.candidates if c.id != q.label) == a.candidates
    for question in (p, a):
        assert question.target == tuple(float(c.id == question.label) for c in question.candidates)
    ep, ea = encoder(present), encoder(absent)
    assert ep.state == ea.state and ep.questions[0].prefix == ea.questions[0].prefix
    rp = dict(zip([c.id for c in p.candidates], leaf_rows(ep)[0]))
    ra = dict(zip([c.id for c in a.candidates], leaf_rows(ea)[0]))
    assert all(tokens == rp[cid] for cid, tokens in ra.items())
    assert ((present, absent), lineage) == make_pair(record, q, 'test', seed=17, wording=0)


def test_ineligible_targets_and_existing_none_are_not_rewritten(record):
    q = record.questions[0]
    assert eligible(q)
    assert not eligible(record.questions[1])
    assert not eligible(replace(q, type='score'))
    assert not eligible(replace(q, target=(0.8, 0.1, 0.1)))
    assert not eligible(replace(q, target=(0.0, 1.0, 0.0)))
    with_none = replace(q, candidates=q.candidates + (Candidate('custom', 'None of the above'),),
                        target=q.target + (0.0,))
    assert not eligible(with_none)
    with pytest.raises(ValueError, match='hard-label Choice'):
        make_pair(record, replace(q, target=(1/3,)*3), 'test', seed=17, wording=0)


def assets(tmp_path, record):
    base = tmp_path / 'base'
    base.mkdir()
    train = [replace(record, id=f't{i}', group_id=f'g{i}', source='banking77' if i<3 else 'agnews',
                     state=f'Training item {i}: blue.\u2028Literal separator.') for i in range(6)]
    dev = [replace(record, id='held', group_id='held', state='Held-out independent context.')]
    entries = {}
    for split, records, role in [('train', train, 'train'), ('dev', dev, 'development')]:
        path = base / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r.to_dict(), ensure_ascii=False)+'\n' for r in records))
        entries[split] = {'file':path.name,'sha256':file_hash(path),'records':len(records),
                          'questions':sum(len(r.questions) for r in records),'role':role,
                          'by_source':{'banking77':3,'agnews':3} if split=='train' else {'synthetic':1}}
    (base/'manifest.json').write_text(json.dumps({'schema':'qev.data.v1','files':entries}))
    return base, train


def test_complete_mixture_preserves_original_and_holdout_bytes(tmp_path, record, encoder):
    base, original = assets(tmp_path, record)
    before = (base/'train.jsonl').read_bytes()
    out = tmp_path/'mixed'
    m = build_dataset(base, out, encoder, pairs=4, seed=17, families=('banking77','agnews'))
    assert (out/'train.jsonl').read_bytes().startswith(before)
    assert (base/'train.jsonl').read_bytes() == before
    assert (out/'dev.jsonl').read_bytes() == (base/'dev.jsonl').read_bytes()
    mixed, loaded = load_records(out, 'train', training=True)
    assert mixed[:len(original)] == original
    assert len(mixed) == 14 and m['files']['train']['questions'] == 20
    assert m['augmentation']['none_gold'] == m['augmentation']['original_gold_retained'] == 4
    assert m['augmentation']['unique_parent_groups'] == 4
    parents = list(json_rows(out/'none_pairs.provenance.jsonl'))
    assert len(parents)==4 and all(x['parent_record_id'] != 'held' for x in parents)
    out2 = tmp_path/'repeat'
    build_dataset(base, out2, encoder, pairs=4, seed=17, families=('banking77','agnews'))
    assert (out/'none_pairs.jsonl').read_bytes() == (out2/'none_pairs.jsonl').read_bytes()
    with pytest.raises(FileExistsError):
        build_dataset(base, out, encoder, pairs=4, families=('banking77','agnews'))


def test_existing_group_leakage_aborts_before_publishing(tmp_path, record, encoder):
    base, _ = assets(tmp_path, record)
    bad = replace(record, id='held', group_id='g0', state='Different context but same source group.')
    path = base/'dev.jsonl'
    path.write_text(json.dumps(bad.to_dict())+'\n')
    manifest=json.loads((base/'manifest.json').read_text())
    manifest['files']['dev']['sha256']=file_hash(path)
    (base/'manifest.json').write_text(json.dumps(manifest))
    out=tmp_path/'bad'
    with pytest.raises(ValueError, match='overlap held-out'):
        build_dataset(base, out, encoder, pairs=2, families=('banking77','agnews'))
    assert not out.exists()
