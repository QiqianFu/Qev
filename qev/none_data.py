# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Build a versioned train-only mixture of balanced none-of-the-above pairs."""
import argparse
from collections import Counter, defaultdict, deque
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import re
import shutil

from .data import json_rows, load_records, write_json
from .encoding import ContextOverflow, Encoder, Limits
from .schema import Candidate, Record

FAMILY = {
    'banking77': 'banking77', 'legacy-datasets/banking77': 'banking77',
    'agnews': 'agnews', 'fancyzhx/ag_news': 'agnews',
    'mnli': 'mnli', 'nyu-mll/multi_nli': 'mnli',
    'trec': 'trec', 'CogComp/trec': 'trec',
    'dbpedia14': 'dbpedia14', 'fancyzhx/dbpedia_14': 'dbpedia14',
}
FAMILIES = ('banking77', 'agnews', 'mnli', 'trec', 'dbpedia14')
NONE_OPTIONS = (
    ('none_of_these', 'None of these options is correct.'),
    ('none_of_the_above', 'None of the listed options matches the answer.'),
    ('not_listed', 'The correct answer is not among the listed options.'),
    ('no_match', 'None of the available options applies.'),
    ('other', 'An answer not covered by any of the other options.'),
)
NONE_IDS = {k for k, _ in NONE_OPTIONS} | {'none', 'unknown', 'none_of_above', 'abstain'}
NONE_TEXT = re.compile(r'none[-_ ]of|none of|not listed|not among|no (?:listed|provided|available) option', re.I)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def input_signature(record, question):
    # Ignore candidate order as well as all targets and metadata.
    return digest([record.state, question.type, question.instructions,
                   sorted((c.id, c.text) for c in question.candidates)])


def eligible(question):
    if question.type != 'choice' or len(question.candidates) < 3 or question.label is None:
        return False
    ids = [c.id for c in question.candidates]
    if question.target != tuple(float(k == question.label) for k in ids):
        return False  # Never rewrite soft targets, or inconsistent hard labels.
    return not any(c.id.lower() in NONE_IDS or NONE_TEXT.search(c.id + ' ' + c.text)
                   for c in question.candidates)


def make_pair(record, question, family, *, seed, wording):
    if not eligible(question):
        raise ValueError('none pairs require hard-label Choice with >=3 options and no existing none option')
    none_id, text = NONE_OPTIONS[wording % len(NONE_OPTIONS)]
    pair_id = 'none3000-v1/' + digest([seed, record.id, question.id])[:24]
    candidates = list(question.candidates) + [Candidate(none_id, none_id + ': ' + text)]
    rng = random.Random(int(digest([seed, record.id, question.id, 'candidate-order']), 16))
    rng.shuffle(candidates)
    present = replace(question, candidates=tuple(candidates),
                      target=tuple(float(c.id == question.label) for c in candidates))
    absent_candidates = tuple(c for c in candidates if c.id != question.label)
    absent = replace(question, candidates=absent_candidates, label=none_id,
                     target=tuple(float(c.id == none_id) for c in absent_candidates))
    records = tuple(Record(pair_id + '/' + variant, record.group_id, 'none_pair/' + family,
                           record.state, (q,))
                    for variant, q in [('present', present), ('absent', absent)])
    provenance = {'pair_id': pair_id, 'parent_record_id': record.id, 'parent_question_id': question.id,
                  'parent_group_id': record.group_id, 'parent_source': record.source, 'family': family,
                  'original_label': question.label, 'none_id': none_id, 'none_text': text,
                  'parent_candidate_count': len(question.candidates),
                  'present_record_id': records[0].id, 'absent_record_id': records[1].id}
    return records, provenance


def build_dataset(base, out, encoder, *, pairs=1500, seed=17, families=FAMILIES):
    base, out = Path(base), Path(out)
    if out.exists():
        raise FileExistsError(f'refusing to overwrite {out}')
    if pairs < 1 or not families or pairs % len(families):
        raise ValueError('pair count must be positive and divisible by the number of families')
    train, manifest = load_records(base, 'train', training=True)
    held_ids, held_groups, held_inputs = set(), set(), set()
    for split, entry in manifest['files'].items():
        if entry['role'] == 'train':
            continue
        records, _ = load_records(base, split)
        held_ids.update(r.id for r in records)
        held_groups.update(r.group_id for r in records)
        held_inputs.update(input_signature(r, q) for r in records for q in r.questions)
    train_inputs = {input_signature(r, q) for r in train for q in r.questions}
    base_overlap = {'record_ids': len({r.id for r in train} & held_ids),
                    'group_ids': len({r.group_id for r in train} & held_groups),
                    'order_invariant_question_inputs': len(train_inputs & held_inputs)}
    if any(base_overlap.values()):
        raise ValueError(f'base training data overlap held-out data: {base_overlap}')
    pools = defaultdict(lambda: defaultdict(list))
    for record in train:
        family = FAMILY.get(record.source)
        if family in families:
            for question in record.questions:
                if eligible(question):
                    pools[family][question.label].append((record, question))
    children, lineage, excluded = [], [], Counter()
    selected_groups, parent_inputs, child_inputs = set(), set(), set()
    maxima = Counter()
    for family in families:
        buckets = {label: deque(sorted(items, key=lambda item: digest([seed, item[0].id, item[1].id])))
                   for label, items in pools[family].items()}
        labels = sorted(buckets, key=lambda label: digest([seed, family, label]))
        accepted, quota = 0, pairs // len(families)
        while accepted < quota:
            if not any(buckets.values()):
                raise ValueError(f'not enough eligible, distinct, in-limit parents for {family}: {accepted}/{quota}')
            for label in labels:
                if accepted == quota:
                    break
                bucket = buckets[label]
                while bucket:
                    record, question = bucket.popleft()
                    signature = input_signature(record, question)
                    if record.group_id in selected_groups or signature in parent_inputs:
                        excluded['duplicate_parent_group_or_input'] += 1
                        continue
                    pair, origin = make_pair(record, question, family, seed=seed, wording=len(lineage))
                    signatures = {input_signature(r, r.questions[0]) for r in pair}
                    if any(r.id in held_ids or r.group_id in held_groups for r in pair) or signatures & held_inputs:
                        excluded['heldout_collision'] += 1
                        continue
                    if len(signatures) != 2 or signatures & (train_inputs | child_inputs):
                        excluded['duplicate_generated_input'] += 1
                        continue
                    try:
                        encoded = [encoder(r) for r in pair]
                    except ContextOverflow:
                        excluded['token_or_candidate_limit'] += 1
                        continue
                    for r in encoded:
                        maxima['state_tokens'] = max(maxima['state_tokens'], len(r.state))
                        for q in r.questions:
                            maxima['question_tokens'] = max(maxima['question_tokens'], len(q.prefix))
                            maxima['candidates'] = max(maxima['candidates'], len(q.candidates))
                            for c in q.candidates:
                                maxima['candidate_tokens'] = max(maxima['candidate_tokens'], len(c))
                                maxima['path_tokens'] = max(maxima['path_tokens'], len(r.state) + len(q.prefix) + len(c))
                    children.extend(pair)
                    lineage.append(origin)
                    selected_groups.add(record.group_id)
                    parent_inputs.add(signature)
                    child_inputs.update(signatures)
                    accepted += 1
                    break
    assert len(children) == 2 * pairs and len({r.id for r in children}) == len(children)
    assert len(selected_groups) == pairs
    out.mkdir(parents=True)
    # Preserve original data byte for byte. The training sampler shuffles the full mixture every epoch.
    for entry in manifest['files'].values():
        shutil.copyfile(base / entry['file'], out / entry['file'])
    original_train = base / manifest['files']['train']['file']
    train_path = out / manifest['files']['train']['file']
    if not original_train.read_bytes().endswith(b'\n'):
        raise ValueError('base JSONL must end with a newline before appending')
    with (out / 'none_pairs.jsonl').open('w', encoding='utf-8') as stream, train_path.open('a', encoding='utf-8') as mixed:
        for record in children:
            line = json.dumps(record.to_dict(), ensure_ascii=False) + '\n'
            stream.write(line)
            mixed.write(line)
    with (out / 'none_pairs.provenance.jsonl').open('w', encoding='utf-8') as stream:
        for origin in lineage:
            stream.write(json.dumps(origin, ensure_ascii=False) + '\n')
    entry = manifest['files']['train']
    counts = Counter(entry['by_source']) + Counter(r.source for r in children)
    new_manifest = {**manifest, 'recipe': 'choice-none-pairs', 'files': dict(manifest['files'])}
    new_manifest['files']['train'] = {**entry,
                                      'records': entry['records'] + len(children),
                                      'questions': entry['questions'] + len(children), 'by_source': dict(counts)}
    new_manifest['files']['train'].pop('sha256', None)
    augmentation = {'pairs': pairs, 'new_records': len(children), 'new_questions': len(children),
                    'none_gold': pairs, 'original_gold_retained': pairs, 'seed': seed,
                    'families': dict(Counter(x['family'] for x in lineage)),
                    'parent_sources': dict(Counter(x['parent_source'] for x in lineage)),
                    'wordings': dict(Counter(x['none_id'] for x in lineage)),
                    'unique_parent_groups': len(selected_groups), 'excluded': dict(excluded),
                    'limits': asdict(encoder.limits), 'observed_maxima': dict(maxima),
                    'original_train_bytes': original_train.stat().st_size,
                    'base_train_holdout_overlap': base_overlap, 'generated_holdout_overlap': 0}
    new_manifest['augmentation'] = augmentation
    new_manifest['notes'] = list(manifest.get('notes', [])) + [
        'All original training records retained; exactly 1500 present/absent pairs added by default.',
        'Pairs are generated only from hard-label training Choice questions; original soft targets remain unchanged.',
        'Held-out files are byte-identical; previously evaluated tests are regression sets, not new untouched evidence.',
        'For a fresh base/LoRA/head/optimizer run; do not resume old-dataset checkpoints.']
    write_json(out / 'manifest.json', new_manifest)
    return new_manifest


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--base-data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--tokenizer', required=True, help='local, pinned base tokenizer (no model weights loaded)')
    ap.add_argument('--pairs', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=17)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    config = json.loads(Path(a.config).read_text())
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    encoder = Encoder(tokenizer, Limits(**config['limits']))
    manifest = build_dataset(a.base_data, a.out, encoder, pairs=a.pairs, seed=a.seed)
    # Record which tokenizer was used for strict context admission.
    manifest['augmentation']['base_revision'] = config['model']['revision']
    write_json(Path(a.out) / 'manifest.json', manifest)
    print(json.dumps({'train': manifest['files']['train'], 'augmentation': manifest['augmentation']}, indent=2))


if __name__ == '__main__':
    main()
