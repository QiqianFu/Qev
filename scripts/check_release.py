#!/usr/bin/env python3
"""Offline source, documentation and frozen-result consistency checks."""
import ast
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT=Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.targets=[]
    def handle_starttag(self,tag,attrs):
        self.targets.extend(v for k,v in attrs if k in {'href','src'} and v)


def main():
    errors=[];checked_links=0
    docs=[*ROOT.glob('*.md'),*(ROOT/'docs').rglob('*.md'),*(ROOT/'docs').rglob('*.html')]
    docs += [ROOT/p/'README.md' for p in ['configs','examples']]
    docs += list((ROOT/'results').rglob('README.md'))
    for p in docs:
        text=p.read_text()
        if p.suffix=='.md':text=re.sub(r'```.*?```','',text,flags=re.S)
        hrefs=re.findall(r'!?\[[^\]\n]*\]\(([^\s)]+)\)',text)
        parser=Links();parser.feed(text);hrefs+=parser.targets
        for href in hrefs:
            u=urlsplit(href.strip('<>'))
            if u.scheme or u.netloc or not u.path:continue
            target=(p.parent/unquote(u.path)).resolve()
            checked_links+=1
            if not target.is_relative_to(ROOT) or not target.exists():
                errors.append(f'{p.relative_to(ROOT)}: missing or external local target: {href}')
    pyfiles=[*(ROOT/'qev').glob('*.py'),*(ROOT/'scripts').glob('*.py'),*(ROOT/'tests').glob('*.py')]
    for p in pyfiles:
        try:ast.parse(p.read_text(),filename=str(p))
        except SyntaxError as exc:errors.append(str(exc))
        # The distribution must not import or default to the research filesystem.
        if re.search(r'/(?:home|shared|scratch)/[A-Za-z0-9_.-]+/',p.read_text()):
            errors.append(f'{p.relative_to(ROOT)}: machine-specific path')
    for p in (ROOT/'configs').glob('*.json'):
        value=json.loads(p.read_text())
        if 'model' in value and str(value['model']['base']).startswith('/'):
            errors.append(f'{p.name}: machine-specific base')
    results=json.loads((ROOT/'results/benchmarks.json').read_text())
    for name,entry in results['results'].items():
        for model,row in entry['models'].items():
            if not 0<=row['correct']<=entry['total'] or not math.isclose(row['accuracy'],row['correct']/entry['total']):
                errors.append(f'{name}/{model}: inconsistent metric')
    evidence=ROOT/'results/qev-9b/jevbench-predictions.jsonl'
    provenance=json.loads((evidence.parent/'provenance.json').read_text())
    if provenance.get('predictions_sha256') and hashlib.sha256(evidence.read_bytes()).hexdigest()!=provenance['predictions_sha256']:
        errors.append('frozen JevBench predictions changed')
    rows=[json.loads(line) for line in evidence.read_text().splitlines()]
    benchmark=results['results']['jevbench_public']
    if len(rows)!=benchmark['total'] or sum(r['correct'] for r in rows)!=benchmark['models']['qev_9b']['correct']:
        errors.append('Qev JevBench numerator/denominator changed')
    for row in rows:
        probs=row['probabilities'];values=list(probs.values())
        if not all(math.isfinite(p) and 0<=p<=1 for p in values) or not math.isclose(sum(values),1,abs_tol=1e-5):
            errors.append(f"invalid probabilities: {row['record_id']}")
        if probs[row['prediction']]!=max(values) or row['correct']!=(row['prediction']==row['label']):
            errors.append(f"prediction/label mismatch: {row['record_id']}")
    prediction_count=len(rows)
    for folder, model in [('qev-4b', 'qev_4b'), ('qev-2b', 'qev_2b'), ('qwen3.5-2b-base', 'qwen35_2b_base')]:
        records=[json.loads(line) for line in (ROOT/'results'/folder/'jevbench-predictions.jsonl').read_text().splitlines()]
        prediction_count+=len(records)
        if len(records)!=benchmark['total'] or sum(row['correct'] for row in records)!=benchmark['models'][model]['correct']:
            errors.append(f'{folder}: inconsistent JevBench results')
        for row in records:
            ps=row['probabilities']
            if not all(math.isfinite(p) and 0<=p<=1 for p in ps.values()) or not math.isclose(sum(ps.values()),1,abs_tol=1e-5):
                errors.append(f'{folder}: invalid probabilities')
            if ps[row['prediction']]!=max(ps.values()) or row['correct']!=(row['prediction']==row['label']):
                errors.append(f'{folder}: inconsistent prediction')
    for folder,model in [('qev-4b','qev_4b'),('qev-9b','qev_9b')]:
        evaluation=json.loads((ROOT/'results'/folder/'evaluation.json').read_text())
        for scope,row in evaluation['matched_subsets'].items():
            expected=results['results'][scope]
            if row['total']!=expected['total'] or row['correct']!=expected['models'][model]['correct']:
                errors.append(f'{folder}/{scope}: full report and benchmark table differ')
    archived=ROOT/'results/history/qev-9b-v0.1.0'
    old_predictions=archived/'jevbench-predictions.jsonl'
    old_provenance=json.loads((archived/'provenance.json').read_text())
    if hashlib.sha256(old_predictions.read_bytes()).hexdigest()!=old_provenance['predictions_sha256']:
        errors.append('archived Qev-9B v0.1.0 predictions changed')
    prediction_count+=len(old_predictions.read_text().splitlines())
    for error in errors:print(error)
    print(f'{len(docs)} documents, {checked_links} local links, {len(pyfiles)} Python files, '
          f'{prediction_count} current and archived predictions; {len(errors)} errors')
    return bool(errors)


if __name__=='__main__':
    raise SystemExit(main())
