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
    if hashlib.sha256(evidence.read_bytes()).hexdigest()!=provenance['predictions_sha256']:
        errors.append('frozen JevBench predictions changed')
    rows=[json.loads(line) for line in evidence.read_text().splitlines()]
    if len(rows)!=231 or sum(r['correct'] for r in rows)!=188:
        errors.append('Qev JevBench numerator/denominator changed')
    for row in rows:
        probs=row['probabilities'];values=list(probs.values())
        if not all(math.isfinite(p) and 0<=p<=1 for p in values) or not math.isclose(sum(values),1,abs_tol=1e-5):
            errors.append(f"invalid probabilities: {row['record_id']}")
        if probs[row['prediction']]!=max(values) or row['correct']!=(row['prediction']==row['label']):
            errors.append(f"prediction/label mismatch: {row['record_id']}")
    for error in errors:print(error)
    print(f'{len(docs)} documents, {checked_links} local links, {len(pyfiles)} Python files, '
          f'231 frozen predictions; {len(errors)} errors')
    return bool(errors)


if __name__=='__main__':
    raise SystemExit(main())
