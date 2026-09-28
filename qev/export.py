"""Create a portable inference checkpoint without optimizer state or base weights."""
import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile

from .data import file_hash, write_json


def export_checkpoint(checkpoint, out, *, base=None, revision=None):
    source, out = Path(checkpoint).expanduser(), Path(out).expanduser()
    if out.exists():
        raise FileExistsError(out)
    meta = json.loads((source / 'model.json').read_text())
    if meta.get('format') not in {'qev.checkpoint.v1', 'branchkev.checkpoint.v1'}:
        raise ValueError('expected a Qev or compatible research checkpoint')
    if not meta.get('adapter') or meta.get('backbone') == 'full':
        raise ValueError('this exporter packages LoRA checkpoints; full weights require a separate release')
    base = base or meta['spec']['base']
    revision = revision or meta['spec'].get('revision')
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', base) or not re.fullmatch(r'[0-9a-f]{40}', revision or ''):
        raise ValueError('portable export needs a Hub base ID and its full 40-character commit SHA')
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + out.name + '-', dir=out.parent))
    try:
        for folder in ('adapter', 'tokenizer'):
            (stage / folder).mkdir()
            for file in (source / folder).iterdir():
                if file.is_file() and file.suffix in {'.json', '.safetensors', '.txt', '.model'}:
                    shutil.copyfile(file, stage / folder / file.name)
        required = ['head.safetensors', 'adapter/adapter_model.safetensors',
                    'adapter/adapter_config.json', 'tokenizer/tokenizer_config.json']
        if meta['spec'].get('candidate_interaction', 'none') != 'none':
            required.append('joint.safetensors')
        for name in required:
            if '/' not in name:
                shutil.copyfile(source / name, stage / name)
            if not (stage / name).is_file():
                raise ValueError(f'missing inference artifact: {name}')
        adapter = json.loads((stage / 'adapter/adapter_config.json').read_text())
        adapter.update(base_model_name_or_path=base, revision=revision)
        write_json(stage / 'adapter/adapter_config.json', adapter)
        for p in (stage / 'tokenizer').glob('*.json'):
            if p.name != 'tokenizer.json':
                value = json.loads(p.read_text())
                for key in ('name_or_path', '_name_or_path'):
                    if key in value:
                        value[key] = base
                write_json(p, value)
        meta['format'] = 'qev.checkpoint.v1'
        meta['spec'].update(base=base, revision=revision)
        write_json(stage / 'model.json', meta)
        inventory = {str(p.relative_to(stage)): file_hash(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        write_json(stage / 'SHA256SUMS.json', {'source_model_json_sha256': file_hash(source / 'model.json'),
                                           'files': inventory})
        stage.rename(out)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return out


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--base')
    p.add_argument('--base-revision')
    a = p.parse_args()
    print(export_checkpoint(a.checkpoint, a.out, base=a.base, revision=a.base_revision))


if __name__ == '__main__':
    main()
