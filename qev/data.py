# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Local datasets and training/evaluation split checks."""
import hashlib
import json
from pathlib import Path

from .schema import Record


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_rows(path):
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc


def rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            yield from batch.to_pylist()
    else:
        yield from json_rows(path)


def load_records(directory, split, *, training=False):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    entry = manifest["files"][split]
    if training and entry["role"] != "train":
        raise ValueError(f"{split} is {entry['role']}; it cannot be used for training")
    path = directory / entry["file"]
    if entry.get("sha256") and file_hash(path) != entry["sha256"]:
        raise ValueError(f"data hash mismatch: {path}")
    result = [Record.from_dict(r) for r in json_rows(path)]
    if len(result) != entry["records"]:
        raise ValueError(f"record count mismatch: {path}")
    return result, manifest


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
