#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Download pinned JevBench and preserve its public tasks as external evaluation.

Preserves every public question and its original labels. Run --verify to check
an existing conversion against the downloaded questions.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import urllib.request

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from qev.data import json_rows, load_records, write_json
from qev.schema import typed_record

UPSTREAM = "https://github.com/fstandhartinger/jevbench"
VERSION = "v1.4.2"
REVISION = "1df665e3956d7aab7fa0208ff6c4f2d8557f9f90"
ARCHIVE_URL = f"https://codeload.github.com/fstandhartinger/jevbench/tar.gz/{REVISION}"
SUBSETS = {"original": 72, "easy": 48, "hard": 111}
DEFAULT_ROOT = REPO / "data" / "jevbench"
DATA_NAME = "jevbench-public-v1.4.2"
DEFAULT_COMPARE = []


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_rows(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def fetch_source(root, offline=False, source_dir=None):
    source = Path(source_dir) if source_dir is not None else root / "source"
    if source.exists():
        return source
    require(not offline, f"Missing JevBench source: {source}")
    source.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".jevbench-", dir=source.parent) as temporary:
        stage = Path(temporary)
        archive = stage / "source.tar.gz"
        with urllib.request.urlopen(ARCHIVE_URL, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        prefix = f"jevbench-{REVISION}"
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                parts = Path(member.name).parts
                require(parts and parts[0] == prefix and ".." not in parts
                        and not Path(member.name).is_absolute()
                        and (member.isfile() or member.isdir()), "Unexpected archive member")
            tar.extractall(stage, filter="data")
        (stage / prefix).rename(source)
    return source


def convert_task(task, subset):
    require(task["split"] == "public", "Only public tasks may be imported")
    require(task.get("expected") is not None and not task["provenance"].get("exclude_reason"),
            "Unscored upstream task needs explicit handling")
    question = task["question"]
    typ = question["type"]
    labels = task["labels"]
    mapping = {k: k for k in labels}
    if typ == "noul":
        require(labels == ["no", "yes"], "Unexpected Noul label convention")
        mapping = {"no": "false", "yes": "true"}
    elif typ == "choice":
        require(set(labels) == set(question["criteria"]), "Choice labels/criteria differ")
    elif typ == "score":
        require(labels == [str(i) for i in range(len(question["criteria"]))], "Invalid Score levels")
    expected = str(task["expected"])
    require(expected in mapping, "Unknown expected label")
    gold = {"label": mapping[expected]}
    if "gold_probs" in task["provenance"]:
        probs = task["provenance"]["gold_probs"]
        require(set(probs) == set(labels), "Gold probability keys differ from labels")
        gold["probabilities"] = {mapping[k]: v for k, v in probs.items()}
    record = typed_record(
        {"state": task["state"], "questions": {"decision": question}},
        source=f"jevbench/{subset}/{task['family']}", record_id=f"jevbench:{task['id']}",
        group_id=f"jevbench:{task.get('group') or task['id']}", gold={"decision": gold},
    )
    if "probabilities" in gold:
        require(all(p == gold["probabilities"][c.id] for c, p in
                    zip(record.questions[0].candidates, record.questions[0].target)),
                "Gold probabilities would change under normalization")
    metadata = {"record_id": record.id, "upstream_id": task["id"], "subset": subset,
                "family": task["family"], "upstream_group": task.get("group"),
                "upstream_labels": labels, "upstream_expected": task["expected"],
                "label_mapping": mapping, "provenance": task["provenance"],
                "target_kind": "gold_probs" if "probabilities" in gold else "one_hot_expected"}
    return record, metadata


def read_source(source):
    upstream_manifest = json.loads((source / "datasets/manifest.json").read_text())
    declared = {x["name"]: x for x in upstream_manifest["splits"]}
    records, metadata, checks = {}, [], {}
    for subset, count in SUBSETS.items():
        path = source / f"datasets/public/{subset}.jsonl"
        tasks = list(json_rows(path))
        check = declared[subset]
        require(len(tasks) == count == check["n"], f"Upstream count mismatch: {subset}")
        records[subset] = []
        for line, task in enumerate(tasks, 1):
            record, meta = convert_task(task, subset)
            records[subset].append(record)
            meta.update(source_file=str(path.relative_to(source)), source_line=line)
            metadata.append(meta)
        checks[subset] = {"file": str(path.relative_to(source)), "records": count}
    all_records = [r for subset in SUBSETS for r in records[subset]]
    require(len({r.id for r in all_records}) == 231, "Duplicate or missing public task IDs")
    records["public"] = all_records  # A union view, not an additional split.
    return records, metadata, checks


def input_keys(record):
    keys = []
    for q in record.questions:
        keys.append(fingerprint([record.state, q.type, q.instructions,
                                 sorted((c.id, c.text) for c in q.candidates)]))
    return keys


def overlap_audit(records, directories):
    inputs, states, groups = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in records:
        for key in input_keys(r):
            inputs[key].append(r.id)
        states[" ".join(r.state.casefold().split())].append(r.id)
        groups[r.group_id].append(r.id)
    audits = []
    for directory in directories:
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for split, entry in manifest["files"].items():
            path = directory / entry["file"]
            item = {"path": str(path), "role": entry["role"], "split": split}
            others, _ = load_records(directory, split)
            exact, state_matches, group_matches = [], [], []
            for other in others:
                hits = sorted({rid for key in input_keys(other) for rid in inputs.get(key, [])})
                if hits:
                    exact.append({"other_id": other.id, "jevbench_ids": hits})
                hits = states.get(" ".join(other.state.casefold().split()), [])
                if hits:
                    state_matches.append({"other_id": other.id, "jevbench_ids": hits})
                if other.group_id in groups:
                    group_matches.append({"other_id": other.id, "jevbench_ids": groups[other.group_id]})
            item.update(records=len(others), questions=sum(len(r.questions) for r in others),
                        exact_question_matches=exact, normalized_state_matches=state_matches,
                        group_id_matches=group_matches)
            audits.append(item)
    return {"scope": "Full canonical inputs (candidate-order independent), casefold/whitespace-normalized state, namespaced group ID. Exact audit only; not semantic or pretraining decontamination.",
            "files": audits, "internal_duplicate_inputs": [v for v in inputs.values() if len(v) > 1],
            "paraphrase_groups": [v for v in groups.values() if len(v) > 1]}


def token_audit(records, tokenizer_path, config_path):
    from transformers import AutoTokenizer
    from qev.encoding import Encoder, Limits, ContextOverflow
    config = json.loads(config_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    encoder = Encoder(tokenizer, Limits(**config["limits"]), choice_none_policy="as-provided")
    details = []
    for record in records:
        state = 1 + len(encoder.text(record.state))
        q = record.questions[0]
        question = 2 + len(encoder.text(q.type + " question: " + q.instructions))
        candidates = [len(encoder.candidate_tokens(c.text)) for c in q.candidates]
        row = {"id": record.id, "source": record.source, "state_tokens": state,
               "question_tokens": question, "max_candidate_tokens": max(candidates),
               "max_path_tokens": state + question + max(candidates), "candidates": len(candidates)}
        try:
            encoder(record)
            row.update(admitted=True, reason=None)
        except ContextOverflow as exc:
            row.update(admitted=False, reason=str(exc))
        details.append(row)
    return details, {"tokenizer": str(tokenizer_path),
                     "config": str(config_path),
                     "limits": asdict(encoder.limits), "choice_none_policy": "as-provided",
                     "admitted": sum(x["admitted"] for x in details),
                     "over_limit_retained": sum(not x["admitted"] for x in details),
                     "maxima": {key: max(x[key] for x in details) for key in
                                ("state_tokens", "question_tokens", "max_candidate_tokens", "max_path_tokens", "candidates")}}


def verify(root, source_dir=None):
    source = fetch_source(root, offline=True, source_dir=source_dir)
    expected, expected_meta, _ = read_source(source)
    data = root / "model_data" / DATA_NAME
    manifest = json.loads((data / "manifest.json").read_text())
    require(manifest["revision"] == REVISION, "Wrong release revision")
    require(set(manifest["files"]) == set(expected), "Unexpected release splits")
    for split, records in expected.items():
        got, _ = load_records(data, split)
        require(got == records, f"Conversion differs: {split}")
        require(manifest["files"][split]["role"] == "external_evaluation", "Invalid split role")
        try:
            load_records(data, split, training=True)
        except ValueError as exc:
            require("cannot be used for training" in str(exc), "Unexpected training guard error")
        else:
            raise ValueError("Evaluation data passed the training guard")
    saved_metadata = list(json_rows(data / "provenance.jsonl"))
    # Older exports also carried per-file checksums; only task metadata matters here.
    require(len(saved_metadata) == len(expected_meta), "Provenance count differs")
    for saved, expected_row in zip(saved_metadata, expected_meta):
        require(all(saved.get(key) == value for key, value in expected_row.items()), "Provenance differs")
    for name in ("LICENSE", "THIRD-PARTY.md"):
        require((data / name).read_bytes() == (source / name).read_bytes(), f"Missing or changed {name}")
    return {"status": "verified", "data": str(data), "unique_records": 231,
            "splits": {split: len(rows) for split, rows in expected.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--source-dir", type=Path,
                    help="existing JevBench source directory (downloaded automatically by default)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--tokenizer", type=Path, help="local tokenizer directory, required for a new release")
    ap.add_argument("--config", type=Path, default=REPO / "configs/qev-9b.json")
    ap.add_argument("--compare-data", nargs="*", default=DEFAULT_COMPARE)
    args = ap.parse_args()
    root = args.root.absolute()
    args.source_dir = args.source_dir or root / "source"
    destination = root / "model_data" / DATA_NAME
    if args.verify or destination.exists():
        print(json.dumps(verify(root, args.source_dir), ensure_ascii=False, indent=2))
        return
    if args.tokenizer is None:
        ap.error("--tokenizer is required to prepare a new evaluation dataset")
    source = fetch_source(root, source_dir=args.source_dir)
    records, metadata, checks = read_source(source)
    timestamp = datetime.now(timezone.utc).isoformat()
    details, token_summary = token_audit(records["public"], args.tokenizer, args.config)
    overlap = overlap_audit(records["public"], args.compare_data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{DATA_NAME}-", dir=destination.parent))
    try:
        files = {}
        for split, rows in records.items():
            path = stage / f"{split}.jsonl"
            write_rows(path, (r.to_dict() for r in rows))
            files[split] = {"file": path.name, "records": len(rows),
                            "questions": len(rows), "role": "external_evaluation",
                            "questions_by_type": dict(Counter(r.questions[0].type for r in rows)),
                            "by_source": dict(Counter(r.source for r in rows)),
                            "soft_target_questions": sum(any(0 < p < 1 for p in r.questions[0].target) for r in rows)}
        write_rows(stage / "provenance.jsonl", metadata)
        write_rows(stage / "token_admission.jsonl", details)
        write_json(stage / "overlap_audit.json", overlap)
        write_json(stage / "validation.json", {"schema_records": 231, "unique_groups": len({r.group_id for r in records['public']}),
                   "upstream_checks": checks, "token_admission": token_summary,
                   "choice_order": "native TypeSafe adapter criteria insertion order; no sorting or shuffling",
                   "choice_labels_order_differs": sum(m["upstream_labels"] != [c.id for c in r.questions[0].candidates]
                                                        for m, r in zip(metadata, records["public"])
                                                        if r.questions[0].type == "choice"),
                   "semantic_review": "Upstream golds retained; no independent relabelling or model review"})
        for name in ("LICENSE", "THIRD-PARTY.md"):
            shutil.copyfile(source / name, stage / name)
        manifest = {"schema": "qev.data.v1", "recipe": "jevbench-public-native-v1",
                    "repo": UPSTREAM, "version": VERSION, "revision": REVISION, "created_at": timestamp,
                    "license": "MIT", "source_task_protocol": "jevbench::v1.2", "leaderboard_protocol": "jevbench::v1.4",
                    "unique_records": 231,
                    "views": "public = original + easy + hard, in that order; do not sum the four views",
                    "unavailable": {"v1_2_nonpublic": 303, "v1_4_additional_sealed": 308},
                    "policy": "Evaluation only. Preserve all items, golds and paraphrase groups, including context overflows. No training split.",
                    "target_policy": "10 gold_probs distributions preserved; remaining 221 targets one-hot expected. Original labels retained via explicit mapping.",
                    "scoring_note": "Local metrics are not the official composite. Re-score with pinned upstream code, inverse-map Noul, use upstream lexical tie-breaking. Local Brier/NLL use target distributions, which differs from official categorical calibration.",
                    "files": files}
        write_json(stage / "manifest.json", manifest)
        for split in files:
            load_records(stage, split)
        stage.rename(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    result = verify(root, args.source_dir)
    result["token_admission"] = token_summary
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
