"""Read-only adapters for actual SAL SQLite caches and portable JSONL fixtures.

Purpose: explicit score/grid/provenance validation and nested source-group splits.
Inputs: SAL run directory (config.json, predictions.sqlite), source-map JSONL;
  or records JSONL. Outputs: Record objects and reproducible split manifests.
Assumptions: saved raw float32 Nx2 logits, uint8 labels, 0=spoof as pipeline.py.
Usage: load_records(path); make_splits(records,seed,validation_groups).
Limitations: source identities cannot be inferred reliably from uid strings;
caller must supply verified source groups. Labels reflect saved evaluation grid,
not higher-resolution original annotations. SQLite is never modified.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
from .core import Record, runs


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load_records(path: Path, source_map: Path | None = None, semantic_path: Path | None = None) -> list[Record]:
    words = {}
    if semantic_path:
        for row in read_jsonl(semantic_path):
            if row['uid'] in words:
                raise ValueError('Duplicate semantic uid')
            # D intentionally accepts only transcript lexical categories, not fitted
            # test-error tables or full-train priors that can see modifier dev labels.
            if set(row)-{'uid', 'words', 'language', 'audio', 'asr_model', 'asr_seconds', 'discarded_zero_duration_words', 'annotation', 'annotation_method', 'ner_model'}:
                raise ValueError('Use annotated transcripts, not outcome/prior tables')
            words[row['uid']] = row['words']
    if path.is_dir():
        meta = json.loads((path/'config.json').read_text(encoding='utf-8-sig'))
        if (path/'status.json').exists() and json.loads((path/'status.json').read_text())['status'] != 'complete':
            raise ValueError('Refuse incomplete SAL cache')
        dt = float(meta['resolution_seconds'])
        expected = .02 if meta['split'] == 'lps' else .16
        if abs(dt-expected) > 1e-9:
            raise ValueError('Grid differs from actual SAL PS/LPS implementation')
        if not source_map:
            raise ValueError('SQLite import requires verified uid/source_group/split map')
        entries = read_jsonl(source_map)
        mapping = {row['uid']: row for row in entries}
        if len(mapping) != len(entries):
            raise ValueError('Duplicate source-map uid')
        records = []
        with sqlite3.connect((path/'predictions.sqlite').resolve().as_uri()+'?mode=ro', uri=True) as db:
            for uid, blob, labels, seconds in db.execute('SELECT uid,logits,labels,seconds FROM predictions ORDER BY ordinal'):
                entry = mapping[uid]
                expected_split = {'train': 'train', 'dev': 'dev', 'eval': 'test', 'lps': 'test'}[meta['split']]
                if entry['split'] != expected_split:
                    raise ValueError('Cache/source map split mismatch')
                logits = np.frombuffer(blob, dtype=np.float32).reshape(-1, 2).astype(float)
                label = np.frombuffer(labels, dtype=np.uint8)
                if len(logits) != len(label) or not np.isin(label, [0, 1]).all() or not np.isfinite(logits).all():
                    raise ValueError('Invalid SAL cache')
                prob = 1/(1+np.exp(np.clip(logits[:, 1]-logits[:, 0], -700, 700)))
                horizon = min(float(seconds), len(label)*dt)
                records.append(Record(uid, prob, dt, float(seconds), runs(logits.argmax(1) == 0, dt, horizon),
                                      entry['source_group'], entry['split'], runs(label == 0, dt, horizon), words.get(uid)))
    else:
        records = []
        for row in read_jsonl(path):
            allowed = {'uid', 'scores', 'dt', 'seconds', 'original', 'source_group', 'split', 'truth'}
            if set(row)-allowed:
                raise ValueError('Unexpected record fields')
            row['original'] = tuple(map(tuple, row['original']))
            if row.get('truth') is not None:
                row['truth'] = tuple(map(tuple, row['truth']))
            records.append(Record(**row, words=words.get(row['uid'])))
    if not records or len({r.uid for r in records}) != len(records):
        raise ValueError('Empty dataset or duplicate uid')
    return records


def make_splits(records: list[Record], seed: int = 888, validation_groups: int = 100, sizes: tuple[int, ...] = (2000, 5000, 20000)) -> dict:
    """Keep official dev/test; nested train prefixes keep source groups intact.

    If official dev is absent, reserve groups from train before sampling. Requested
    sizes are recording budgets: group-preserving prefixes may undershoot them.
    Insufficient data is reported explicitly; never relabel train as official test.
    """
    groups: dict[str, list[Record]] = {}
    for r in records:
        groups.setdefault(r.source_group, []).append(r)
    if any(len({r.split for r in group}) != 1 for group in groups.values()):
        raise ValueError('Source-group leakage across official splits')
    train = sorted(g for g, rows in groups.items() if rows[0].split == 'train')
    dev = sorted(g for g, rows in groups.items() if rows[0].split == 'dev')
    test = sorted(g for g, rows in groups.items() if rows[0].split == 'test')
    np.random.default_rng(seed).shuffle(train)
    if not dev:
        if validation_groups <= 0 or validation_groups >= len(train):
            raise ValueError('Need independent validation groups')
        dev, train = train[:validation_groups], train[validation_groups:]
    def ids(names: list[str]) -> list[str]:
        return [r.uid for g in names for r in sorted(groups[g], key=lambda x: x.uid)]
    subsets = {}
    for size in sizes:
        chosen: list[str] = []
        for group in train:
            if len(chosen)+len(groups[group]) > size:
                break
            chosen.extend(ids([group]))
        subsets[str(size)] = dict(ids=chosen, requested=size, actual=len(chosen), complete=len(chosen) == size)
    return dict(seed=seed, train_subsets=subsets, validation_ids=ids(dev), official_test_ids=ids(test),
                source_groups={r.uid:r.source_group for r in records},
                original_splits={r.uid:r.split for r in records},
                official_test_available=bool(test), future_recheck_train_ids=[],
                future_recheck_note='Reserve independent source groups before adding any reinspection training.')


def validate_partitions(manifest: dict) -> None:
    dev, test = set(manifest['validation_ids']), set(manifest['official_test_ids'])
    mapping = manifest['source_groups']
    def group_set(ids: set[str]) -> set[str]:
        return {mapping[uid] for uid in ids}
    if dev & test or group_set(dev) & group_set(test):
        raise ValueError('Validation/test overlap')
    previous: set[str] = set()
    for size, subset in sorted(manifest['train_subsets'].items(), key=lambda x:int(x[0])):
        train = set(subset['ids'])
        if len(train) != len(subset['ids']) or not previous <= train or train & (dev | test) or group_set(train) & group_set(dev | test):
            raise ValueError('Invalid nested source-group split')
        if any(manifest['original_splits'][uid] != 'train' for uid in train):
            raise ValueError('Training subset contains non-training recording')
        previous = train
