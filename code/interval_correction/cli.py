"""Reproducible cached-output A/B/C/D CLI; never runs SAL or full audio training.

Inputs: repeated --input SAL run directories / JSONL, verified source map,
partition manifest and optional annotated transcripts/checkpoints.
Outputs: fresh run directories, config/provenance, metrics/predictions/traces.
Usage: python -m interval_correction.cli --help (from code directory).
Assumptions: source grouping is externally verified; thresholds chosen on dev.
Limitations: CPU first implementation; real cache/ASR validation needs local data.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import csv
import numpy as np
from .core import Config, Record, generate, rule_utilities, select
from .data import load_records, make_splits, read_jsonl, sha256, validate_partitions
from .metrics import aggregate, diagnostics, evaluate, gain, risk


def dump(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def fixture(output: Path) -> None:
    """Synthetic fixture only; no research performance claims."""
    output.mkdir(parents=True, exist_ok=False)
    rows, transcripts = [], []
    for i in range(30):
        scores = np.array([.05]*5+[.85]*4+[.1]*2+[.8]*4+[.05]*5+[.45]*3+[.05]*7)
        if i % 3 == 0:
            original = ((.5, 1.5),)
        elif i % 3 == 1:
            original = ((.5, .9), (1.1, 1.5))
        else:
            original = ()
        split = 'train' if i < 20 else 'dev' if i < 26 else 'test'
        uid = f'synthetic_{i:03d}'
        rows.append(dict(uid=uid, scores=scores.tolist(), dt=.1, seconds=3., original=original,
                         truth=((.5, .9), (1.1, 1.5), (2., 2.3)), source_group=f'source_{i:03d}', split=split))
        transcripts.append(dict(uid=uid, language='en', words=[dict(text='not', start=.5, end=.8, confidence=.9, primary_type='negation')]))
    for name, items in [('records.jsonl', rows), ('transcripts.jsonl', transcripts)]:
        (output/name).write_text(''.join(json.dumps(row)+'\n' for row in items), encoding='utf8')
    dump(output/'NOTICE.json', dict(synthetic=True, research_result=False))


def load_all(args: argparse.Namespace) -> list[Record]:
    records = [r for path in args.input for r in load_records(path, args.source_map, args.semantic)]
    if len({r.uid for r in records}) != len(records):
        raise ValueError('Duplicate uid across inputs: namespace datasets explicitly')
    if not records:
        raise ValueError('Supply --input')
    return records


def partition(records: list[Record], manifest: dict, size: int, allow_smaller: bool) -> tuple[list[Record], list[Record], list[Record]]:
    validate_partitions(manifest)
    by_id = {r.uid:r for r in records}
    subset = manifest['train_subsets'][str(size)]
    if subset['actual'] != size and not allow_smaller:
        raise ValueError('Insufficient/group-rounded training size; explicitly pass --allow-smaller for smoke tests')
    def resolve(ids: list[str]) -> list[Record]:
        out = [by_id[uid] for uid in ids]
        if any(r.source_group != manifest['source_groups'][r.uid] or r.split != manifest['original_splits'][r.uid] for r in out):
            raise ValueError('Manifest source/split differs from input')
        return out
    return resolve(subset['ids']), resolve(manifest['validation_ids']), resolve(manifest['official_test_ids'])


def run_evaluation(records: list[Record], cfg: Config, modes: list[str], output: Path,
                   checkpoints: dict[str, Path | None], split: str, offline: bool,
                   manifest: dict, semantic_path: Path | None) -> list[dict]:
    from .model import Predictor
    if not records or any(r.truth is None for r in records):
        raise ValueError('Evaluation requires nonempty labeled partition')
    if 'D' in modes and semantic_path is None:
        raise ValueError('D needs annotated transcripts, not missing semantic inputs')
    predictors = {}
    for mode in modes:
        if mode in ('C', 'D'):
            if checkpoints[mode] is None:
                raise ValueError(f'{mode} requires checkpoint')
            predictor = Predictor(checkpoints[mode], cfg, mode == 'D')
            if set(predictor.meta['train_groups']) & {r.source_group for r in records}:
                raise ValueError('Evaluating training groups would be in-sample')
            if split == 'test' and set(predictor.meta['validation_groups']) & {r.source_group for r in records}:
                raise ValueError('Official test overlaps model selection')
            predictors[mode] = predictor
    output.mkdir(parents=True, exist_ok=False)
    totals = {m:[] for m in modes}
    times = Counter()
    operations = {m:Counter() for m in modes}
    harmed = Counter()
    edits = Counter()
    harmed_recordings = Counter()
    modified_recordings = Counter()
    predictions, failures, diagnostic_rows = [], [], []
    generation_time = 0.
    for r in records:
        started = time.perf_counter()
        pool = generate(r, cfg)
        generation_time += time.perf_counter()-started
        if offline:
            diagnostic_rows.append(dict(uid=r.uid, **diagnostics(r, pool, cfg)))
        for mode in modes:
            started = time.perf_counter()
            selected = []
            if mode == 'A':
                prediction = r.original
            else:
                utility = rule_utilities(r, pool, cfg) if mode == 'B' else predictors[mode].predict(r, pool)
                prediction, selected = select(r, pool, utility, 0. if mode == 'B' else cfg.minimum_gain)
            times[mode] += time.perf_counter()-started
            m = evaluate(prediction, r.truth, cfg)
            totals[mode].append(m)
            edit_harm = sum(gain(r, c, cfg) < -1e-8 for c in selected)
            changed = bool(selected)
            worse = risk(prediction, r.truth, cfg) > risk(r.original, r.truth, cfg)+1e-8
            harmed[mode] += edit_harm
            edits[mode] += len(selected)
            modified_recordings[mode] += int(changed)
            harmed_recordings[mode] += int(changed and worse)
            operations[mode].update(c.operation for c in selected)
            predictions.append(dict(uid=r.uid, mode=mode, split=split, original=r.original, prediction=prediction,
                                    selected_operations=[c.to_dict() for c in selected], metrics=m))
            if worse:
                failures.append(dict(uid=r.uid, mode=mode, before=r.original, after=prediction, truth=r.truth,
                                     risk_before=risk(r.original, r.truth, cfg), risk_after=risk(prediction, r.truth, cfg)))
    rows = []
    for mode in modes:
        rows.append(dict(mode=mode, partition=split, metric_family='research_event_tiou_v1',
                         **aggregate(totals[mode]), harmful_modification_rate=harmed[mode]/edits[mode] if edits[mode] else None,
                         harmful_recording_rate=harmed_recordings[mode]/modified_recordings[mode] if modified_recordings[mode] else None,
                         selected_edits=edits[mode], harmful_edits=harmed[mode],
                         runtime_seconds=times[mode]+(generation_time if mode != 'A' else 0.),
                         candidate_generation_seconds=generation_time if mode != 'A' else 0.,
                         candidate_coverage=(sum(d['candidate_covered_events'] for d in diagnostic_rows)/sum(d['candidate_truth_events'] for d in diagnostic_rows)) if offline and sum(d['candidate_truth_events'] for d in diagnostic_rows) else None))
    with (output/'metrics.csv').open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/'predictions.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in predictions), encoding='utf8')
    dump(output/'failure_cases.json', failures[:100])
    dump(output/'operations.json', operations)
    if offline:
        dump(output/'OFFLINE_DIAGNOSTICS_NOT_PREDICTIONS.json', diagnostic_rows)
    dump(output/'config.json', dict(config=asdict(cfg), partition=split, seed=manifest['seed'], modes=modes,
                                   numpy_version=np.__version__, runtime_excludes='SAL/ASR/IO/model loading/offline oracle',
                                   harmful_modification_definition='fraction of selected individual edits with negative fixed-baseline risk gain',
                                   harmful_recording_definition='fraction of modified recordings with increased combined risk',
                                   threshold_status='illustrative defaults unless independently selected on development data'))
    dump(output/'partitions.json', manifest)
    dump(output/'run_log.json', dict(status='complete', recordings=len(records), metrics_rows=len(rows), offline_diagnostics=offline))
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('fixture', 'check', 'split', 'train', 'evaluate'))
    parser.add_argument('--input', type=Path, action='append', default=[])
    parser.add_argument('--source-map', type=Path)
    parser.add_argument('--semantic', type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=2000)
    parser.add_argument('--allow-smaller', action='store_true')
    parser.add_argument('--validation-groups', type=int, default=100)
    parser.add_argument('--seed', type=int, default=888)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--modes', nargs='+', choices=('A', 'B', 'C', 'D'), default=['A', 'B'])
    parser.add_argument('--split', choices=('dev', 'test'), default='dev')
    parser.add_argument('--c-checkpoint', type=Path)
    parser.add_argument('--d-checkpoint', type=Path)
    parser.add_argument('--offline-diagnostics', action='store_true', help='GT-only oracle diagnostics; never an operational system')
    args = parser.parse_args(argv)
    if args.action == 'fixture':
        fixture(args.output)
        return
    cfg = Config(**json.loads(args.config.read_text()))
    records = load_all(args)
    if args.action == 'check':
        for r in records:
            generate(r, cfg)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf8') as stream:
            json.dump(dict(recordings=len(records), splits=dict(Counter(r.split for r in records)),
                           grids=sorted({r.dt for r in records}), scored_seconds=sum(r.horizon for r in records),
                           unscored_tail_seconds=sum(r.seconds-r.horizon for r in records),
                           source_identity='externally supplied; not verified from audio'), stream, indent=2)
        return
    if args.action == 'split':
        manifest = make_splits(records, args.seed, args.validation_groups)
        validate_partitions(manifest)
        with args.output.open('x', encoding='utf8') as stream:
            json.dump(manifest, stream, indent=2)
        return
    if not args.manifest:
        parser.error('--manifest required for train/evaluate')
    manifest = json.loads(args.manifest.read_text())
    training, dev, test = partition(records, manifest, args.size, args.allow_smaller)
    if args.action == 'train':
        from .model import train
        if len(args.modes) != 1 or args.modes[0] not in ('C', 'D'):
            parser.error('Train exactly one of --modes C / D')
        if args.seed != manifest['seed']:
            raise ValueError('Seed differs from manifest')
        if args.modes[0] == 'D' and args.semantic is None:
            raise ValueError('D requires --semantic')
        result = train(training, dev, cfg, args.output, args.modes[0] == 'D', args.seed,
                       args.epochs, args.patience, args.batch_size, args.learning_rate)
        dump(args.output/'partitions.json', manifest)
    else:
        result = run_evaluation(dev if args.split == 'dev' else test, cfg, list(dict.fromkeys(args.modes)), args.output,
                                {'C': args.c_checkpoint, 'D': args.d_checkpoint}, args.split,
                                args.offline_diagnostics, manifest, args.semantic)
    provenance = dict(inputs=[dict(path=str(p), sha256=sha256(p/'predictions.sqlite' if p.is_dir() else p)) for p in args.input],
                      config_sha256=sha256(args.config), manifest_sha256=sha256(args.manifest),
                      semantic_sha256=sha256(args.semantic) if args.semantic else None,
                      checkpoints={k:sha256(p) for k, p in {'C':args.c_checkpoint, 'D':args.d_checkpoint}.items() if p},
                      command=sys.argv if argv is None else argv)
    dump(args.output/'provenance.json', provenance)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
