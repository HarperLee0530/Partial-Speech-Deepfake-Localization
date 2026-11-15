"""Replay the fixed PS comparison from explicit frozen-CNN caches.

Calibrate all reporting thresholds on dev, write them to disk, THEN open eval.
The selected head checkpoint is never changed by this reporting calibration.
160ms scores average all eight stored native logits, including the original
storage tail. Native20 duration/event reports, when requested, remove that tail.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import re
import time

import numpy as np
import torch

from .cached_data import CachedData
from .head import load_head
from .io import read_json, save_json, sha256
from . import metrics160, metrics20
from .train_head import predict_native_scores


FIELDS160 = ('spoof_f1_percent', 'eer_percent', 'event_f1_iou05_percent',
             'short_event_recall_le320ms_percent', 'real_fpr_percent')
FIELDS20 = ('duration_f1_percent', 'event_f1_iou05_percent',
            'short_event_recall_le320ms_percent', 'real_fpr_percent')


def cached_rows(data, flat_scores=None):
    """Yield truth, all stored native cells, and corresponding frozen context."""
    flat = data.native.numpy().reshape(-1) if flat_scores is None else np.asarray(flat_scores).reshape(-1)
    if len(flat) != len(data)*8:
        raise ValueError('Scores do not cover every stored native cell')
    for i, uid in enumerate(data.uids):
        a, b = int(data.offsets[i]), int(data.offsets[i+1])
        nf = int(data.lengths20[i])
        yield dict(uid=uid, labels=data.labels[a:b].numpy(),
                   labels20=data.fine_labels[a:b].numpy().reshape(-1)[:nf],
                   native=flat[a*8:b*8].reshape(-1, 8),
                   context=data.context[a:b].numpy(), duration=float(data.durations[i]))


def npz_rows(manifest):
    """Read explicit per-record NPZ paths, resolved relative to their manifest."""
    manifest = Path(manifest).resolve()
    payload = read_json(manifest)
    if isinstance(payload, dict):
        if payload.get('split') != 'eval':
            raise ValueError('Evaluation manifest must declare split=eval')
        records = payload.get('records')
    else:
        raise ValueError('Evaluation manifest must contain split=eval and records')
    if not isinstance(records, list) or not records:
        raise ValueError('Empty evaluation manifest')
    seen = set()
    for entry in records:
        uid = entry['uid']
        if uid in seen or entry.get('split', 'eval') != 'eval':
            raise ValueError('Duplicate ID or non-eval record')
        marker = re.search(r'_(T|D|E)_', uid)
        if marker and marker.group(1) != 'E':
            raise ValueError(f'Not an official eval ID: {uid}')
        seen.add(uid)
        path = Path(entry['path']).expanduser()
        path = path if path.is_absolute() else manifest.parent/path
        with np.load(path, allow_pickle=False) as stored:
            if 'uid' in stored and str(stored['uid'].item()) != uid:
                raise ValueError(f'Cache UID differs from manifest: {uid}')
            labels = np.asarray(stored['labels'], dtype=np.float32).reshape(-1)
            labels20 = np.asarray(stored['labels20'], dtype=np.float32).reshape(-1)
            native = np.asarray(stored['native_logits'], dtype=np.float32).reshape(-1)
            context = np.asarray(stored['context'], dtype=np.float16)
            duration = float(stored['duration'])
        if (not len(labels) or len(native) != 8*len(labels) or context.shape != (len(native), 256)
                or not 0 < len(labels20) <= len(native)):
            raise ValueError(f'Invalid cache shapes: {uid}')
        if not np.isin(labels, (0, 1)).all() or not np.isin(labels20, (0, 1)).all():
            raise ValueError(f'Labels must use 1=fake, 0=real: {uid}')
        if not np.isfinite(native).all() or not np.isfinite(context).all():
            raise ValueError(f'Nonfinite cache: {uid}')
        if not np.isfinite(duration) or duration <= 0 or len(labels20)*.02 < duration-1e-9:
            raise ValueError(f'Actual20 labels do not cover duration: {uid}')
        yield dict(uid=uid, labels=labels, labels20=labels20,
                   native=native.reshape(-1, 8), context=context.reshape(-1, 8, 256), duration=duration)


def report_row(row, fine, native20=False):
    fine = np.asarray(fine, dtype=np.float32).reshape(-1)
    if native20:
        return dict(uid=row['uid'], labels20=row['labels20'],
                    scores20=fine[:len(row['labels20'])], duration=row['duration'])
    # Historical comparison uses NumPy float32 mean, not sigmoid averaging or
    # repeating the final real cell over the stored padding cells.
    return dict(uid=row['uid'], labels=row['labels'], labels20=row['labels20'],
                scores=fine.reshape(-1, 8).mean(1), duration=row['duration'])


def dev_threshold(data, fine):
    labels = data.labels.numpy()
    scores = np.asarray(fine, dtype=np.float32).reshape(-1, 8).mean(1)
    threshold, _ = metrics160.choose_threshold(labels, scores)
    return threshold


def model_name(stored, path):
    arm, seed = stored.get('arm'), stored.get('seed')
    name = f'{arm}_{seed}' if arm is not None and seed is not None else Path(path).stem
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
        raise ValueError('Checkpoint arm/name must use letters, digits, underscore, dot or dash')
    return name


def summarize_groups(summary, models, fields):
    grouped = defaultdict(list)
    for name, description in models.items():
        grouped[description.get('arm') or name].append(name)
    result = {}
    for arm, names in grouped.items():
        result[arm] = {key: dict(mean=float(np.mean([summary[name][key] for name in names])),
                                 sd=float(np.std([summary[name][key] for name in names], ddof=1))
                                 if len(names) > 1 else None)
                       for key in fields}
    return result


@torch.inference_mode()
def evaluate(dev_cache, output, heads=(), eval_cache=None, eval_records=None, native20=False, threads=4):
    if (eval_cache is None) == (eval_records is None):
        raise ValueError('Supply exactly one of eval_cache or eval_records')
    if threads <= 0:
        raise ValueError('threads must be positive')
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a new or empty evaluation directory: {output}')
    torch.set_num_threads(threads)
    dev = CachedData(dev_cache)
    if dev.role != 'dev':
        raise ValueError('Threshold selection requires a cache marked split=dev')
    baseline_scores = dev.native.numpy().reshape(-1)
    thresholds160 = dict(cnn_original_threshold0=0., cnn_dev_f1_calibrated=dev_threshold(dev, baseline_scores))
    thresholds20 = {}
    if native20:
        thresholds20 = dict(cnn_original_threshold0=0., cnn_dev_fpr_calibrated=
                            metrics20.choose_threshold(dev.prediction_records(baseline_scores), .05)['threshold'])
    models, descriptions = {}, {}
    for path in heads:
        path = Path(path).resolve()
        stored = torch.load(path, map_location='cpu', weights_only=False)
        name = model_name(stored, path)
        if name in models or name in thresholds160:
            raise ValueError(f'Duplicate model name: {name}')
        model = load_head(path)
        fine = predict_native_scores(model, dev)
        thresholds160[name] = dev_threshold(dev, fine)
        if native20:
            if 'threshold' not in stored or not np.isfinite(stored['threshold']):
                raise ValueError(f'Native20 reporting needs stored dev-selected threshold: {name}')
            thresholds20[name] = float(stored['threshold'])
        models[name] = model
        descriptions[name] = dict(checkpoint=str(path), checkpoint_sha256=sha256(path),
                                  arm=stored.get('arm'), seed=stored.get('seed'),
                                  selected_epoch=stored.get('epoch'))
    # Deliberately no eval file is opened before this immutable calibration.
    output.mkdir(parents=True, exist_ok=True)
    calibration = dict(models=descriptions, thresholds=thresholds160, thresholds20=thresholds20,
                       dev_recordings=len(dev.uids), dev_metadata_sha256=sha256(dev.folder/'complete.json'),
                       scope='Frozen head checkpoint; mean8 ALL storage logits; dev-only 160ms F1 threshold',
                       native20_scope='Original threshold0 plus dev5% real-duration FPR; heads keep stored native20 dev threshold',
                       checkpoint_reselection=False, eval_read=False, source_sha256=sha256(__file__))
    save_json(output/'frozen_dev_calibration.json', calibration)
    if eval_cache is not None:
        test = CachedData(eval_cache)
        if test.role != 'eval':
            raise ValueError('Evaluation cache must declare split=eval')
        rows = cached_rows(test)
    else:
        rows = npz_rows(eval_records)
    predictions160 = {name: [] for name in thresholds160}
    predictions20 = {name: [] for name in thresholds20}
    dev_ids = set(dev.uids)
    seen = set()
    started = last = time.monotonic()
    for row in rows:
        uid = row['uid']
        if uid in seen or uid in dev_ids:
            raise ValueError('Duplicate eval ID or dev/eval ID overlap')
        seen.add(uid)
        baseline = report_row(row, row['native'])
        predictions160['cnn_original_threshold0'].append(baseline)
        predictions160['cnn_dev_f1_calibrated'].append(baseline)
        if native20:
            baseline20 = report_row(row, row['native'], native20=True)
            predictions20['cnn_original_threshold0'].append(baseline20)
            predictions20['cnn_dev_fpr_calibrated'].append(baseline20)
        native = torch.from_numpy(np.asarray(row['native']))
        context = torch.from_numpy(np.asarray(row['context']))
        for name, model in models.items():
            fine = model(native, context, return_native=True)[2].cpu().numpy()
            predictions160[name].append(report_row(row, fine))
            if native20:
                predictions20[name].append(report_row(row, fine, native20=True))
        if time.monotonic()-last > 20:
            print(f'Evaluated frozen scores for {len(seen)} recordings', flush=True)
            last = time.monotonic()
    if not seen:
        raise ValueError('Empty evaluation split')
    summary = {}
    for name, records in predictions160.items():
        result = metrics160.evaluate(records, thresholds160[name])
        save_json(output/'metrics160'/f'{name}_metrics.json', result)
        summary[name] = {key: value for key, value in result.items() if key != 'per_recording'}
    save_json(output/'summary.json', summary)
    save_json(output/'mean_sd.json', summarize_groups(summary, descriptions, FIELDS160))
    if native20:
        summary20 = {}
        for name, records in predictions20.items():
            result = metrics20.evaluate(records, thresholds20[name])
            save_json(output/'metrics20'/f'{name}_metrics.json', result)
            summary20[name] = {key: value for key, value in result.items() if key != 'per_recording'}
        save_json(output/'summary20.json', summary20)
        save_json(output/'mean_sd20.json', summarize_groups(summary20, descriptions, FIELDS20))
    for description in descriptions.values():
        if sha256(description['checkpoint']) != description['checkpoint_sha256']:
            raise ValueError('Checkpoint changed during reporting')
    complete = dict(status='complete', eval_recordings=len(seen), models_changed=False,
                    head_models=len(models), elapsed_seconds=time.monotonic()-started,
                    native20_reported=bool(native20), calibration_sha256=sha256(output/'frozen_dev_calibration.json'))
    save_json(output/'complete.json', complete)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev-cache', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--eval-cache')
    source.add_argument('--eval-records', help='JSON with split=eval and records containing uid/path to NPZ')
    parser.add_argument('--heads', nargs='*', default=[])
    parser.add_argument('--output', required=True)
    parser.add_argument('--native20', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    evaluate(**vars(args))


if __name__ == '__main__':
    main()
