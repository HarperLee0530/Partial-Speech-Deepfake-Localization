"""Re-evaluate three fixed PS checkpoints at one native-20ms operating protocol.

No training or checkpoint selection is performed. Original/head scores are
replayed from the archived frozen-CNN features; the fine-tuned model uses its
hash-verified archived raw scores. Every threshold is recalculated on dev and
written before eval sources are opened. Historical checkpoint selection is
reported explicitly and is not relabelled as v2 event-F1 selection.
"""
import argparse
import hashlib
import platform
from pathlib import Path
import time

import numpy as np
import scipy
import sklearn
import torch

from . import metrics20
from .cached_data import CachedData
from .evaluate import npz_rows, report_row
from .head import load_head
from .io import read_json, save_json, sha256
from .report_native20 import evaluate_report
from .train_head import predict_native_scores


NAMES = ('original_cnn', 'channel_head_seed666', 'channel_finetuned_seed666')
FILES = ('original_cnn.pt', 'head_aug_erm_seed666.pt', 'cnn_finetuned_seed666.pt')
PROTOCOL = 'ps_native20_event_v2_fixed_checkpoint_reevaluation'


def check_ids(uids, expected, split):
    if len(uids) != len(set(uids)) or set(uids) != set(expected):
        raise ValueError(f'{split}: IDs differ from the fixed PS subset')


def check_labels(records, path):
    with np.load(path, allow_pickle=False) as labels:
        if set(labels.files) != {r['uid'] for r in records}:
            raise ValueError('Actual20 label IDs differ from the score records')
        for row in records:
            if not np.array_equal(row['labels20'], labels[row['uid']]):
                raise ValueError(f"Actual20 labels disagree: {row['uid']}")


def read_saved_scores(path, reference, block_lengths):
    with np.load(path, allow_pickle=False) as stored:
        scores = np.asarray(stored['scores20'], dtype=np.float32).reshape(-1)
        uids = stored['uids'].tolist()
        blocks = np.asarray(stored['lengths'])
        fine_lengths = np.asarray(stored['lengths20'])
        durations = np.asarray(stored['durations'], dtype=np.float64)
    if uids != [r['uid'] for r in reference]:
        raise ValueError('Saved raw-score IDs/order differ from reference')
    if any(a.shape != (len(reference),) for a in (blocks, fine_lengths, durations)):
        raise ValueError('Saved raw-score metadata dimensions disagree')
    if not np.array_equal(blocks, block_lengths) or not np.isfinite(scores).all():
        raise ValueError('Saved score storage lengths/values disagree')
    if len(scores) != int(np.sum(block_lengths))*8:
        raise ValueError('Saved scores do not cover exactly the storage grid')
    rows, offset = [], 0
    for i, ref in enumerate(reference):
        if (fine_lengths[i] != len(ref['labels20'])
                or not np.isfinite(durations[i])
                or abs(durations[i]-ref['duration']) > 1e-9):
            raise ValueError(f"Saved score duration/labels mismatch: {ref['uid']}")
        count = int(block_lengths[i])*8
        rows.append(dict(uid=ref['uid'], duration=ref['duration'], labels20=ref['labels20'],
                         scores20=scores[offset:offset+len(ref['labels20'])]))
        offset += count
    return rows


def save_predictions(path, records):
    """Portable audit format, actual reference lengths; no pickle or masks."""
    lengths = np.asarray([len(r['labels20']) for r in records], dtype=np.int64)
    np.savez_compressed(path, uids=np.asarray([r['uid'] for r in records]),
                        lengths20=lengths, offsets=np.r_[0, np.cumsum(lengths)],
                        durations=np.asarray([r['duration'] for r in records], dtype=np.float64),
                        labels20=np.concatenate([r['labels20'] for r in records]).astype(np.uint8),
                        scores20=np.concatenate([r['scores20'] for r in records]).astype(np.float32))
    return sha256(path)


def summary(report):
    return {key: value for key, value in report.items() if key != 'per_recording'}


@torch.inference_mode()
def compare(args):
    started = time.monotonic()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new or empty output directory')
    if args.threads <= 0:
        raise ValueError('threads must be positive')
    torch.set_num_threads(args.threads)
    weights = Path(args.weights_dir).resolve()
    run, old_eval = Path(args.finetuned_run).resolve(), Path(args.finetuned_evaluation).resolve()
    fixed = {split:[entry['uid'] if isinstance(entry, dict) else entry for entry in entries]
             for split,entries in read_json(args.fixed_ids).items()}
    if any(set(fixed[a]) & set(fixed[b]) for a,b in [('train','dev'),('train','eval'),('dev','eval')]):
        raise ValueError('Fixed official splits overlap')
    manifest = {r['file']: r for r in read_json(weights/'manifest.json')['weights']}
    descriptions = {}
    for name, filename in zip(NAMES, FILES):
        digest = sha256(weights/filename)
        if digest != manifest[filename]['sha256']:
            raise ValueError(f'Published checkpoint checksum mismatch: {filename}')
        descriptions[name] = dict(file=filename, checkpoint_sha256=digest, seed=666,
                                  selected_epoch=manifest[filename]['selected_epoch'],
                                  checkpoint_reselected=False,
                                  historical_selection=('fixed_historical_baseline; original dev EER'
                                      if name == NAMES[0] else 'dev J=0.5 durationF1+0.5 eventF1'),
                                  new_v2_checkpoint_selection=False)
    trained = read_json(run/'complete.json')
    training_protocol = read_json(run/'protocol.json')
    if (trained['status'] != 'complete'
            or trained['checkpoint_sha256'] != descriptions[NAMES[2]]['checkpoint_sha256']
            or trained['backbone_source_sha256'] != descriptions[NAMES[0]]['checkpoint_sha256']):
        raise ValueError('Archived fine-tuning run does not match published weights')
    if sha256(args.dev_labels20) != training_protocol['dev_labels20_sha256']:
        raise ValueError('Dev labels differ from fine-tuning selection labels')
    fine_payload = torch.load(weights/FILES[2], map_location='cpu', weights_only=False)
    head_payload = torch.load(weights/FILES[1], map_location='cpu', weights_only=False)
    if (head_payload.get('seed') != 666 or fine_payload.get('seed') != 666
            or head_payload.get('arm') != 'aug_erm' or fine_payload.get('arm') != 'cnn_channel_finetune'):
        raise ValueError('Expected the fixed seed666 augmented head and CNN')
    output.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parent
    save_json(output/'protocol.json', dict(protocol_version=PROTOCOL,
        metric_protocol='ps_native20_event_v2', checkpoint_selection='fixed historical weights',
        checkpoint_reselection=False, training_performed=False, eval_previously_inspected=True,
        prediction_grid_seconds=.02, iou_threshold=.5, short_truth_max_seconds=.32,
        dev_real_duration_fpr_limit=.05, postprocessing='merge consecutive positive frames only',
        models=descriptions, inputs={k:str(v) for k,v in vars(args).items()},
        source_sha256={f:sha256(source_root/f) for f in
                       ['compare_native20.py','report_native20.py','metrics20.py','metrics160.py']},
        baseline_training_history='Original CNN trained on full PS train25380/dev24844; adapters used train6345/dev2484.'))
    print('Loading dev cache and replaying the frozen head', flush=True)
    dev = CachedData(args.dev_cache)
    if dev.role != 'dev':
        raise ValueError('Calibration requires split=dev')
    check_ids(dev.uids, fixed['dev'], 'dev')
    head = load_head(weights/FILES[1])
    dev_records = {
        NAMES[0]: dev.prediction_records(dev.native.numpy().reshape(-1)),
        NAMES[1]: dev.prediction_records(predict_native_scores(head, dev))}
    check_labels(dev_records[NAMES[0]], args.dev_labels20)
    dev_records[NAMES[2]] = read_saved_scores(run/'dev_scores20.npz', dev_records[NAMES[0]], dev.lengths)
    calibration = dict(protocol_version=PROTOCOL, dev_recordings=len(dev.uids), eval_read=False,
                       checkpoint_reselection=False, models=descriptions, thresholds={},
                       dev_predictions_sha256={}, dev_metrics={},
                       dev_cache_metadata_sha256=sha256(Path(args.dev_cache)/'complete.json'),
                       dev_cache_array_sha256={f:sha256(Path(args.dev_cache)/f) for f in
                                               ['native.npy','context.npy','fine_labels.npy']},
                       dev_labels20_sha256=sha256(args.dev_labels20),
                       fixed_ids_sha256=sha256(args.fixed_ids),
                       source_dev_scores_sha256=sha256(run/'dev_scores20.npz'))
    for name in NAMES:
        rows = dev_records[name]
        choice = metrics20.choose_threshold(rows, max_real_fpr=.05)
        report = evaluate_report(rows, choice['threshold'])
        if report['event_counts']['truth'] == 0 or report['real_fpr_percent'] > 5+1e-10:
            raise ValueError('Invalid development operating point')
        calibration['thresholds'][name] = choice
        calibration['dev_predictions_sha256'][name] = save_predictions(output/f'{name}_dev_scores.npz', rows)
        calibration['dev_metrics'][name] = summary(report)
        # The historical adapted models already used this dev calibration rule.
        payload = head_payload if name == NAMES[1] else fine_payload if name == NAMES[2] else None
        if payload is not None:
            if choice['threshold'] != float(payload['threshold']):
                raise ValueError(f'{name}: replayed threshold differs from archived checkpoint')
            for key in ['event_f1_iou05_percent','duration_f1_percent','real_fpr_percent']:
                if not np.isclose(report[key], payload['dev_metrics'][key], rtol=0, atol=1e-7):
                    raise ValueError(f'{name}: raw dev scores fail archived checkpoint metric parity: {key}')
        print(f"Dev calibrated {name}: threshold={choice['threshold']:.17g}, FPR={report['real_fpr_percent']:.8f}%", flush=True)
    save_json(output/'frozen_dev_calibration.json', calibration)
    # Only now open any eval source. No threshold or checkpoint can change below.
    print('Dev thresholds frozen; loading fixed eval records', flush=True)
    eval_done = read_json(old_eval/'complete.json')
    if (eval_done['status'] != 'complete'
            or eval_done['checkpoint_sha256'] != descriptions[NAMES[2]]['checkpoint_sha256']
            or sha256(old_eval/'eval_scores20.npz') != eval_done['eval_scores20_sha256']
            or sha256(args.eval_labels20) != eval_done['eval_labels20_sha256']):
        raise ValueError('Archived fine-tuned eval predictions/labels fail provenance checks')
    test_records = {NAMES[0]:[], NAMES[1]:[]}
    block_lengths = []
    for i, row in enumerate(npz_rows(args.eval_records)):
        test_records[NAMES[0]].append(report_row(row, row['native'], native20=True))
        fine = head(torch.from_numpy(row['native']), torch.from_numpy(row['context']), return_native=True)[2].numpy()
        test_records[NAMES[1]].append(report_row(row, fine, native20=True))
        block_lengths.append(len(row['labels']))
        if (i+1) % 1000 == 0:
            print(f'Replayed original/head scores: {i+1}', flush=True)
    test_uids = [r['uid'] for r in test_records[NAMES[0]]]
    check_ids(test_uids, fixed['eval'], 'eval')
    if set(test_uids) & set(dev.uids):
        raise ValueError('Development and evaluation IDs overlap')
    check_labels(test_records[NAMES[0]], args.eval_labels20)
    test_records[NAMES[2]] = read_saved_scores(old_eval/'eval_scores20.npz', test_records[NAMES[0]], block_lengths)
    reports, score_hashes = {}, {}
    for name in NAMES:
        threshold = calibration['thresholds'][name]['threshold']
        result = evaluate_report(test_records[name], threshold)
        result.update(protocol_version=PROTOCOL, model_id=name, split='eval',
                      checkpoint=descriptions[name], threshold_source='dev real-duration FPR cap5%',
                      checkpoint_reselection=False, eval_previously_inspected=True)
        per_recording = result.pop('per_recording')
        save_json(output/f'{name}_per_recording.json', per_recording)
        result['per_recording_file'] = f'{name}_per_recording.json'
        score_hashes[name] = save_predictions(output/f'{name}_eval_scores.npz', test_records[name])
        result['eval_predictions_sha256'] = score_hashes[name]
        save_json(output/f'{name}_metrics.json', result)
        reports[name] = result
        print(f"Eval {name}: eventF1={result['event_f1_iou05_percent']:.6f}, shortR={result['short_event_recall_le320ms_percent']:.6f}, realFPR={result['real_fpr_percent']:.6f}", flush=True)
    summary_fields = ['event_f1_iou05_percent','event_precision_percent','event_recall_percent',
                      'short_event_recall_le320ms_percent','real_duration_fpr_percent','duration_f1_percent']
    deltas = {name:{key:reports[name][key]-reports[NAMES[0]][key] for key in summary_fields} for name in NAMES[1:]}
    comparison = dict(protocol_version=PROTOCOL, status='complete', dev_recordings=len(dev.uids),
        eval_recordings=len(test_uids), paired_adaptation_seed=666, primary_metric='event_f1_iou05_percent',
        checkpoint_reselection=False, eval_previously_inspected=True,
        models=reports, new_minus_original_percentage_points=deltas,
        scope_note='Fixed historical checkpoint reevaluation; metric/operating-point rules match v2, but checkpoint selection remains historical. Single-seed adaptation comparison, no retraining.',
        eval_uid_order_sha256=hashlib.sha256('\n'.join(test_uids).encode()).hexdigest(),
        eval_labels20_sha256=sha256(args.eval_labels20), eval_manifest_sha256=sha256(args.eval_records),
        fixed_ids_sha256=sha256(args.fixed_ids),
        calibration_sha256=sha256(output/'frozen_dev_calibration.json'))
    save_json(output/'comparison.json', comparison)
    save_json(output/'environment.json', dict(python=platform.python_version(), numpy=np.__version__,
              torch=torch.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, threads=args.threads))
    for name, filename in zip(NAMES, FILES):
        if sha256(weights/filename) != descriptions[name]['checkpoint_sha256']:
            raise ValueError('Checkpoint changed during evaluation')
    save_json(output/'complete.json', dict(status='complete', protocol_version=PROTOCOL,
              models_unchanged=True, threshold_frozen_before_eval=True, eval_recordings=len(test_uids),
              elapsed_seconds=time.monotonic()-started, comparison_sha256=sha256(output/'comparison.json')))
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dev-cache','eval-records','dev-labels20','eval-labels20','finetuned-run',
                 'finetuned-evaluation','weights-dir','fixed-ids','output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--threads', type=int, default=4)
    compare(parser.parse_args())


if __name__ == '__main__':
    main()
