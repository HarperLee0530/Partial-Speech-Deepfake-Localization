"""Read-only archive audit and metric recomputation; Python standard library only."""
from pathlib import Path, PureWindowsPath
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent
ARMS = ('A0_full', 'A5_fixed_pool')

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    manifest = read(ROOT / 'archive_manifest.json')
    for entry in manifest['included_original_files']:
        p = ROOT / entry['path']
        require(p.is_file() and p.stat().st_size == entry['bytes'], f"Missing/size mismatch: {p}")
        require(digest(p) == entry['sha256'], f"Hash mismatch: {p}")
    plan = read(ROOT / 'results/plan.json')
    for rel, expected in plan['source_sha256'].items():
        require(digest(ROOT.joinpath(*PureWindowsPath(rel).parts)) == expected, f'Source changed: {rel}')
    frozen = read(ROOT / 'results/all_four_frozen.json')
    require(digest(ROOT / 'results/plan.json') == frozen['plan_sha256'], 'Frozen plan mismatch')
    summary = read(ROOT / 'results/three_seed_summary.json')
    results = {}
    uids = None
    for seed in (666, 667, 668):
        protocols = []
        for arm in ARMS:
            folder = ROOT / ('reference/seed666' if seed == 666 else f'results/seed{seed}') / arm
            complete = read(folder / 'complete.json')
            metrics = read(folder / 'eval_metrics.json')
            records = read(folder / 'eval_per_recording.json')
            protocol = read(folder / 'protocol.json')
            protocols.append(protocol)
            current = [x['uid'] for x in records]
            require(len(current) == 3562 and len(set(current)) == 3562, 'Unexpected eval UIDs')
            if uids is None:
                uids = current
            require(current == uids, 'Eval UID order mismatch')
            total = lambda key: sum(x[key] for x in records)
            event_f1 = 200 * total('event_tp') / (total('event_pred') + total('event_truth'))
            fpr = 100 * total('fp_seconds') / (total('fp_seconds') + total('tn_seconds'))
            short_recall = 100 * total('short_tp') / total('short_truth')
            for key, value in [('event_f1_iou05_percent', event_f1), ('real_duration_fpr_percent', fpr), ('short_event_recall_le320ms_percent', short_recall)]:
                require(math.isclose(value, metrics[key], abs_tol=1e-8, rel_tol=0), f'Metric mismatch: {seed}/{arm}/{key}')
            require(digest(folder / 'best.pt') == complete['checkpoint_sha256'], 'Checkpoint mismatch')
            require(complete['trained_epochs'] == 20 and complete['updates'] == 15880, 'Training budget mismatch')
            require(complete['threshold'] == metrics['threshold'], 'Evaluation threshold changed')
            if seed != 666:
                history = read(folder / 'history.json')
                selected = max(history, key=lambda x: x['event_f1_iou05_percent'])
                require(len(history) == 21 and selected['epoch'] == complete['epoch'], 'Dev selection mismatch')
                require(frozen['models'][str(seed)][arm] == complete, 'Pre-eval freeze changed')
            results[seed, arm] = (event_f1, fpr, short_recall)
        keys = ['initial_backbone_sha256', 'train_uids_sha256', 'dev_uids_sha256', 'seed', 'epochs', 'batch_size', 'learning_rate', 'weight_decay', 'gradient_clip']
        if seed != 666:
            keys += ['initial_projection_sha256', 'initial_adapter_sha256']
        for key in keys:
            require(protocols[0][key] == protocols[1][key], f'Paired condition mismatch: {seed}/{key}')
    print(f"PASS: {len(manifest['included_original_files'])} original files, source hashes, frozen models, budgets, selection and metrics.")
    print('seed | A0 eventF1 | A5 eventF1 | delta pp | A0 FPR | A5 FPR')
    for seed in (666, 667, 668):
        a, b = (results[seed, arm] for arm in ARMS)
        print(f'{seed} | {a[0]:.4f} | {b[0]:.4f} | {b[0]-a[0]:+.4f} | {a[1]:.4f} | {b[1]:.4f}')
    for arm in ARMS:
        vals = [results[s, arm][0] for s in (666, 667, 668)]
        stored = summary['arms'][arm]['event_f1_iou05_percent']
        require(math.isclose(statistics.mean(vals), stored['mean'], abs_tol=1e-8), 'Summary mean mismatch')
        require(math.isclose(statistics.stdev(vals), stored['sample_sd'], abs_tol=1e-8), 'Summary SD mismatch')
        print(f'{arm}: event F1 {statistics.mean(vals):.4f} +/- {statistics.stdev(vals):.4f}% (sample SD)')
    print('Historical seed666 A0 lacks projection/adapter initial hashes; these fields are audited only for new seeds.')
    print('UID split integrity was checked during the experiment; raw datasets are not included for independent split rechecking.')

if __name__ == '__main__':
    main()
