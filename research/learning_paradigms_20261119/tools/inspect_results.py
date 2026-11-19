"""Audit the research archive without training data, weights, GPU or third-party packages."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Verify all archived file hashes and per-record aggregate event/FPR metrics')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'archive_manifest.json').read_text(encoding='utf-8'))
    if args.verify:
        for entry in manifest['files']:
            path = ROOT/entry['path']
            assert path.is_file(), path
            assert path.stat().st_size == entry['bytes'], path
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], path
        checked = 0
        for path in (ROOT/'experiments').rglob('*_metrics.json'):
            d = json.loads(path.read_text(encoding='utf-8'))
            rows = d.get('per_recording', [])
            if not rows or 'event_tp' not in rows[0]:
                continue
            tp = sum(x['event_tp'] for x in rows)
            den = sum(x['event_pred']+x['event_truth'] for x in rows)
            expected = 200*tp/den if den else 0.
            assert abs(expected-d['event_f1_iou05_percent']) < 1e-8, path
            fp = sum(x['fp_seconds'] for x in rows)
            tn = sum(x['tn_seconds'] for x in rows)
            expected = 100*fp/(fp+tn) if fp+tn else 0.
            assert abs(expected-d['real_duration_fpr_percent']) < 1e-8, path
            checked += 1
        print(f'PASS: {len(manifest["files"])} exact archived files; {checked} per-record event/FPR aggregates.')
    print('Experiment | CSV rows | entry script')
    for folder in sorted((ROOT/'experiments').iterdir()):
        with (folder/'results/summary.csv').open(encoding='utf-8-sig', newline='') as f:
            count = sum(1 for _ in csv.DictReader(f))
        script = next((n for n in ('run.py','suite.py','run_experiment.py') if (folder/n).exists()), '?')
        print(f'{folder.name} | {count} | {script}')
    print('\nEvent F1 (%): clean_full / noise10 / noise0 / pink10')
    for name in ('multilayer_noise_replay_20261119','multilayer_noise_replay31_20261119',
                 'multilayer_noise_replay51_20261119','multilayer_noise_replay31_lr5e5_20261119'):
        d = json.loads((ROOT/'experiments'/name/'results/means.json').read_text())['replay']
        values = ' / '.join(f'{d[c]["event_f1"]:.3f}' for c in ('clean_full','noise10','noise0','pink10'))
        print(f'{name}: {values}')
    print('Archived results are exploratory dev analyses, not independent blind-test evidence.')

if __name__ == '__main__':
    main()
