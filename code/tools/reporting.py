import sys
from pathlib import Path
_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

"""One entry for SAL/CNN results: summary, comparison, verification, cached subsets.

Default: compare completed fixed10 PS/LPS results. Never trains a model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from project_config import CODE_ROOT, RESULT_ROOT
from tools.data_tools import read_manifest, profile_info, dataset_paths

ROOT = CODE_ROOT

def find_results(model, split, data_profile='full'):
    # These named files are the user's selected SAL runs. Do not replace them
    # with a later abandoned duplicate whose old status still says running.
    if model == 'sal' and data_profile == 'full':
        selected = RESULT_ROOT/('ps_result.json' if split == 'eval' else 'lps_result.json')
        if selected.is_file():
            data = json.loads(selected.read_text(encoding='utf-8-sig'))
            if data.get('split') != split or data.get('smoke_test') is not False:
                raise ValueError(f'Invalid selected SAL result: {selected}')
            return data, selected.name
    candidates = {}
    patterns = [f'cnn_bigru_{split}_{data_profile}_*.json'] if model == 'cnn_bigru' else [f'sal_wavlm_{split}_{data_profile}_*.json']
    paths = [path for pattern in patterns for path in RESULT_ROOT.glob(pattern)]
    if model == 'sal' and data_profile == 'full':
        paths += [RESULT_ROOT/('ps_result.json' if split == 'eval' else 'lps_result.json')]
    for path in paths:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if (data.get('split') == split and data.get('smoke_test') is False
                and data.get('status') == 'complete' and data.get('data_profile', 'full') == data_profile):
            candidates[data['run']] = (data, path.name)
    return max(candidates.values(), key=lambda row: row[0].get('started_local', '')) if candidates else (None, None)


def compare_pair(sal, baseline):
    if sal is None or baseline is None:
        return dict(comparable=False, reason='One or both models have no completed evaluation on the requested profile')
    if sal.get('status') != 'complete' or baseline.get('status') != 'complete':
        return dict(comparable=False, reason='Both evaluations must be complete')
    if sal.get('data_profile', 'full') != baseline.get('data_profile', 'full'):
        return dict(comparable=False, reason='Data profiles differ: cannot compare full and fixed10 scores')
    fields = ('split', 'manifest_sha256', 'resolution_seconds', 'utterance_count', 'frame_count',
              'f1_rule', 'eer_rule', 'precision', 'batch_size', 'full_utterance')
    mismatches = [field for field in fields if sal.get(field) is None or sal.get(field) != baseline.get(field)]
    if mismatches:
        return dict(comparable=False, reason='Evaluation protocol mismatch', mismatched_fields=mismatches)
    return dict(comparable=True, comparison_scope='Matched evaluation protocol, not matched pretraining/training conditions',
                eer_difference_pp=baseline['eer_percent']-sal['eer_percent'],
                f1_difference_pp=baseline['f1_bonafide_percent']-sal['f1_bonafide_percent'])


def summarize_record(data, file):
    if data is None:
        return dict(status='not_started')
    row = dict(file=file, status=data.get('status'), completed=data.get('completed'),
               total=data.get('planned_utterances'), checkpoint_sha256=data.get('checkpoint_sha256'))
    if data.get('status') == 'complete':
        row.update({field: data.get(field) for field in ('eer_percent', 'f1_bonafide_percent', 'accuracy_percent')})
    return row


def compare(baseline_paths=None, data_profile='fixed10', sal_paths=None):
    rows, baseline_hashes = [], []
    for split in ('eval', 'lps'):
        if sal_paths is None:
            sal, sal_file = find_results('sal', split, data_profile)
        else:
            sal_path = (CODE_ROOT/Path(sal_paths[split])).resolve()
            sal = json.loads(sal_path.read_text(encoding='utf-8-sig'))
            sal_file = sal_path.name
            if sal.get('split') != split or sal.get('data_profile', 'full') != data_profile or sal.get('smoke_test'):
                raise ValueError('Explicit SAL results have wrong split/profile')
        if baseline_paths is None:
            baseline, baseline_file = find_results('cnn_bigru', split, data_profile)
        else:
            path = Path(baseline_paths[split])
            baseline = json.loads(path.read_text(encoding='utf-8-sig'))
            if baseline.get('split') != split or baseline.get('smoke_test') is not False:
                raise ValueError(f'Invalid requested CNN-BiGRU result: {path}')
            baseline_file = path.name
        if baseline and baseline.get('status') == 'complete':
            baseline_hashes.append(baseline['checkpoint_sha256'])
        rows.append(dict(split=split, sal=summarize_record(sal, sal_file),
                         cnn_bigru=summarize_record(baseline, baseline_file), **compare_pair(sal, baseline)))
    result = dict(updated_local=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), data_profile=data_profile, comparisons=rows,
                  same_baseline_checkpoint_on_both_tests=(len(set(baseline_hashes)) == 1) if len(baseline_hashes) == 2 else None,
                  note='SAL uses pretrained WavLM; consult SAL result checkpoint_kind for public vs retrained weights. CNN-BiGRU is trained from scratch. No architecture-only causal claim.')
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    path = RESULT_ROOT/f'baseline_comparison_{stamp}.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('RESULT_FILE', path)
    return path


def verify_run(run_directory):
    import sqlite3
    import numpy as np
    from sklearn.metrics import confusion_matrix, f1_score, roc_curve
    run_directory = Path(run_directory)
    if not run_directory.is_absolute():
        run_directory = ROOT/run_directory
    run_directory = run_directory.resolve()
    m = json.loads((run_directory/'metrics.json').read_text())
    db = sqlite3.connect((run_directory/'predictions.sqlite').as_uri()+'?mode=ro', uri=True)
    rows = db.execute('SELECT logits, labels FROM predictions ORDER BY ordinal').fetchall()
    db.close()
    assert len(rows) == m['utterance_count'] == m['planned_utterances']
    ys, ps = [], []
    for raw_p, raw_y in rows:
        p = np.frombuffer(raw_p, dtype=np.float32).reshape(-1, 2)
        y = np.frombuffer(raw_y, dtype=np.uint8)
        assert len(p) == len(y)
        assert np.isfinite(p).all()
        ys.append(y)
        ps.append(p)
    y, p = np.concatenate(ys), np.concatenate(ps)
    assert len(y) == m['frame_count']
    pred = p.argmax(axis=1)
    f1 = 100*f1_score(y, pred, pos_label=1)
    assert abs(f1-m['f1_bonafide_percent']) < 0.0001
    assert confusion_matrix(y,pred).tolist() == m['confusion_true_rows_pred_columns_0_spoof_1_bonafide']
    fpr,tpr,_ = roc_curve(y,p[:,1]-p[:,0], drop_intermediate=False)
    index = np.argmin(abs(fpr-(1-tpr)))
    roc_eer = 100*fpr[index]
    result = dict(status='passed', verified_utterances=len(rows), frames=len(y),
        sklearn_f1_bonafide_percent=f1, official_f1_percent=m['f1_bonafide_percent'],
        sklearn_nearest_roc_eer_percent=roc_eer, official_eer_percent=m['eer_percent'],
        eer_note='ROC groups equal-score ties; official implementation sorts individual frames. Differences may occur at ties.')
    (run_directory/'independent_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    
    return result


def extract_subset(split):
    from sal import pipeline as sal
    sal.torch.set_num_threads(4)
    source_path = RESULT_ROOT/('ps_result.json' if split == 'eval' else 'lps_result.json')
    source = json.loads(source_path.read_text(encoding='utf-8-sig'))
    if source.get('status') != 'complete' or source.get('smoke_test') is not False or source.get('data_profile', 'full') != 'full':
        raise ValueError('Expected a completed full evaluation')
    manifest = read_manifest(split)
    if source['utterance_count'] != manifest['official_count'] or source['split'] != split:
        raise ValueError('Wrong full source split / count')
    database_path = RESULT_ROOT/source['predictions_file']
    source_db = sqlite3.connect(database_path.as_uri()+'?mode=ro', uri=True)
    if source_db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0] != source['utterance_count']:
        raise ValueError('Incomplete source prediction database')
    _, label_root = dataset_paths(split, 'fixed10')
    labels = np.load(label_root/('cf_seglab_0.02.npy' if split == 'lps' else 'eval_seglab_0.16.npy'), allow_pickle=True).item()
    run, out = sal.begin_run(split, 0, data_profile='fixed10')
    started = time.monotonic()
    target = sqlite3.connect(out/'predictions.sqlite')
    try:
        target.execute('CREATE TABLE predictions (uid TEXT PRIMARY KEY, ordinal INTEGER UNIQUE, logits BLOB, labels BLOB, seconds REAL, runtime REAL, device TEXT)')
        for ordinal, uid in enumerate(manifest['utterance_ids']):
            row = source_db.execute('SELECT uid, ordinal, logits, labels, seconds, runtime, device FROM predictions WHERE uid=?', (uid,)).fetchone()
            if row is None or not np.array_equal(np.frombuffer(row[3], dtype=np.uint8), np.asarray(labels[uid]).astype(np.uint8)):
                raise ValueError(f'Missing prediction or label mismatch: {uid}')
            logits = np.frombuffer(row[2], dtype=np.float32).reshape(-1, 2)
            if len(logits) != len(labels[uid]) or not np.isfinite(logits).all():
                raise ValueError(f'Invalid logits: {uid}')
            target.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?,?)', (uid, ordinal, *row[2:]))
        target.commit()
        config = {key: source[key] for key in ('checkpoint', 'checkpoint_sha256', 'official_source_commit',
                  'resolution_seconds', 'precision', 'batch_size', 'full_utterance', 'packages')}
        config.update(run=run, split=split, smoke_test=False, total_dataset=manifest['count'],
                      planned_utterances=manifest['count'], manifest_sha256=manifest['manifest_sha256'],
                      **profile_info(split, 'fixed10'), inference_reused=True,
                      source_full_run=source['run'], source_full_result=source_path.name,
                      source_result_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                      source_predictions_file=source['predictions_file'],
                      runtime_note='No new model inference; per-utterance runtimes copied from original full run. Not a fresh subset speed benchmark.')
        metrics = sal.calculate_metrics(target)
        seconds, runtime = target.execute('SELECT SUM(seconds),SUM(runtime) FROM predictions').fetchone()
        metrics.update(config, audio_seconds=seconds, reused_inference_seconds=runtime,
                       extraction_and_metrics_seconds=time.monotonic()-started)
        sal.write_json(out/'config.json', config)
        sal.write_json(out/'metrics.json', metrics)
        sal.write_json(out/'status.json', dict(status='complete', run=run, completed=manifest['count'], total=manifest['count']))
        sal.record_run('complete', completed=manifest['count'], **metrics)
        print(json.dumps(dict(run=run, split=split, count=manifest['count'], eer=metrics['eer_percent'], f1=metrics['f1_bonafide_percent']), ensure_ascii=False), flush=True)
        return RESULT_ROOT/(run+'.json')
    except BaseException as error:
        sal.finish_failed_run('failed', error)
        raise
    finally:
        target.close()
        source_db.close()

def summarize(data_profile='fixed10'):
    rows = []
    for split in ('dev', 'eval', 'lps'):
        result, file = find_results('sal', split, data_profile)
        row = dict(split=split, data_profile=data_profile, **summarize_record(result, file))
        if result:
            row.update(utterance_count=result.get('utterance_count'),
                       manifest_sha256=result.get('manifest_sha256'),
                       inference_reused=result.get('inference_reused', False))
        rows.append(row)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    path = RESULT_ROOT/f'sal_summary_{data_profile}_{stamp}.json'
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print('RESULT_FILE', path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('compare', 'summary', 'verify', 'subset'), default='compare')
    parser.add_argument('--data-profile', choices=('fixed10', 'full'), default='fixed10')
    parser.add_argument('--run', type=Path, help='Run directory, relative to code or absolute')
    parser.add_argument('--split', choices=('eval', 'lps', 'both'), default='both')
    parser.add_argument('--sal-ps', help='Explicit retrained/custom SAL PS result JSON, relative to code')
    parser.add_argument('--sal-lps', help='Explicit retrained/custom SAL LPS result JSON, relative to code')
    args = parser.parse_args(argv)
    if args.action == 'compare':
        if bool(args.sal_ps) != bool(args.sal_lps):
            parser.error('Provide both --sal-ps and --sal-lps for explicit SAL comparisons')
        return compare(data_profile=args.data_profile,
                       sal_paths={'eval':args.sal_ps,'lps':args.sal_lps} if args.sal_ps else None)
    if args.action == 'summary':
        return summarize(args.data_profile)
    if args.action == 'verify':
        if args.run is None:
            parser.error('--action verify requires --run')
        return verify_run(args.run)
    if args.data_profile != 'fixed10':
        parser.error('--action subset requires --data-profile fixed10')
    for split in ('eval', 'lps') if args.split == 'both' else (args.split,):
        extract_subset(split)


if __name__ == '__main__':
    main()
