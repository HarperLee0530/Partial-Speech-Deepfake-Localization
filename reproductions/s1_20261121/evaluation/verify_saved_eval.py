"""Recompute frozen EVAL metrics from saved logits without model inference."""
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path('/root/autodl-tmp/nndl_partial_speech')
OUT=ROOT/'reports/cross_resolution_eval_20261121/formal'
SOURCE=ROOT/'repo/experiments/wavlm_ablation_ab_20261119/code'
sys.path.insert(0,str(SOURCE))
from cnn_ps.io import manifest_hash, save_json, sha256
from cnn_ps.report_native20 import evaluate_report

frozen=json.loads((ROOT/'reports/cross_resolution_eval_20261121/FROZEN_MANIFEST.json').read_text())
reference=json.loads(Path('/root/research/nndl_partial_speech/cache/fixed_ps/reference/eval_20261121/complete.json').read_text())
uids=reference['uids']
assert len(uids)==3562 and manifest_hash(uids)=='203792d5718e75e54614ad005386a3037cd9ddb83dd6dc7a93266dfc9c4252b1'
labels=ROOT/'reports/eval_label_audit_20261121/eval_native20_fake1.npz'
assert sha256(labels)=='63a02187f759f7684f4c8e88e26d0e873b02e0dbe3d690fe0f4546d6e8de66b6'
rows={r['model']:r for r in csv.DictReader((OUT/'EVAL_METRICS.csv').open())}
verified={}
with np.load(labels,allow_pickle=False) as truth:
    assert truth.files==uids
    for ident in ('A0_666','S1_666','A0_667','S1_667'):
        with np.load(OUT/ident/'eval_scores20.npz',allow_pickle=False) as score:
            assert score['uids'].tolist()==uids
            assert score['lengths20'].tolist()==reference['lengths20']
            assert score['lengths'].tolist()==reference['lengths160']
            assert np.array_equal(score['durations'],reference['durations'])
            raw=score['scores20'];assert len(raw)==sum(reference['storage_lengths20']) and np.isfinite(raw).all()
            records=[];offset=0
            for uid,n,d in zip(uids,reference['storage_lengths20'],reference['durations']):
                assert len(truth[uid])==reference['lengths20'][len(records)]
                records.append(dict(uid=uid,labels20=truth[uid],scores20=raw[offset:offset+n],duration=d))
                offset+=n
        threshold=frozen['models'][ident]['dev_threshold_full_precision']
        metric=evaluate_report(records,threshold)
        row=rows[ident]
        for key,column in [('event_f1_iou05_percent','event_f1_percent'),('event_precision_percent','event_precision_percent'),
                           ('event_recall_percent','event_recall_percent'),('frame_f1_percent','frame_f1_percent'),
                           ('real_fpr_percent','real_duration_fpr_percent')]:
            assert abs(metric[key]-float(row[column]))<1e-10,(ident,key)
        assert metric['event_counts']['matched']==int(row['event_tp'])
        assert metric['event_counts']['unmatched_predictions']==int(row['event_fp'])
        assert metric['event_counts']['missed_truth']==int(row['event_fn'])
        verified[ident]=dict(score_sha256=sha256(OUT/ident/'eval_scores20.npz'),
            scores=len(raw),event_f1_percent=metric['event_f1_iou05_percent'],max_metric_difference=0)
save_json(OUT/'EVAL_SCORE_RECALCULATION.json',dict(status='passed',models=verified))
print('EVAL_SCORE_RECALCULATION_PASSED')
