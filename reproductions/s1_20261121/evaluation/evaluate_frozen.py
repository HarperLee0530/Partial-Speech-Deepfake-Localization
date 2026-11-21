"""Frozen A0/S1 evaluation; DEV replay must pass before EVAL is read."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path('/root/autodl-tmp/nndl_partial_speech')
SOURCE = ROOT / 'repo/experiments/wavlm_ablation_ab_20261119/code'
sys.path.insert(0, str(SOURCE))
from cnn_ps.io import manifest_hash, sha256
from cnn_ps.metrics20 import evaluate, decision_mask, _intervals
from cnn_ps.metrics160 import match_events
from cnn_ps.train_dual import infer
from cnn_ps.train_wavlm_multilayer import make_adapter, model_config, setup, ARMS
from cnn_ps.wavlm_multilayer import LayerFusionCNN, MultiLayerData

OUT = ROOT / 'reports/cross_resolution_eval_20261121/formal'
MANIFEST = ROOT / 'reports/cross_resolution_eval_20261121/FROZEN_MANIFEST.json'
LABELS = ROOT / 'reports/eval_label_audit_20261121/eval_native20_fake1.npz'
LABEL_HASH = '63a02187f759f7684f4c8e88e26d0e873b02e0dbe3d690fe0f4546d6e8de66b6'
UID_HASH = '203792d5718e75e54614ad005386a3037cd9ddb83dd6dc7a93266dfc9c4252b1'

def model_and_adapter(checkpoint, seed):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['protocol']['seed'] == seed
    setup(seed, 2)
    model = LayerFusionCNN(model_config(), 'fusion6_9_12', seed)
    adapter = make_adapter(ARMS['fusion6_9_12'], seed, 'cpu')
    model.load_state_dict(payload['backbone_state_dict'], strict=True)
    adapter.load_state_dict(payload['adapter_state_dict'], strict=True)
    return model.cuda(), adapter.cuda(), payload

def score_rows(path, dataset):
    with np.load(path, allow_pickle=False) as archive:
        scores=np.asarray(archive['scores20'],dtype=np.float32)
        uids=archive['uids'].tolist()
        lengths=np.asarray(archive['lengths'],dtype=np.int64)
        lengths20=np.asarray(archive['lengths20'],dtype=np.int64)
        durations=np.asarray(archive['durations'],dtype=np.float64)
    assert uids==dataset.uids and len(scores)==int(lengths.sum())*8
    rows=[]; offset=0
    for uid,n,n20,d in zip(uids,lengths,lengths20,durations):
        score=scores[offset:offset+8*int(n)];offset+=8*int(n)
        assert len(dataset.fine[uid])==n20
        rows.append(dict(uid=uid,labels20=dataset.fine[uid],scores20=score,duration=float(d)))
    return rows, scores, dict(uids=uids,lengths=lengths,lengths20=lengths20,durations=durations)

def event_signature(row,threshold):
    n=len(row['labels20']);d=row['duration']
    pred=_intervals(decision_mask(row['scores20'][:n],threshold),d)
    truth=_intervals(row['labels20'],d)
    matches,_=match_events(pred,truth,threshold=.5)
    return pred,truth,matches

def replay_dev():
    frozen=json.loads(MANIFEST.read_text())
    assert sha256(LABELS)==LABEL_HASH
    dataset=MultiLayerData(ROOT/'cache/fixed_ps/reference/dev',ROOT/'cache/fixed_ps/layers/dev',
        arm='fusion6_9_12',split='dev',seed=666,config=model_config(),verify_hashes=True)
    assert len(dataset)==2484
    result={'status':'running','inference_source_sha256':sha256(Path(__file__)),
            'training_infer_sha256':sha256(SOURCE/'cnn_ps/train_dual.py'),
            'metrics20_sha256':sha256(SOURCE/'cnn_ps/metrics20.py'),'models':{}}
    for ident,entry in frozen['models'].items():
        checkpoint=Path(entry['checkpoint']);run=checkpoint.parent;seed=int(ident[-3:])
        assert run.name==f'{ident}_formal' and 'budget30' not in str(checkpoint)
        assert sha256(checkpoint)==entry['checkpoint_sha256']
        model,adapter,payload=model_and_adapter(checkpoint,seed)
        assert payload['epoch']==entry['best_epoch'] and payload['threshold']==entry['dev_threshold_full_precision']
        records,scores,metadata=infer(model,adapter,dataset,'cuda')
        old_rows,old_scores,old_meta=score_rows(run/'dev_scores20.npz',dataset)
        assert metadata['uids']==old_meta['uids']
        assert np.array_equal(metadata['lengths'],old_meta['lengths'])
        assert np.array_equal(metadata['lengths20'],old_meta['lengths20'])
        assert np.array_equal(metadata['durations'],old_meta['durations'])
        assert scores.shape==old_scores.shape
        max_error=float(np.max(np.abs(scores.astype(np.float64)-old_scores.astype(np.float64))))
        exact=bool(np.array_equal(scores,old_scores))
        threshold=entry['dev_threshold_full_precision']
        boundaries_equal=matches_equal=True
        for actual,old in zip(records,old_rows):
            assert actual['uid']==old['uid'] and actual['duration']==old['duration']
            ap,at,am=event_signature(actual,threshold)
            bp,bt,bm=event_signature(old,threshold)
            boundaries_equal &= np.array_equal(ap,bp) and np.array_equal(at,bt)
            matches_equal &= am==bm
        metrics=evaluate(records,threshold)
        old_metrics=evaluate(old_rows,threshold)
        fields=('event_f1_iou05_percent','event_precision_percent','event_recall_percent','real_fpr_percent')
        metric_deltas={k:float(metrics[k]-old_metrics[k]) for k in fields}
        saved=json.loads((run/'dev_metrics.json').read_text())
        for key in fields:
            assert abs(metrics[key]-saved[key])<1e-10,(ident,key,metrics[key],saved[key])
        assert round(metrics['event_f1_iou05_percent'],4)==round(entry['dev_event_f1_percent'],4)
        assert exact and boundaries_equal and matches_equal and all(v==0 for v in metric_deltas.values()),ident
        result['models'][ident]={'checkpoint_sha256':entry['checkpoint_sha256'],'epoch':entry['best_epoch'],
            'threshold':threshold,'recordings':len(records),'storage_logits':len(scores),'exact_logits':exact,
            'max_abs_logit_error':max_error,'valid_frame_counts_equal':True,'event_boundaries_equal':True,
            'event_matches_equal':True,'metrics':{k:metrics[k] for k in fields},'metric_deltas':metric_deltas}
        print(ident,metrics['event_f1_iou05_percent'],len(scores),max_error,flush=True)
        del model,adapter,payload
        torch.cuda.empty_cache()
    result['status']='passed'
    OUT.mkdir(exist_ok=True)
    (OUT/'DEV_REPRODUCTION_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('DEV_REPRODUCTION_PASSED',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['dev'])
    args=parser.parse_args()
    if args.phase=='dev':replay_dev()
