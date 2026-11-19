"""Recalibrate frozen audio-review evidence to a total genuine-duration FPR cap.

No CNN, crop feature or reviewer is refit. Report both pooled development
calibration (measured FPR <=5%) and separate-fold threshold generalization.
"""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .pos_probe import read_json,write_json,sha
from .audio_review_probe import candidates,evaluate_masks
from .second_review_probe import context_features,compact
from . import metrics20

ARMS=('baseline','context_review','audio_review')


def rescore(raw,risk,review_cut,original_cut):
    """Map review-cut-relative evidence to a bounded upward ranking correction.

    At original_cut, this exactly reproduces the earlier add-only decisions.
    A newly calibrated FINAL cut may revoke original low-score positives.
    Only original candidate scores change; no authenticity labels enter here.
    """
    raw=np.asarray(raw,dtype=np.float64);risk=np.asarray(risk,dtype=np.float64)
    if raw.shape!=risk.shape or not np.isfinite(raw).all() or not np.isfinite(risk).all():
        raise ValueError('Invalid score arrays')
    mask=candidates(raw,original_cut)
    proposal=original_cut+(risk-review_cut)
    adjusted=np.minimum(raw+1.,np.maximum(raw,proposal))
    # Preserve the existing exact float64 decision across cancellation at ties.
    below=mask&(risk<review_cut)
    above=mask&(risk>=review_cut)
    adjusted[below]=np.minimum(adjusted[below],np.nextafter(float(original_cut),-np.inf))
    adjusted[above]=np.maximum(adjusted[above],float(original_cut))
    return np.where(mask,adjusted,raw)


def calibrated_cut(records,scores):
    rows=[dict(uid=r['uid'],duration=r['duration'],labels20=r['labels20'],scores20=s) for r,s in zip(records,scores)]
    return metrics20.choose_threshold(rows,.05)


def bootstrap(left,right,records,folds,seed):
    """Fixed prediction paired intervals, stratified by the original held-out fold."""
    def counts(masks):
        event=evaluate_masks(records,masks)['per_recording'];out=[]
        for r,p,e in zip(records,masks,event):
            y=r['labels20'].astype(bool)
            out.append([e['event_tp'],e['event_pred'],e['event_truth'],
                        int(np.sum(p&y)),int(np.sum(p&~y)),int(np.sum(~p&y)),e['fp_seconds'],e['tn_seconds']])
        return np.array(out,dtype=np.float64)
    a,b=counts(left),counts(right)
    def measure(x):
        et,ep,eg,tp,fp,fn,fs,ts=x.sum(axis=0)
        return np.array([200*et/(ep+eg),200*tp/(2*tp+fp+fn),100*fs/(fs+ts)])
    rng=np.random.default_rng(seed);groups=[np.flatnonzero(folds==f) for f in range(5)]
    draws=[]
    for _ in range(2000):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        draws.append(measure(a[ix])-measure(b[ix]))
    point=measure(a)-measure(b);ci=np.quantile(draws,[.025,.975],axis=0)
    return {name:dict(delta_pp=float(point[i]),fixed_predictions_95_ci_pp=ci[:,i].tolist()) for i,name in enumerate(['event_f1','frame_f1','real_fpr'])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Refusing to overwrite experiment')
    args.output.mkdir(parents=True)
    source_protocol=read_json(args.source/'protocol.json')
    if read_json(args.source/'verification.json')['status']!='passed':raise ValueError('Source audio-review audit did not pass')
    inputs={name:dict(path=str(args.source/name),sha256=sha(args.source/name)) for name in ['protocol.json','verification.json','oof_review.npz','audio_features.npz','fitted_reviewers.json','fold_results.json','summary.json']}
    original_cut=source_protocol['first_pass_threshold_logit']
    protocol=dict(status='declared_before_new_calibration',source_inputs=inputs,original_cut=original_cut,
                  source_hashes={str(Path(__file__).with_name(n)):sha(Path(__file__).with_name(n)) for n in ['audio_review_fpr5.py','audio_review_probe.py','second_review_probe.py','metrics20.py']},
                  transformation='Candidate s_new=min(s+1,max(s, original_cut+review_logodds-frozen_review_cut)); preserve original tie decisions',
                  no_new_reviewer_fit=True,no_new_waveform_inference=True,no_test_access=True,
                  arms=list(ARMS),max_real_fpr=.05,
                  primary='Pooled OOF development scores: separately calibrate every arm on all genuine dev time to <=5%, then report calibration-set event/frame F1.',
                  secondary='For each old fold k, reconstruct the SAME reviewer on calibration fold (k+1)%5, select total-FPR<=5% cut there, and apply to held-out k.',
                  seed=20261116,main_comparison='audio_review minus baseline',
                  limitations=['Pooled dev evaluation uses labels for the final threshold and is not independent validation.',
                               'Detector and original threshold were already selected on dev; even separated-fold results remain exploratory.',
                               '5% calibration constraint does not guarantee 5% on held-out records or future audio.',
                               'This allows original low-confidence positives to be withdrawn by a higher final cut.',
                               'All review evidence, candidate band, contexts, coefficient fits and old review cuts remain fixed; no parameter search.',
                               'Bootstrap conditions on fixed scores/cuts and does not capture fit/calibration/selection uncertainty.'])
    write_json(args.output/'protocol.json',protocol)
    with np.load(args.source/'oof_review.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    with np.load(args.source/'audio_features.npz',allow_pickle=False) as z:audio={k:z[k] for k in z.files}
    np.testing.assert_equal(a['uids'],audio['uids']);np.testing.assert_equal(a['lengths'],audio['lengths'])
    models={(m['fold'],m['arm']):m for m in read_json(args.source/'fitted_reviewers.json')}
    old_cuts=read_json(args.source/'fold_results.json')
    offsets=np.r_[0,np.cumsum(a['lengths'])];records=[]
    scores={arm:[] for arm in ARMS};folds=a['folds']
    for i,uid in enumerate(a['uids']):
        sl=slice(offsets[i],offsets[i+1]);raw=a['scores'][sl];w=a['widths'][sl];fold=int(folds[i])
        r=dict(uid=str(uid),duration=float(w.sum()),labels20=a['labels'][sl],widths=w,raw=raw,
               context=context_features(raw,w),audio=audio['features'][sl])
        records.append(r);scores['baseline'].append(raw)
        for arm,oldarm in [('context_review','context_add'),('audio_review','audio_add')]:
            result=rescore(raw,a[oldarm+'_risk'][sl],old_cuts[fold]['arms'][oldarm]['threshold'],original_cut)
            np.testing.assert_equal(result>=original_cut,a[oldarm][sl])
            scores[arm].append(result)
    pooled={};pooled_masks={};pooled_choices={}
    for arm in ARMS:
        choice=calibrated_cut(records,scores[arm]);cut=choice['threshold']
        masks=[s>=cut for s in scores[arm]];result=evaluate_masks(records,masks)
        if result['real_fpr_percent']>5+1e-9:raise AssertionError('Pooled calibration did not meet requested cap')
        pooled[arm]=compact(result);pooled_masks[arm]=masks;pooled_choices[arm]=choice
    print('Pooled exact-budget calibration complete',flush=True)
    held_masks={arm:[None]*len(records) for arm in ARMS};held_cuts=[]
    for fold in range(5):
        ci=np.flatnonzero(folds==(fold+1)%5);ti=np.flatnonzero(folds==fold)
        fold_result=dict(fold=fold,arms={})
        for arm in ARMS:
            cal_scores=[]
            for i in ci:
                r=records[i]
                if arm=='baseline':s=r['raw']
                else:
                    oldarm='audio_add' if arm=='audio_review' else 'context_add';model=models[fold,oldarm]
                    x=np.column_stack([r['context'],r['audio']]) if arm=='audio_review' else r['context']
                    risk=((x-np.array(model['mean']))/np.array(model['scale']))@np.array(model['coefficient'])+model['intercept']
                    s=rescore(r['raw'],risk,old_cuts[fold]['arms'][oldarm]['threshold'],original_cut)
                cal_scores.append(s)
            choice=calibrated_cut([records[i] for i in ci],cal_scores)
            for i in ti:held_masks[arm][i]=scores[arm][i]>=choice['threshold']
            fold_result['arms'][arm]=choice
        held_cuts.append(fold_result)
    held={arm:compact(evaluate_masks(records,masks)) for arm,masks in held_masks.items()}
    comparisons={mode:{'audio_minus_'+arm:bootstrap(masks['audio_review'],masks[arm],records,folds,20261116) for arm in ['baseline','context_review']} for mode,masks in [('pooled_calibration',pooled_masks),('separate_fold',held_masks)]}
    summary=dict(status='complete',protocol_sha256=sha(args.output/'protocol.json'),recordings=len(records),
                 pooled_calibration_results=pooled,pooled_thresholds=pooled_choices,
                 separate_fold_results=held,comparisons=comparisons,limitations=protocol['limitations'])
    write_json(args.output/'summary.json',summary);write_json(args.output/'fold_thresholds.json',held_cuts)
    np.savez_compressed(args.output/'decisions.npz',**{f'pooled_{arm}':np.concatenate(pooled_masks[arm]) for arm in ARMS},
                        **{f'held_{arm}':np.concatenate(held_masks[arm]) for arm in ARMS},
                        **{f'scores_{arm}':np.concatenate(scores[arm]) for arm in ARMS},
                        labels=a['labels'],widths=a['widths'],uids=a['uids'],lengths=a['lengths'],folds=folds)
    print(__import__('json').dumps({mode:{arm:{k:v[k] for k in ['event_f1_iou05_percent','frame_f1_percent','real_fpr_percent','duration_recall_percent','fn_seconds','fp_seconds']} for arm,v in block.items()} for mode,block in [('pooled_calibration',pooled),('separate_fold',held)]},ensure_ascii=False,indent=2),flush=True)
    print(__import__('json').dumps(comparisons,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
