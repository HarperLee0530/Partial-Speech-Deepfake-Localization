"""Exploratory dev-only contextual second review of near-threshold real frames.

The detector is frozen. Every error-reviewer evaluation recording is excluded
from reviewer fitting and threshold calibration. Detector selection itself used
this dev set previously, so this is not a new independent test result.
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path
import numpy as np
from scipy.ndimage import convolve1d
from scipy.special import expit
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from . import metrics20
from .pos_probe import read_json,write_json,sha

STEP=.02
FEATURES=['logit','probability','mean_100ms','std_100ms','mean_340ms','std_340ms',
          'mean_660ms','std_660ms','left_max_160ms','right_max_160ms',
          'previous_probability','following_probability']


def candidate_mask(scores,threshold):
    scores=np.asarray(scores,dtype=np.float64)
    return (scores>=threshold-1.)&(scores<threshold)


def context_features(scores,widths):
    """Recording-local features, no authenticity labels or other recordings."""
    s=np.asarray(scores,dtype=np.float64);w=np.asarray(widths,dtype=np.float64)
    if not len(s) or len(s)!=len(w) or not np.isfinite(s).all() or np.any(w<=0):
        raise ValueError('Expected nonempty active finite scores/widths')
    p=expit(np.clip(s,-30,30));columns=[np.clip(s,-30,30),p]
    for radius in (2,8,16):
        kernel=np.ones(2*radius+1)
        denominator=convolve1d(w,kernel,mode='constant',cval=0.)
        mean=convolve1d(w*p,kernel,mode='constant',cval=0.)/denominator
        mean2=convolve1d(w*p*p,kernel,mode='constant',cval=0.)/denominator
        columns.extend([mean,np.sqrt(np.maximum(0,mean2-mean*mean))])
    t=np.arange(len(s))
    left=np.maximum.reduce([p[np.clip(t-k,0,len(s)-1)] for k in range(1,9)])
    right=np.maximum.reduce([p[np.clip(t+k,0,len(s)-1)] for k in range(1,9)])
    columns.extend([left,right,p[np.maximum(t-1,0)],p[np.minimum(t+1,len(s)-1)]])
    return np.column_stack(columns)


def revise(scores,review_logodds,threshold):
    """Only boost original near-threshold real frames, by at most one logit."""
    scores=np.asarray(scores,dtype=np.float64)
    review_logodds=np.asarray(review_logodds,dtype=np.float64)
    if scores.shape!=review_logodds.shape or not np.isfinite(review_logodds).all():
        raise ValueError('Invalid review outputs')
    mask=candidate_mask(scores,threshold)
    correction=np.where(mask,np.clip(review_logodds-scores,0.,1.),0.)
    return scores+correction


def load(args):
    manifest=read_json(args.manifest)
    if manifest['split']!='dev':raise ValueError('Only dev inputs are allowed')
    scores=np.load(args.scores,allow_pickle=False);labels=np.load(args.labels,allow_pickle=False)
    uids=list(map(str,scores['uids']));offsets=np.r_[0,np.cumsum(scores['lengths']*8)]
    if set(uids)!=set(labels.files) or set(uids)!={r['uid'] for r in manifest['records']}:
        raise ValueError('Mismatched UID sets')
    if len(set(uids))!=len(uids) or offsets[-1]!=len(scores['scores20']):raise ValueError('Invalid score archive')
    records=[]
    for i,uid in enumerate(uids):
        y=labels[uid];d=float(scores['durations'][i])
        if len(y)!=scores['lengths20'][i] or not np.isin(y,[0,1]).all():raise ValueError('Invalid reference labels')
        starts=np.arange(len(y))*STEP;widths=np.maximum(0,np.minimum(starts+STEP,d)-starts)
        if not np.isclose(widths.sum(),d,atol=1e-8,rtol=0):raise ValueError('Insufficient labels')
        active=widths>0;s=np.asarray(scores['scores20'][offsets[i]:offsets[i+1]][:len(y)][active],dtype=np.float64)
        w=widths[active]
        records.append(dict(uid=uid,labels20=y[active],scores20=s,duration=d,widths=w,features=context_features(s,w)))
    return records


def score_records(records,score_list):
    return [dict(uid=r['uid'],labels20=r['labels20'],duration=r['duration'],scores20=s) for r,s in zip(records,score_list)]


def compact(result):
    return {k:v for k,v in result.items() if k not in ['per_recording','selection_score_j_percent']}


def fit_reviewer(records,threshold,context):
    xx=[];yy=[];ww=[]
    for r in records:
        selected=candidate_mask(r['scores20'],threshold)
        xx.append(r['features'][selected] if context else r['features'][selected,:1])
        yy.append(r['labels20'][selected]);ww.append(r['widths'][selected])
    x=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww)
    if len(np.unique(y))!=2:raise ValueError('Reviewer needs both candidate classes')
    scaler=StandardScaler().fit(x,sample_weight=w)
    model=LogisticRegression(C=1.,solver='lbfgs',max_iter=1200,tol=1e-7)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(scaler.transform(x),y,sample_weight=w/STEP)
    def apply(record):
        features=record['features'] if context else record['features'][:,:1]
        return revise(record['scores20'],model.decision_function(scaler.transform(features)),threshold)
    state=dict(context=context,training_candidates=len(y),training_candidate_seconds=float(w.sum()),
               scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),
               coefficient=model.coef_[0].tolist(),intercept=float(model.intercept_[0]),iterations=int(model.n_iter_[0]))
    return apply,state


def paired_delta(left,right,fold_ids,seed):
    def counts(result):return np.array([[r['event_tp'],r['event_pred'],r['event_truth'],r['fp_seconds'],r['tn_seconds']] for r in result['per_recording']])
    a,b=counts(left),counts(right)
    def metrics(x):
        t,p,g,fp,tn=x.sum(0)
        return np.array([200*t/(p+g),100*fp/(fp+tn)])
    rng=np.random.default_rng(seed);groups=[np.flatnonzero(fold_ids==f) for f in range(5)];samples=[]
    for _ in range(2000):
        idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        samples.append(metrics(a[idx])-metrics(b[idx]))
    ci=np.quantile(samples,[.025,.975],axis=0)
    diff=metrics(a)-metrics(b)
    return dict(event_f1_delta_pp=float(diff[0]),event_f1_delta_95_ci_pp=ci[:,0].tolist(),
                real_fpr_delta_pp=float(diff[1]),real_fpr_delta_95_ci_pp=ci[:,1].tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['scores','labels','manifest','metrics','checkpoint','output']:
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--seed',type=int,default=20261116)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Refusing to overwrite existing results')
    args.output.mkdir(parents=True)
    archived=read_json(args.metrics);threshold=float(archived['threshold'])
    protocol=dict(status='declared_before_fit',seed=args.seed,
                  source_hashes={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('pos_probe.py'),Path(metrics20.__file__)]},
                  inputs={k:dict(path=str(getattr(args,k)),sha256=sha(getattr(args,k))) for k in ['scores','labels','manifest','metrics','checkpoint']},
                  fixed_detector_logit_threshold=threshold,fixed_detector_probability_threshold=float(expit(threshold)),
                  candidate_logit_range=[threshold-1,threshold],candidate_upper_exclusive=True,
                  candidate_probability_range=[float(expit(threshold-1)),float(expit(threshold))],
                  reviewer='Logistic regression C=1, no class weights; native duration weighted; features standardized on fit records only',
                  correction='candidate only: score += clip(reviewer_logodds - score, 0, 1)',features=FEATURES,
                  split='Five fixed UID folds. Held-out fold k; calibration fold (k+1)%5; other three fit. No reviewer parameter search.',
                  calibration='Most permissive threshold with genuine-duration FPR <=5% on calibration fold only; each arm calibrated separately.',
                  arms=['original_fixed','lower_threshold_only','raw_recalibrated','score_only_review','context_review'],
                  lower_threshold_only='Original frozen threshold minus 0.5 logit; descriptive recall/FPR tradeoff, not budget-matched',
                  primary_comparison='context_review minus raw_recalibrated',secondary_comparison='context_review minus score_only_review',
                  limitations=['The detector and original threshold were already selected on this dev set: this remains exploratory.',
                               'Grouping is by UID, not speaker/source. Fold fits overlap.',
                               'Same 5% calibration constraint does not guarantee equal held-out FPR.',
                               'Bootstrap resamples fixed fold predictions; no retraining, recalibration, or adjustment for multiple analyses.',
                               'Context reviewer reuses detector scores and brings no independent waveform evidence.',
                               'No test-set access or CNN retraining; no promotion to main model based on this probe.'])
    write_json(args.output/'protocol.json',protocol)
    records=load(args);n=len(records);fold_ids=np.zeros(n,dtype=int)
    for fi,(_,idx) in enumerate(KFold(5,shuffle=True,random_state=args.seed).split(records)):fold_ids[idx]=fi
    original=metrics20.evaluate(records,threshold)
    for key in ['event_f1_iou05_percent','real_fpr_percent','fn_seconds','fp_seconds']:
        if not np.isclose(original[key],archived[key],atol=1e-7,rtol=0):raise ValueError('Frozen baseline mismatch')
    print(f'Loaded {n} recordings; frozen baseline event F1={original["event_f1_iou05_percent"]:.4f}',flush=True)
    arms=['raw_recalibrated','score_only_review','context_review']
    margins={arm:[None]*n for arm in arms};fold_results=[];models=[]
    for fold in range(5):
        test_idx=np.flatnonzero(fold_ids==fold);cal_idx=np.flatnonzero(fold_ids==(fold+1)%5)
        train_idx=np.flatnonzero((fold_ids!=fold)&(fold_ids!=(fold+1)%5))
        assert not(set(train_idx)&set(cal_idx) or set(train_idx)&set(test_idx) or set(cal_idx)&set(test_idx))
        train=[records[i] for i in train_idx];cal=[records[i] for i in cal_idx];test=[records[i] for i in test_idx]
        fold_summary=dict(fold=fold,fit_records=len(train),calibration_records=len(cal),heldout_records=len(test),arms={})
        for arm in arms:
            if arm=='raw_recalibrated':apply=lambda r:r['scores20']
            else:
                apply,state=fit_reviewer(train,threshold,arm=='context_review');models.append(dict(fold=fold,arm=arm,**state))
            cal_scores=[apply(r) for r in cal];test_scores=[apply(r) for r in test]
            choice=metrics20.choose_threshold(score_records(cal,cal_scores),.05)
            cut=choice['threshold']
            for i,s in zip(test_idx,test_scores):margins[arm][i]=s-cut
            test_result=metrics20.evaluate(score_records(test,test_scores),cut)
            fold_summary['arms'][arm]=dict(threshold_choice=choice,heldout=compact(test_result))
        fold_results.append(fold_summary)
        print(f'Completed reviewer fold {fold+1}/5',flush=True)
    outputs={'original_fixed':original,'lower_threshold_only':metrics20.evaluate(records,threshold-.5)}
    for arm in arms:outputs[arm]=metrics20.evaluate(score_records(records,margins[arm]),0.)
    candidates=dict(frames=0,seconds=0.,true_fake_seconds=0.,true_real_seconds=0.,contiguous_intervals=0)
    transitions={arm:dict(recovered_fn_seconds=0.,added_fp_seconds=0.,removed_fp_seconds=0.,lost_tp_seconds=0.) for arm in arms}
    for i,r in enumerate(records):
        mask=candidate_mask(r['scores20'],threshold);y=r['labels20'].astype(bool);w=r['widths'];old=r['scores20']>=threshold
        candidates['frames']+=int(mask.sum());candidates['seconds']+=float(w[mask].sum())
        candidates['true_fake_seconds']+=float(w[mask&y].sum());candidates['true_real_seconds']+=float(w[mask&~y].sum())
        candidates['contiguous_intervals']+=int(np.sum(np.diff(np.r_[False,mask,False].astype(int))==1))
        for arm in arms:
            new=margins[arm][i]>=0;changed=transitions[arm]
            changed['recovered_fn_seconds']+=float(w[new&~old&y].sum())
            changed['added_fp_seconds']+=float(w[new&~old&~y].sum())
            changed['removed_fp_seconds']+=float(w[~new&old&~y].sum())
            changed['lost_tp_seconds']+=float(w[~new&old&y].sum())
    comparisons={f'context_minus_{arm}':paired_delta(outputs['context_review'],outputs[arm],fold_ids,args.seed) for arm in ['raw_recalibrated','score_only_review']}
    summary=dict(status='complete',protocol_sha256=sha(args.output/'protocol.json'),recordings=n,
                 candidates=candidates,results={arm:compact(result) for arm,result in outputs.items()},
                 comparisons=comparisons,changes_vs_original_fixed=transitions,limitations=protocol['limitations'])
    write_json(args.output/'summary.json',summary);write_json(args.output/'fold_results.json',fold_results)
    write_json(args.output/'fitted_reviewers.json',models)
    write_json(args.output/'per_recording.json',{arm:r['per_recording'] for arm,r in outputs.items()})
    write_json(args.output/'fold_assignments.json',[dict(uid=r['uid'],fold=int(fold_ids[i])) for i,r in enumerate(records)])
    np.savez_compressed(args.output/'oof_margins.npz',**{arm:np.concatenate(scores) for arm,scores in margins.items()},
                        uids=np.array([r['uid'] for r in records]),lengths=np.array([len(r['labels20']) for r in records]),
                        original=np.concatenate([r['scores20']-threshold for r in records]),
                        labels=np.concatenate([r['labels20'] for r in records]),widths=np.concatenate([r['widths'] for r in records]))
    print(json.dumps(dict(candidates=candidates,results={arm:{k:r[k] for k in ['event_f1_iou05_percent','event_precision_percent','event_recall_percent','real_fpr_percent','fn_seconds','fp_seconds']} for arm,r in outputs.items()},comparisons=comparisons,changes=transitions),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
