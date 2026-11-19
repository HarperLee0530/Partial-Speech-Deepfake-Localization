"""Dev-only, recording-held-out POS error-ranking probe over a frozen detector.

This does not change the detector's scores, threshold, or predictions. The same
dev set previously selected the detector, so OOF here is exploratory, not an
independent estimate of the whole modelling pipeline's generalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import sklearn
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

STEP = .02
POS = ('ADJ','ADP','ADV','AUX','CCONJ','DET','INTJ','NOUN','NUM','PART',
       'PRON','PROPN','PUNCT','SCONJ','SYM','VERB','X')
ACOUSTIC = ['signed_margin','signed_log_margin','fake_probability','entropy']
QUALITY = ['word_coverage','mean_log_word_seconds','mean_asr_confidence',
           'missing_confidence_coverage','overlap_excess']
ARMS = ('acoustic', 'control', 'control_pos')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_words(words, widths):
    """Exact overlap integration; overlapping words equally share each instant.

    No labels enter this function. At a word boundary, POS features are a soft
    mixture. A mixed-authenticity word retains unchanged POS across its frames.
    Returns POS fractions and quality features per frame, plus valid words.
    """
    widths = np.asarray(widths, dtype=np.float64)
    n = len(widths)
    duration = float(widths.sum())
    frame_words = [[] for _ in range(n)]
    valid_words = []
    for original in words:
        start, end = float(original['start']), float(original['end'])
        if not np.isfinite([start, end]).all():
            raise ValueError('Nonfinite word timestamp')
        a, b = max(0., start), min(duration, end)
        if b <= a:
            continue
        conf = original.get('confidence')
        missing = conf is None or not np.isfinite(float(conf))
        if not missing and not 0 <= float(conf) <= 1:
            raise ValueError('ASR confidence outside [0,1]')
        tag = original.get('pos', 'X')
        tag = tag if tag in POS else 'X'
        w = dict(start=a, end=b, pos=tag, text=original.get('text',''),
                 confidence=0. if missing else float(conf), missing=float(missing),
                 log_duration=float(np.log(max(b-a,1e-6))))
        wi = len(valid_words)
        valid_words.append(w)
        lo, hi = max(0,int(np.floor(a/STEP))), min(n,int(np.ceil(b/STEP)))
        for t in range(lo,hi):
            if min(b,t*STEP+widths[t])-max(a,t*STEP)>1e-12:
                frame_words[t].append(wi)
    pos = np.zeros((n,len(POS)), dtype=np.float64)
    quality = np.zeros((n,len(QUALITY)), dtype=np.float64)
    for t, ids in enumerate(frame_words):
        if not ids:
            continue
        a, b = t*STEP, t*STEP+widths[t]
        cuts = sorted(set([a,b]+[max(a,min(b,valid_words[i][k])) for i in ids for k in ['start','end']]))
        for left,right in zip(cuts[:-1],cuts[1:]):
            mid=(left+right)/2
            active=[valid_words[i] for i in ids if valid_words[i]['start']<=mid<valid_words[i]['end']]
            if not active:
                continue
            frac=(right-left)/widths[t]
            quality[t,0]+=frac
            quality[t,4]+=frac*max(0,len(active)-1)
            for w in active:
                part=frac/len(active)
                pos[t,POS.index(w['pos'])]+=part
                quality[t,1]+=part*w['log_duration']
                quality[t,2]+=part*w['confidence']
                quality[t,3]+=part*w['missing']
        # Conditional means on covered time; uncovered portions remain explicit.
        if quality[t,0]>0:
            quality[t,1:3]/=quality[t,0]
    if not np.allclose(pos.sum(axis=1),quality[:,0],atol=1e-10,rtol=0):
        raise AssertionError('POS/coverage conservation failed')
    if np.any(quality[:,0]>1+1e-9):
        raise AssertionError('Overlap double-counted')
    return pos,quality,valid_words


def word_fake_fraction(word, labels, widths):
    starts=np.arange(len(widths))*STEP
    overlap=np.maximum(0,np.minimum(starts+widths,word['end'])-np.maximum(starts,word['start']))
    fake=float(np.dot(overlap,labels))
    real=float(np.dot(overlap,1-labels))
    state='mixed' if min(fake,real)>1e-9 else ('fake' if fake>1e-9 else 'real')
    return fake,real,state


def acoustic_features(scores,threshold):
    margin=np.clip(np.asarray(scores,dtype=np.float64)-threshold,-30,30)
    probability=expit(np.clip(scores,-30,30))
    p=np.clip(probability,1e-12,1-1e-12)
    entropy=-(p*np.log(p)+(1-p)*np.log1p(-p))
    return np.column_stack([margin,np.sign(margin)*np.log1p(np.abs(margin)),probability,entropy])


def budget_weights(risk, widths, fraction):
    """Fixed duration budget; prorate only the final 20ms frame for exact cost.

    This is a continuous-time review-ranking diagnostic, not an event editor.
    Input order is deterministic UID/frame order; ties use that order.
    """
    if not 0 < fraction <= 1:
        raise ValueError('Budget fraction must be in (0,1]')
    widths=np.asarray(widths,dtype=np.float64)
    order=np.argsort(-np.asarray(risk),kind='stable')
    budget=float(widths.sum()*fraction)
    before=np.r_[0.,np.cumsum(widths[order])[:-1]]
    selected=np.minimum(widths[order],np.maximum(0.,budget-before))
    result=np.zeros_like(widths);result[order]=selected
    if not np.isclose(result.sum(),budget,atol=1e-7,rtol=0):
        raise AssertionError('Budget not conserved')
    return result


def fit_predict(x,y,w,train,test):
    scaler=StandardScaler().fit(x[train],sample_weight=w[train])
    model=LogisticRegression(C=1.,max_iter=1200,solver='lbfgs',tol=1e-7)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(scaler.transform(x[train]),y[train],sample_weight=w[train]/STEP)
    scores=model.predict_proba(scaler.transform(x[test]))[:,1]
    state=dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
               coefficient=model.coef_[0].tolist(),intercept=float(model.intercept_[0]),
               iterations=int(model.n_iter_[0]))
    return scores,state


def build(args):
    archive=np.load(args.scores,allow_pickle=False)
    truth=np.load(args.labels,allow_pickle=False)
    annotations={}
    for line in Path(args.words).read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            r=json.loads(line)
            if r['uid'] in annotations: raise ValueError('Duplicate annotation UID')
            annotations[r['uid']]=r
    manifest=read_json(args.manifest)
    if manifest['split']!='dev': raise ValueError('This probe only permits dev')
    uids=list(map(str,archive['uids']))
    if len(set(uids))!=len(uids):raise ValueError('Duplicate score UID')
    if set(uids)!=set(truth.files) or set(uids)!=set(annotations) or set(uids)!={r['uid'] for r in manifest['records']}:
        raise ValueError('UID sets do not match')
    metrics=read_json(args.metrics)
    threshold=float(metrics['threshold'])
    lengths=archive['lengths']*8
    if int(lengths.sum())!=len(archive['scores20']):raise ValueError('Unexpected score storage layout')
    offsets=np.r_[0,np.cumsum(lengths)]
    folds=np.zeros(len(uids),dtype=np.int64)
    for fi,(_,held) in enumerate(KFold(5,shuffle=True,random_state=args.seed).split(uids)):
        folds[held]=fi
    chunks=[];word_rows=[];record_rows=[]
    for ri,uid in enumerate(uids):
        y=np.asarray(truth[uid],dtype=np.int8)
        d=float(archive['durations'][ri])
        if len(y)!=int(archive['lengths20'][ri]) or not np.isin(y,[0,1]).all():raise ValueError('Invalid labels')
        if abs(float(annotations[uid]['audio_seconds'])-d)>1e-6:raise ValueError('ASR/audio duration mismatch')
        widths=np.maximum(0,np.minimum((np.arange(len(y))+1)*STEP,d)-np.arange(len(y))*STEP)
        if not np.isclose(widths.sum(),d,atol=1e-8,rtol=0):raise ValueError('Incomplete native labels')
        active=widths>0
        widths=widths[active];y=y[active]
        scores=np.asarray(archive['scores20'][offsets[ri]:offsets[ri+1]][:len(active)][active],dtype=np.float64)
        if not np.isfinite(scores).all():raise ValueError('Invalid score')
        pos,quality,words=project_words(annotations[uid]['words'],widths)
        mixed_overlap=np.zeros(len(y),dtype=bool)
        starts=np.arange(len(y))*STEP
        for word in words:
            fake,real,state=word_fake_fraction(word,y,widths)
            word_rows.append(dict(uid=uid,fold=int(folds[ri]),**word,fake_seconds=fake,real_seconds=real,state=state))
            if state=='mixed':mixed_overlap|=(starts<word['end'])&(starts+widths>word['start'])
        record_rows.append(dict(uid=uid,fold=int(folds[ri]),duration=d,word_count=len(words)))
        chunks.append(dict(scores=scores,labels=y,widths=widths,pos=pos,quality=quality,
                           acoustic=acoustic_features(scores,threshold),mixed=mixed_overlap,
                           rid=np.full(len(y),ri),fold=np.full(len(y),folds[ri])))
    arrays={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    pred=arrays['scores']>=threshold
    arrays['pred']=pred;arrays['error']=(pred!=arrays['labels']).astype(np.int8)
    fn=float(arrays['widths'][(~pred)&(arrays['labels']==1)].sum())
    fp=float(arrays['widths'][pred&(arrays['labels']==0)].sum())
    if not np.isclose(fn,metrics['fn_seconds'],atol=1e-7,rtol=0) or not np.isclose(fp,metrics['fp_seconds'],atol=1e-7,rtol=0):
        raise ValueError('Frozen detector error totals not reproduced')
    return arrays,record_rows,word_rows,threshold,dict(fn_seconds=fn,fp_seconds=fp)


def descriptive_stats(a,words):
    stats=[]
    for tag in (*POS,'UNCOVERED'):
        share=1-a['quality'][:,0] if tag=='UNCOVERED' else a['pos'][:,POS.index(tag)]
        exposure=a['widths']*np.clip(share,0,1)
        fake=float(exposure[a['labels']==1].sum());real=float(exposure[a['labels']==0].sum())
        fn=float(exposure[(a['labels']==1)&~a['pred']].sum())
        fp=float(exposure[(a['labels']==0)&a['pred']].sum())
        ws=[w for w in words if w['pos']==tag]
        stats.append(dict(pos=tag,recordings=int(len(np.unique(a['rid'][exposure>1e-10]))),
                          words=len(ws),mixed_words=sum(w['state']=='mixed' for w in ws),
                          exposure_seconds=float(exposure.sum()),fake_seconds=fake,real_seconds=real,
                          fn_seconds=fn,fp_seconds=fp,fnr_percent=100*fn/fake if fake else None,
                          fpr_percent=100*fp/real if real else None))
    assert np.isclose(sum(s['exposure_seconds'] for s in stats),a['widths'].sum(),atol=1e-7,rtol=0)
    return stats


def evaluate_ranking(a,risk,records,seed):
    results={};comparisons={};n=len(records);folds=[]
    for side in ['fn','fp']:
        side_mask=~a['pred'] if side=='fn' else a['pred']
        error_by_rec=np.bincount(a['rid'],weights=a['widths']*a['error']*side_mask,minlength=n)
        side_result={};selected_cache={}
        for arm in ['raw_margin',*ARMS]:
            review=np.zeros(len(a['widths']))
            for fi in range(5):
                idx=np.flatnonzero(side_mask&(a['fold']==fi))
                review[idx]=budget_weights(risk[arm][idx],a['widths'][idx],.05)
            captured=np.bincount(a['rid'],weights=review*a['error'],minlength=n)
            cost=np.bincount(a['rid'],weights=review,minlength=n)
            selected_cache[arm]=captured
            side_result[arm]=dict(review_seconds=float(cost.sum()),captured_error_seconds=float(captured.sum()),
                                  error_seconds=float(error_by_rec.sum()),error_coverage_percent=float(100*captured.sum()/error_by_rec.sum()),
                                  review_precision_percent=float(100*captured.sum()/cost.sum()),
                                  correct_seconds_reviewed=float((cost-captured).sum()))
            for ri in range(n):
                records[ri][f'{side}_{arm}_captured_seconds']=float(captured[ri])
                records[ri][f'{side}_{arm}_review_seconds']=float(cost[ri])
            for subset,mask in [('mixed_word',a['mixed']),('uncovered',a['quality'][:,0]<1e-9)]:
                denom=float(np.sum(a['widths']*a['error']*side_mask*mask))
                side_result[arm][subset]=dict(error_seconds=denom,captured_error_seconds=float(np.sum(review*a['error']*mask)))
        delta=selected_cache['control_pos']-selected_cache['control']
        rng=np.random.default_rng(seed);boot=[]
        # Fixed OOF selections; stratify bootstrap by held-out fold. No refit/rerank.
        groups=[np.flatnonzero(np.array([r['fold'] for r in records])==fi) for fi in range(5)]
        for _ in range(2000):
            indices=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
            denom=error_by_rec[indices].sum()
            boot.append(100*delta[indices].sum()/denom)
        comparisons[side]=dict(delta_captured_seconds=float(delta.sum()),
                              delta_coverage_pp=float(100*delta.sum()/error_by_rec.sum()),
                              recording_bootstrap_95_ci_pp=np.quantile(boot,[.025,.975]).tolist(),
                              improved_recordings=int((delta>1e-9).sum()),worsened_recordings=int((delta< -1e-9).sum()))
        for fi,g in enumerate(groups):
            folds.append(dict(side=side,fold=fi,delta_captured_seconds=float(delta[g].sum()),
                              delta_coverage_pp=float(100*delta[g].sum()/error_by_rec[g].sum())))
        results[side]=side_result
    return results,comparisons,folds


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['scores','labels','words','manifest','metrics','checkpoint','output']:
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--seed',type=int,default=20261116)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Refusing to overwrite existing probe')
    args.output.mkdir(parents=True)
    inputs={k:dict(path=str(getattr(args,k)),sha256=sha(getattr(args,k))) for k in ['scores','labels','words','manifest','metrics','checkpoint']}
    protocol=dict(status='declared_before_fitting',inputs=inputs,script_sha256=sha(__file__),seed=args.seed,
                  folds=5,split='dev_only',primary_comparison='control_pos minus control',primary_side='fn',
                  secondary_side='fp',review_budget_percent=5,selection='Within held-out fold and predicted side; identical duration budget; final frame prorated',
                  arms={'raw_margin':'FN raw fake logit; FP negative fake logit','acoustic':ACOUSTIC,
                        'control':ACOUSTIC+QUALITY,'control_pos':ACOUSTIC+QUALITY+list(POS)},
                  model='Separate FN/FP logistic regressions; C=1; train-fold weighted StandardScaler; duration/.02 fit weights; no class balancing or hyperparameter search',
                  uncovered_policy='POS model falls back to control risk wherever word coverage is zero',
                  labels='Only error targets and descriptive statistics use authenticity labels; feature projection does not',
                  versions={'numpy':np.__version__,'sklearn':sklearn.__version__},
                  limitations=['Detector and threshold were already selected on this dev set: OOF only holds out error-ranker training, not detector selection.',
                               'UID grouping cannot exclude shared speaker/source dependence.',
                               'No test-set input or new CNN inference; no actual error correction.',
                               'POS and ASR timestamps are automatic; descriptive association is not causation.',
                               'Bootstrap resamples fixed OOF selections, not training/fold uncertainty; secondary analyses are exploratory.'])
    write_json(args.output/'protocol.json',protocol)
    a,records,words,threshold,baseline=build(args)
    print(f'Loaded {len(records)} recordings, {len(a["labels"])} frames, {len(words)} words; baseline {baseline}',flush=True)
    matrices={'acoustic':a['acoustic'],'control':np.column_stack([a['acoustic'],a['quality']]),
              'control_pos':np.column_stack([a['acoustic'],a['quality'],a['pos']])}
    risks={arm:np.full(len(a['labels']),np.nan) for arm in ARMS}
    risks['raw_margin']=np.where(a['pred'],-a['scores'],a['scores'])
    models=[]
    for fi in range(5):
        for side in ['fn','fp']:
            side_mask=~a['pred'] if side=='fn' else a['pred']
            train=np.flatnonzero(side_mask&(a['fold']!=fi));test=np.flatnonzero(side_mask&(a['fold']==fi))
            assert not set(a['rid'][train])&set(a['rid'][test])
            for arm in ARMS:
                prediction,state=fit_predict(matrices[arm],a['error'],a['widths'],train,test)
                if arm=='control_pos':
                    missing=a['quality'][test,0]<1e-9
                    prediction[missing]=risks['control'][test[missing]]
                risks[arm][test]=prediction
                models.append(dict(fold=fi,side=side,arm=arm,**state))
        print(f'Completed held-out fold {fi+1}/5',flush=True)
    assert all(np.isfinite(r).all() for r in risks.values())
    stats=descriptive_stats(a,words)
    results,comparison,folds=evaluate_ranking(a,risks,records,args.seed)
    summary=dict(status='complete',protocol_sha256=sha(args.output/'protocol.json'),threshold=threshold,
                 recordings=len(records),frames=len(a['labels']),words=len(words),baseline=baseline,
                 word_states={state:sum(w['state']==state for w in words) for state in ['real','fake','mixed']},
                 word_coverage_percent=float(100*np.dot(a['widths'],a['quality'][:,0])/a['widths'].sum()),
                 results=results,primary_comparisons=comparison,fold_comparisons=folds,
                 limitations=protocol['limitations'])
    write_json(args.output/'summary.json',summary)
    write_json(args.output/'pos_statistics.json',stats)
    write_json(args.output/'fitted_rankers.json',models)
    for name,rows in [('per_recording.jsonl',records),('word_diagnostics.jsonl',words)]:
        with (args.output/name).open('w',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    np.savez_compressed(args.output/'oof_risks.npz',**risks,rid=a['rid'],fold=a['fold'],widths=a['widths'],error=a['error'],pred=a['pred'])
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
