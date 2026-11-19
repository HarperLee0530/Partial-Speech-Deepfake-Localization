"""Re-read near-threshold real audio with two crop contexts and frozen models.

Add-only review preserves all noncandidate decisions. Compare reviewers at an
identical additional genuine-duration false-alarm budget (0.25 percentage point
on each calibration fold). This is not a replacement for the main <=5% protocol.
"""
from __future__ import annotations
import argparse
import math
import time
import warnings
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from scipy.special import expit
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from . import metrics20
from .backbone import CNNBiGRU,ModelConfig
from .predict_main import load_main_checkpoint,waveform_batch
from .train_dual import forward_batch
from .pos_probe import read_json,write_json,sha
from .second_review_probe import context_features,paired_delta,compact

STEP=.02
BAND=.5
CONTEXTS=(.32,.64)
AUDIO_FEATURES=('main_crop_032','clean_crop_032','main_crop_064','clean_crop_064')


def candidates(scores,threshold):
    s=np.asarray(scores,dtype=np.float64)
    return (s>=threshold-BAND)&(s<threshold)


def plan_crops(scores,threshold,samples):
    """Use only first-pass scores and audio length; preserve global 160ms phase."""
    mask=candidates(scores,threshold)
    changes=np.diff(np.r_[False,mask,False].astype(np.int8))
    requests=[]
    for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
        for mode,context in enumerate(CONTEXTS):
            pad=round(context/STEP)
            start=max(0,((int(a)-pad)//8)*8)
            end=min(samples,math.ceil((int(b)+pad)/8)*8*320)
            if end<=start*320:raise ValueError('Invalid crop')
            requests.append(dict(start_sample=start*320,end_sample=end,start_frame=start,
                                 target_start=int(a),target_end=int(b),mode=mode))
    return requests


def select_cut(risk,negative_seconds,budget):
    """Largest candidate set respecting whole score ties and a real-time budget."""
    risk=np.asarray(risk,dtype=np.float64);w=np.asarray(negative_seconds,dtype=np.float64)
    if len(risk)!=len(w) or np.any(w<0) or budget<0 or not np.isfinite(risk).all():
        raise ValueError('Invalid calibration inputs')
    if not len(risk):return dict(threshold=1e30,budget_seconds=float(budget),selected_real_seconds=0.)
    neg=w>0
    if not neg.any():
        cut=float(np.nextafter(risk.max(),np.inf))
    else:
        order=np.argsort(-risk[neg],kind='stable');s=risk[neg][order];v=w[neg][order]
        starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
        totals=np.add.reduceat(v,starts);scores=s[starts];cumulative=np.cumsum(totals)
        over=np.flatnonzero(cumulative>budget+1e-10)
        cut=float(np.nextafter(scores[over[0]],np.inf)) if len(over) else float(np.nextafter(risk.min(),-np.inf))
    used=float(w[risk>=cut].sum())
    if used>budget+1e-7:raise AssertionError('Calibration exceeded added-FP budget')
    return dict(threshold=cut,budget_seconds=float(budget),selected_real_seconds=used)


def load_records(args):
    manifest=read_json(args.manifest)
    if manifest['split']!='dev':raise ValueError('Only dev permitted in this exploratory probe')
    mapping={r['uid']:r['audio'] for r in manifest['records']}
    with np.load(args.scores,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(args.labels,allow_pickle=False) as z:labels={k:z[k] for k in z.files}
    uids=list(map(str,data['uids']));offsets=np.r_[0,np.cumsum(data['lengths']*8)]
    if set(uids)!=set(mapping) or set(uids)!=set(labels) or offsets[-1]!=len(data['scores20']):
        raise ValueError('Input schemas or UID sets differ')
    records=[]
    for i,uid in enumerate(uids):
        y=labels[uid];d=float(data['durations'][i]);start=np.arange(len(y))*STEP
        w=np.maximum(0,np.minimum(start+STEP,d)-start);active=w>0
        if not np.isin(y,[0,1]).all() or not np.isclose(w.sum(),d,atol=1e-8,rtol=0):raise ValueError('Invalid labels/duration')
        s=np.asarray(data['scores20'][offsets[i]:offsets[i+1]][:len(y)][active],dtype=np.float64)
        records.append(dict(uid=uid,audio=mapping[uid],labels20=y[active],scores20=s,
                            duration=d,widths=w[active],context=context_features(s,w[active])))
    return records


def make_batch(items):
    lengths=[int(item['lengths'][0]) for item in items];maximum=max(lengths)
    return dict(features=torch.stack([F.pad(item['features'][0],(0,0,0,maximum-n)) for item,n in zip(items,lengths)]),
                lengths=torch.tensor(lengths),fine_weights=torch.stack([F.pad(item['fine_weights'][0],(0,maximum-n)) for item,n in zip(items,lengths)]))


@torch.inference_mode()
def extract_audio(records,args,threshold):
    model,adapter,frontend,_,loaded_cut=load_main_checkpoint(args.checkpoint,args.device)
    if loaded_cut!=threshold:raise ValueError('Checkpoint/score threshold differs')
    payload=torch.load(args.review_checkpoint,map_location='cpu',weights_only=False)
    if payload['protocol']['arm']!='cnn_clean' or payload['adapter_options'] is not None:
        raise ValueError('Independent reviewer must be the frozen cnn_clean arm')
    clean=CNNBiGRU(ModelConfig(**payload['model_config']))
    clean.load_state_dict(payload['backbone_state_dict'],strict=True);clean.to(args.device).eval().requires_grad_(False)
    output=[];crop_rows=[];wave_hashes={};full_checks=[];total_crops=0;started=time.monotonic()
    for i,r in enumerate(records):
        result=np.zeros((len(r['scores20']),4),dtype=np.float64)
        coverage=np.zeros((len(r['scores20']),2),dtype=np.int8)
        mask=candidates(r['scores20'],threshold)
        if not mask.any():output.append(result);continue
        wave,sr=sf.read(r['audio'],dtype='float32')
        if sr!=16000 or wave.ndim!=1 or abs(len(wave)/sr-r['duration'])>1e-8:
            raise ValueError('Audio is not aligned with first pass')
        wave_hashes[r['uid']]=dict(path=r['audio'],sha256=sha(r['audio']),samples=len(wave))
        # Reproduce full-waveform first-pass scores before interpreting recrops.
        if len(full_checks)<3:
            batch,_,actual=waveform_batch(wave,sr,frontend)
            logits,_=forward_batch(model,adapter,batch,args.device)
            fresh=logits[0,:actual].cpu().numpy();old=r['scores20']
            error=float(np.max(np.abs(fresh-old)))
            if not np.allclose(fresh,old,atol=2e-4,rtol=2e-5):raise ValueError('Full waveform reference check failed')
            full_checks.append(dict(uid=r['uid'],max_absolute_logit_difference=error))
        plans=plan_crops(r['scores20'],threshold,len(wave))
        unique=list(dict.fromkeys((p['start_sample'],p['end_sample']) for p in plans))
        crop_outputs={}
        for left in range(0,len(unique),16):
            keys=unique[left:left+16];items=[waveform_batch(wave[a:b],sr,frontend)[0] for a,b in keys]
            batch=make_batch(items)
            main_scores,_=forward_batch(model,adapter,batch,args.device)
            clean_scores=clean(batch['features'].to(args.device),batch['lengths'])
            for j,(key,length) in enumerate(zip(keys,batch['lengths'].tolist())):
                crop_outputs[key]=(main_scores[j,:length].cpu().numpy().copy(),clean_scores[j,:length].cpu().numpy().copy())
        for p in plans:
            a,b=p['target_start'],p['target_end'];mode=p['mode'];offset=p['start_frame']
            main_score,clean_score=crop_outputs[p['start_sample'],p['end_sample']]
            result[a:b,mode*2]=main_score[a-offset:b-offset]
            result[a:b,mode*2+1]=clean_score[a-offset:b-offset]
            coverage[a:b,mode]+=1
            crop_rows.append(dict(uid=r['uid'],**p,context_seconds=CONTEXTS[mode],
                                  full_recording_crop=p['start_sample']==0 and p['end_sample']==len(wave)))
        if not np.all(coverage[mask]==1) or not np.all(coverage[~mask]==0):raise AssertionError('Crop-to-frame mapping failed')
        if not np.isfinite(result).all():raise ValueError('Nonfinite recrop scores')
        total_crops+=len(unique);output.append(result)
        if (i+1)%100==0:
            progress=dict(stage='waveform_review',completed_recordings=i+1,total_recordings=len(records),
                          unique_crops=total_crops,elapsed_seconds=time.monotonic()-started)
            write_json(args.output/'progress.json',progress);print(progress,flush=True)
    np.savez_compressed(args.output/'audio_features.npz',features=np.concatenate(output),
                        uids=np.array([r['uid'] for r in records]),lengths=np.array([len(r['scores20']) for r in records]))
    write_json(args.output/'crop_requests.json',crop_rows);write_json(args.output/'waveform_hashes.json',wave_hashes)
    info=dict(unique_crops=total_crops,crop_requests=len(crop_rows),
              full_recording_requests=sum(r['full_recording_crop'] for r in crop_rows),
              full_waveform_reference_checks=full_checks,elapsed_seconds=time.monotonic()-started,
              feature_sha256=sha(args.output/'audio_features.npz'))
    write_json(args.output/'extraction.json',info)
    return output,info


def fit_reviewer(records,audio,indices,threshold,use_audio):
    xx=[];yy=[];ww=[]
    for i in indices:
        r=records[i];mask=candidates(r['scores20'],threshold)
        x=r['context'] if not use_audio else np.column_stack([r['context'],audio[i]])
        xx.append(x[mask]);yy.append(r['labels20'][mask]);ww.append(r['widths'][mask])
    x=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww)
    if len(np.unique(y))!=2:raise ValueError('Both candidate classes required')
    scaler=StandardScaler().fit(x,sample_weight=w)
    model=LogisticRegression(C=1.,max_iter=1200,tol=1e-7,solver='lbfgs')
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(scaler.transform(x),y,sample_weight=w/STEP)
    def predict(i):
        x=records[i]['context'] if not use_audio else np.column_stack([records[i]['context'],audio[i]])
        return model.decision_function(scaler.transform(x))
    state=dict(use_audio=use_audio,fit_candidate_frames=len(y),
               mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
               coefficient=model.coef_[0].tolist(),intercept=float(model.intercept_[0]),iterations=int(model.n_iter_[0]))
    return predict,state


def evaluate_masks(records,masks):
    result=metrics20.evaluate([dict(uid=r['uid'],duration=r['duration'],labels20=r['labels20'],scores20=p.astype(float)) for r,p in zip(records,masks)],.5)
    tp=fp=fn=0
    for r,p in zip(records,masks):
        y=r['labels20'].astype(bool);tp+=int((p&y).sum());fp+=int((p&~y).sum());fn+=int((~p&y).sum())
    result['frame_f1_percent']=200*tp/(2*tp+fp+fn)
    result['frame_confusion_tp_fp_fn']=[tp,fp,fn]
    result.pop('threshold',None);result.pop('selection_score_j_percent',None)
    result['evaluation_input']='Final binary decisions; 0.5 was only a mask-decoding constant, not a detector threshold'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['scores','labels','manifest','metrics','checkpoint','review-checkpoint','output']:
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--seed',type=int,default=20261116)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Refusing to overwrite experiment')
    args.output.mkdir(parents=True)
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    old=read_json(args.metrics);threshold=float(old['threshold'])
    protocol=dict(status='declared_before_audio_extraction_and_fit',seed=args.seed,device=args.device,
                  inputs={name:dict(path=str(getattr(args,name)),sha256=sha(getattr(args,name))) for name in ['scores','labels','manifest','metrics','checkpoint','review_checkpoint']},
                  source_hashes={str(Path(__file__).with_name(name)):sha(Path(__file__).with_name(name)) for name in ['audio_review_probe.py','second_review_probe.py','pos_probe.py','backbone.py','predict_main.py','train_dual.py','dual_resolution.py','metrics20.py']},
                  first_pass_threshold_logit=threshold,candidate_logit_band=[threshold-BAND,threshold],upper_exclusive=True,
                  crop_context_seconds=list(CONTEXTS),crop_start_alignment_seconds=.16,crop_selection='First-pass scores and waveform duration only',
                  reviewer_inputs={'raw_add':['original score'],'context_add':['original score neighborhood features'],
                                   'audio_add':['original score neighborhood features',*AUDIO_FEATURES]},
                  models='Current boundary checkpoint and separately trained original-clean CNN checkpoint; both frozen; recompute logMel on raw waveform crops',
                  fusion='Candidate fake-class logistic regression C=1; fit-only weighted standardization; no class rebalancing',
                  validation='Five UID folds: k held out, (k+1)%5 calibration, other three fitting; no parameter search',
                  added_fpr_budget_percentage_points=.25,
                  calibration='Candidate false-positive seconds <=0.0025 * all genuine seconds in calibration fold; identical added budget for all arms',
                  decision='Preserve original positives and all noncandidate decisions; add only candidates exceeding the frozen review cut',
                  primary='audio_add minus raw_add',secondary='audio_add minus context_add',
                  limitations=['Exploratory dev-only; the detector and its first-pass threshold were already selected on this dev set.',
                               'This probes add-only review with an added 0.25pp calibration false-alarm budget, not the main <=5% total-FPR protocol.',
                               'Held-out added FPR need not equal the calibration budget.',
                               'Review models reuse PS training data; they are not statistically independent and crop inference changes input distribution.',
                               'UID grouping does not exclude shared speakers/sources. Bootstrap conditions on fixed predictions.',
                               'No test set read, no CNN training, no replacement of main model.'])
    write_json(args.output/'protocol.json',protocol)
    records=load_records(args);n=len(records)
    original=[r['scores20']>=threshold for r in records]
    baseline=evaluate_masks(records,original)
    for key in ['event_f1_iou05_percent','real_fpr_percent','fn_seconds','fp_seconds']:
        if not np.isclose(baseline[key],old[key],atol=1e-7,rtol=0):raise ValueError('Baseline mismatch')
    print(f'Loaded {n} recordings; extracting actual waveform crops',flush=True)
    audio,extraction=extract_audio(records,args,threshold)
    folds=np.zeros(n,dtype=int)
    for fold,(_,test) in enumerate(KFold(5,shuffle=True,random_state=args.seed).split(records)):folds[test]=fold
    arms=['raw_add','context_add','audio_add'];masks={arm:[None]*n for arm in arms};risks={arm:[None]*n for arm in arms}
    fitted=[];fold_results=[]
    for fold in range(5):
        test=np.flatnonzero(folds==fold);cal=np.flatnonzero(folds==(fold+1)%5);train=np.flatnonzero((folds!=fold)&(folds!=(fold+1)%5))
        assert not(set(train)&set(cal) or set(train)&set(test) or set(cal)&set(test))
        real_seconds=sum(float(r['widths'][r['labels20']==0].sum()) for r in [records[i] for i in cal])
        fresult=dict(fold=fold,fit_recordings=len(train),calibration_recordings=len(cal),heldout_recordings=len(test),arms={})
        for arm in arms:
            if arm=='raw_add':predict=lambda i:records[i]['scores20']
            else:
                predict,state=fit_reviewer(records,audio,train,threshold,arm=='audio_add');fitted.append(dict(fold=fold,arm=arm,**state))
            cs=[];cw=[]
            for i in cal:
                r=records[i];selected=candidates(r['scores20'],threshold)
                cs.append(predict(i)[selected]);cw.append(r['widths'][selected]*(r['labels20'][selected]==0))
            choice=select_cut(np.concatenate(cs),np.concatenate(cw),real_seconds*.0025)
            for i in test:
                r=records[i];risk=predict(i);risks[arm][i]=risk
                masks[arm][i]=original[i]|(candidates(r['scores20'],threshold)&(risk>=choice['threshold']))
                assert np.all(masks[arm][i][original[i]])
            fresult['arms'][arm]=choice
        fold_results.append(fresult);print(f'Completed review fit/calibration fold {fold+1}/5',flush=True)
    results={'original':baseline}
    for arm in arms:results[arm]=evaluate_masks(records,masks[arm])
    changes={}
    for arm in arms:
        recovered=added=0.;frames=0
        for i,r in enumerate(records):
            new=masks[arm][i]&~original[i];y=r['labels20'].astype(bool)
            recovered+=float(r['widths'][new&y].sum());added+=float(r['widths'][new&~y].sum());frames+=int(new.sum())
        changes[arm]=dict(recovered_fn_seconds=recovered,added_fp_seconds=added,changed_frames=frames)
    comparison={f'audio_minus_{arm}':paired_delta(results['audio_add'],results[arm],folds,args.seed) for arm in ['raw_add','context_add']}
    summary=dict(status='complete',protocol_sha256=sha(args.output/'protocol.json'),recordings=n,extraction=extraction,
                 results={k:compact(v) for k,v in results.items()},changes=changes,comparisons=comparison,
                 limitations=protocol['limitations'])
    write_json(args.output/'summary.json',summary);write_json(args.output/'fold_results.json',fold_results)
    write_json(args.output/'fitted_reviewers.json',fitted)
    write_json(args.output/'per_recording.json',{k:v['per_recording'] for k,v in results.items()})
    np.savez_compressed(args.output/'oof_review.npz',**{arm:np.concatenate(masks[arm]) for arm in arms},
                        **{arm+'_risk':np.concatenate(risks[arm]) for arm in arms},
                        original=np.concatenate(original),labels=np.concatenate([r['labels20'] for r in records]),
                        scores=np.concatenate([r['scores20'] for r in records]),widths=np.concatenate([r['widths'] for r in records]),
                        uids=np.array([r['uid'] for r in records]),lengths=np.array([len(r['scores20']) for r in records]),folds=folds)
    write_json(args.output/'progress.json',dict(status='complete'))
    print(__import__('json').dumps(dict(results={k:{m:v[m] for m in ['event_f1_iou05_percent','frame_f1_percent','real_fpr_percent','duration_recall_percent']} for k,v in results.items()},changes=changes,comparisons=comparison),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
