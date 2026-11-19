"""Predeclared dev-only five-fold experiment for optional boundary correction."""
import csv
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import KFold
from . import metrics20 as metrics
from .local_boundary import LocalBoundaryRefiner,make_candidates,apply_gains,MODEL_OPTIONS,STEP

ROOT=Path('D:/SAL_精简实验')
BASE=ROOT/'result/ps_four_eventf1_20261115_run1/boundary'
OUTPUT=ROOT/'result/ps_local_boundary_20261116_run1'
INPUTS={'scores':BASE/'dev_scores20.npz','labels':ROOT/'data/cnn_ps_inputs_v2/labels20/dev.npz',
        'manifest':ROOT/'data/cnn_ps_inputs_v2/manifests/dev.json','baseline_metrics':BASE/'dev_metrics.json',
        'selection':BASE/'complete.json','checkpoint':BASE/'best.pt'}
MARGINS=(0.,.1,.25,.5,1.,None)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()),encoding='utf-8')
def dump_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def load():
    manifest=read(INPUTS['manifest']);assert manifest['split']=='dev'
    assert all(r['split']=='dev' for r in manifest['records'])
    with np.load(INPUTS['scores'],allow_pickle=False) as z:s={k:z[k] for k in z.files}
    uids=list(map(str,s['uids']));offsets=np.r_[0,np.cumsum(s['lengths']*8)]
    assert len(set(uids))==len(uids) and offsets[-1]==len(s['scores20'])
    raw=[]
    with np.load(INPUTS['labels'],allow_pickle=False) as labels:
        assert set(uids)==set(labels.files)=={r['uid'] for r in manifest['records']}
        for i,uid in enumerate(uids):
            y=labels[uid];assert len(y)==s['lengths20'][i]
            raw.append(dict(uid=uid,labels20=y,scores20=s['scores20'][offsets[i]:offsets[i+1]],duration=float(s['durations'][i])))
    return metrics._prepare(raw)

def evaluate(records,masks,originals):
    standard=metrics.evaluate([dict(uid=r['uid'],duration=r['duration'],labels20=r['labels'],scores20=p.astype(float)) for r,p in zip(records,masks)],.5)
    cm=np.zeros(4,dtype=int); fixed=bad=0.;fixed_n=bad_n=0; rows=[]; match_extra=np.zeros(2,dtype=int)
    for r,p,old,prior in zip(records,masks,originals,standard['per_recording']):
        y=r['labels'];w=r['widths']
        counts=np.array([(p&y).sum(),(p&~y).sum(),(~p&y).sum(),(~p&~y).sum()]);cm+=counts
        repairs=(old!=y)&(p==y);harms=(old==y)&(p!=y)
        f=float(w[repairs].sum());h=float(w[harms].sum());fixed+=f;bad+=h
        fixed_n+=int(repairs.sum());bad_n+=int(harms.sum())
        ps=metrics._intervals(p,r['duration']);ts=metrics._intervals(y,r['duration'])
        extra=[len(metrics.match_events(ps,ts,t)[0]) for t in (.7,.9)];match_extra+=extra
        rows.append(dict(uid=r['uid'],tp=int(counts[0]),fp=int(counts[1]),fn=int(counts[2]),tn=int(counts[3]),
                         event_tp=prior['event_tp'],event_pred=prior['event_pred'],event_truth=prior['event_truth'],
                         fixed_seconds=f,introduced_seconds=h,event_tp07=extra[0],event_tp09=extra[1]))
    standard.pop('per_recording');standard.pop('threshold')
    tp,fp,fn,tn=map(int,cm)
    standard.update(frame_f1_percent=200*tp/(2*tp+fp+fn),frame_confusion_tp_fp_fn_tn=cm.tolist(),
                    fixed_error_seconds=fixed,introduced_error_seconds=bad,net_error_reduction_seconds=fixed-bad,
                    fixed_error_frames=fixed_n,introduced_error_frames=bad_n,
                    event_f1_iou07_percent=200*int(match_extra[0])/(standard['event_counts']['predicted']+standard['event_counts']['truth']),
                    event_f1_iou09_percent=200*int(match_extra[1])/(standard['event_counts']['predicted']+standard['event_counts']['truth']))
    return standard,rows

def bootstrap(base,changed):
    keys=['tp','fp','fn','event_tp','event_pred','event_truth','fixed_seconds','introduced_seconds']
    a=np.array([[r[k] for k in keys] for r in base]);b=np.array([[r[k] for k in keys] for r in changed])
    rng=np.random.default_rng(20261116);values=[]
    for _ in range(2000):
        indices=rng.integers(0,len(a),len(a));aa=a[indices].sum(0);bb=b[indices].sum(0)
        values.append([200*bb[0]/(2*bb[0]+bb[1]+bb[2])-200*aa[0]/(2*aa[0]+aa[1]+aa[2]),
                       200*bb[3]/(bb[4]+bb[5])-200*aa[3]/(aa[4]+aa[5]),bb[6]-bb[7]])
    return {key:np.quantile(np.array(values)[:,j],[.025,.975]).tolist() for j,key in enumerate(['frame_f1_delta_pp','event_f1_delta_pp','net_error_reduction_seconds'])}

def main():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    protocol={'status':'declared_before_fit','model':'Frozen boundary epoch19 seed666',
              'inputs':{k:{'path':str(p),'sha256':sha(p)} for k,p in INPUTS.items()},
              'source_hashes':{str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('local_boundary.py'),Path(metrics.__file__)]},
              'threshold':read(INPUTS['baseline_metrics'])['threshold'],'max_shift_frames':2,'step_seconds':STEP,
              'features':'Original score context +/-160ms, probabilities, observed event length/gaps, changed-cell scores; no waveform or truth features.',
              'target':'Net correct duration gained by each candidate shift, in units of20ms; all predicted boundaries, no IoU filtering.',
              'model_options':MODEL_OPTIONS,'no_move':'Always available with utility zero; preserve audio edges and event count.',
              'split':'Five UID folds shuffled seed20261116; held k, calibration (k+1)%5, fit remaining3. Same folds as diagnostic.',
              'margins':MARGINS,'calibration':'Among candidates with frame F1 and event F1 both >= unmodified calibration baseline and duration FPR <=max(5%,calibration baseline), maximize mean(frameF1,eventF1); tie prefer larger margin. None=identity fallback.',
              'arms':['baseline','fixed_margin_0','fixed_margin_025','calibrated'],
              'primary':'calibrated minus baseline, pooled held-out records; also report fixed margins without selection.',
              'limitations':['Detector and original threshold previously selected on this dev set; exploratory, not independent test.',
                             'UID folds do not guarantee speaker/source separation. Bootstrap resamples frozen fold outputs only.',
                             'Local score evidence only; cannot create/delete/merge events or repair distant misses.',
                             'No test files accessed. No original checkpoint, threshold, or default inference code changed.']}
    write(OUTPUT/'protocol.json',protocol)
    records=load();threshold=protocol['threshold'];originals=[r['scores']>=threshold for r in records]
    candidates=[make_candidates(r['scores'],r['duration'],threshold) for r in records]
    assert all(np.array_equal(c.original,p) for c,p in zip(candidates,originals))
    baseline,base_rows=evaluate(records,originals,originals);saved=read(INPUTS['baseline_metrics'])
    assert abs(baseline['duration_f1_percent']-saved['duration_f1_percent'])<1e-9
    assert baseline['event_counts']==saved['event_counts']
    print('Baseline verified',baseline['frame_f1_percent'],baseline['event_f1_iou05_percent'],flush=True)
    folds=[held for _,held in KFold(5,shuffle=True,random_state=20261116).split(records)]
    write(OUTPUT/'folds.json',[{'fold':k,'uids':[records[i]['uid'] for i in ids]} for k,ids in enumerate(folds)])
    outputs={name:[None]*len(records) for name in protocol['arms'][1:]}; movements=[];fold_results=[];cal_rows=[]
    for k in range(5):
        held=folds[k];cal=folds[(k+1)%5];fit=np.concatenate([folds[j] for j in range(5) if j not in (k,(k+1)%5)])
        assert not(set(held)&set(cal) or set(held)&set(fit) or set(cal)&set(fit))
        model=LocalBoundaryRefiner().fit([candidates[i] for i in fit],[records[i]['labels'] for i in fit])
        joblib.dump(model,OUTPUT/f'fold_{k}_model.joblib')
        cg=model.predict_gains([candidates[i] for i in cal]);hg=model.predict_gains([candidates[i] for i in held])
        cr=[records[i] for i in cal];co=[originals[i] for i in cal]
        cb,_=evaluate(cr,co,co);options=[]
        for margin in MARGINS:
            masks=[apply_gains(candidates[i],g,margin)[0] for i,g in zip(cal,cg)]
            result,_=evaluate(cr,masks,co)
            eligible=(result['frame_f1_percent']>=cb['frame_f1_percent']-1e-10 and
                      result['event_f1_iou05_percent']>=cb['event_f1_iou05_percent']-1e-10 and
                      result['real_fpr_percent']<=max(5.,cb['real_fpr_percent'])+1e-10)
            objective=.5*(result['frame_f1_percent']+result['event_f1_iou05_percent'])
            options.append((eligible,objective,100 if margin is None else margin,margin))
            cal_rows.append(dict(fold=k,margin=margin,eligible=eligible,frame_f1=result['frame_f1_percent'],event_f1=result['event_f1_iou05_percent'],real_fpr=result['real_fpr_percent'],baseline_frame=cb['frame_f1_percent'],baseline_event=cb['event_f1_iou05_percent']))
        chosen=max(option for option in options if option[0])[3]
        for arm,margin in [('fixed_margin_0',0.),('fixed_margin_025',.25),('calibrated',chosen)]:
            masks=[]
            for i,g in zip(held,hg):
                mask,spans=apply_gains(candidates[i],g,margin);outputs[arm][i]=mask;masks.append(mask)
                for event,((a,b),(aa,bb)) in enumerate(zip(candidates[i].spans,spans)):
                    movements.append(dict(arm=arm,fold=k,uid=records[i]['uid'],event=event,start_frame=int(a),end_frame_exclusive=int(b),
                                          start_shift_frames=int(aa-a),end_shift_frames=int(bb-b)))
            result,_=evaluate([records[i] for i in held],masks,[originals[i] for i in held])
            ref,_=evaluate([records[i] for i in held],[originals[i] for i in held],[originals[i] for i in held])
            fold_results.append(dict(arm=arm,fold=k,margin=margin,fit_records=len(fit),cal_records=len(cal),held_records=len(held),
                                     frame_f1=result['frame_f1_percent'],event_f1=result['event_f1_iou05_percent'],
                                     frame_delta=result['frame_f1_percent']-ref['frame_f1_percent'],event_delta=result['event_f1_iou05_percent']-ref['event_f1_iou05_percent'],
                                     model_fit=model.fit_summary))
        write(OUTPUT/f'fold_{k}_selection.json',{'fold':k,'selected_margin':chosen,'fit_summary':model.fit_summary,'fit_uids':[records[i]['uid'] for i in fit],
                                               'calibration_uids':[records[i]['uid'] for i in cal],'held_uids':[records[i]['uid'] for i in held]})
        print('Fold',k,'chosen margin',chosen,'held calibrated',fold_results[-1]['frame_delta'],fold_results[-1]['event_delta'],flush=True)
    results={'baseline':baseline};dump_csv(OUTPUT/'baseline_per_recording.csv',base_rows)
    archive={'uids':np.array([r['uid'] for r in records]),'lengths':np.array([len(r['labels']) for r in records]),
             'baseline':np.concatenate(originals).astype(np.uint8)}
    for arm,masks in outputs.items():
        assert all(mask is not None for mask in masks)
        result,rows=evaluate(records,masks,originals)
        result['delta_frame_f1_pp']=result['frame_f1_percent']-baseline['frame_f1_percent']
        result['delta_event_f1_pp']=result['event_f1_iou05_percent']-baseline['event_f1_iou05_percent']
        result['paired_uid_bootstrap95']=bootstrap(base_rows,rows)
        assert result['event_counts']['predicted']==baseline['event_counts']['predicted']
        assert abs((baseline['fp_seconds']+baseline['fn_seconds'])-(result['fp_seconds']+result['fn_seconds'])-result['net_error_reduction_seconds'])<1e-8
        results[arm]=result;dump_csv(OUTPUT/f'{arm}_per_recording.csv',rows)
        archive[arm]=np.concatenate(masks).astype(np.uint8)
    np.savez_compressed(OUTPUT/'oof_masks.npz',**archive)
    dump_csv(OUTPUT/'boundary_movements.csv',movements);dump_csv(OUTPUT/'calibration_candidates.csv',cal_rows)
    write(OUTPUT/'fold_results.json',fold_results)
    assert all(sha(p)==protocol['inputs'][k]['sha256'] for k,p in INPUTS.items())
    write(OUTPUT/'summary.json',{'status':'complete','protocol_sha256':sha(OUTPUT/'protocol.json'),'results':results,
                               'input_hashes_unchanged':True,'test_accessed':False,'full_dev_refit_performed':False})
    print(json.dumps({arm:{k:v[k] for k in ['frame_f1_percent','event_f1_iou05_percent','fixed_error_seconds','introduced_error_seconds']} for arm,v in results.items()},indent=2),flush=True)

if __name__=='__main__':main()
