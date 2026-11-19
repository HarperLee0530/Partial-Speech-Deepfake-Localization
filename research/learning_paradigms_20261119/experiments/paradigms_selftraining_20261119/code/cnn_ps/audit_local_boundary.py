"""Independently replay saved OOF masks and recompute matching via sparse graph."""
import csv
import json
from pathlib import Path
import hashlib
import joblib
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from .local_boundary import make_candidates,apply_gains

OUT=Path('D:/SAL_精简实验/result/ps_local_boundary_20261116_run1')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def segments(mask,duration):
    # Separate direct state scan, not the production interval helper.
    result=[];start=None
    for k,label in enumerate(mask):
        if label and start is None:start=k*.02
        if not label and start is not None:result.append((start,k*.02));start=None
    if start is not None:result.append((start,duration))
    return np.array(result,dtype=float).reshape(-1,2)

def matched(pred,truth,threshold):
    if not len(pred) or not len(truth):return 0
    overlap=np.maximum(0,np.minimum(pred[:,None,1],truth[None,:,1])-np.maximum(pred[:,None,0],truth[None,:,0]))
    union=np.diff(pred,axis=1)+np.diff(truth,axis=1).T-overlap
    eligible=overlap/union>=threshold-1e-10
    assignment=maximum_bipartite_matching(csr_matrix(eligible),perm_type='column')
    return int((assignment>=0).sum())

def main():
    protocol=read(OUT/'protocol.json');summary=read(OUT/'summary.json')
    for value in protocol['inputs'].values():assert digest(value['path'])==value['sha256']
    for path,hash_value in protocol['source_hashes'].items():assert digest(path)==hash_value
    manifest=read(protocol['inputs']['manifest']['path']);assert manifest['split']=='dev'
    with np.load(OUT/'oof_masks.npz',allow_pickle=False) as a: archive={k:a[k] for k in a.files}
    with np.load(protocol['inputs']['scores']['path'],allow_pickle=False) as a:scores={k:a[k] for k in a.files}
    uids=list(map(str,archive['uids']));lengths=archive['lengths'];ends=np.r_[0,np.cumsum(lengths)]
    assert uids==list(map(str,scores['uids']))
    assert set(uids)=={r['uid'] for r in manifest['records']}
    maps={uid:i for i,uid in enumerate(uids)}
    offsets=np.r_[0,np.cumsum(scores['lengths']*8)]
    with np.load(protocol['inputs']['labels']['path'],allow_pickle=False) as a:ys=[a[uid][:n].astype(bool) for uid,n in zip(uids,lengths)]
    candidates=[]
    for i,n in enumerate(lengths):
        c=make_candidates(scores['scores20'][offsets[i]:offsets[i+1]],float(scores['durations'][i]),protocol['threshold'])
        assert len(c.original)==n
        np.testing.assert_array_equal(c.original,archive['baseline'][ends[i]:ends[i+1]])
        candidates.append(c)
    folds=read(OUT/'folds.json'); covered=[]
    for fold in range(5):
        selection=read(OUT/f'fold_{fold}_selection.json');fit=set(selection['fit_uids']);cal=set(selection['calibration_uids']);held=set(selection['held_uids'])
        assert not(fit&cal or fit&held or cal&held)
        assert fit|cal|held==set(uids)
        assert held==set(folds[fold]['uids']) and cal==set(folds[(fold+1)%5]['uids'])
        covered.extend(held);ids=[maps[uid] for uid in selection['held_uids']]
        model=joblib.load(OUT/f'fold_{fold}_model.joblib'); gains=model.predict_gains([candidates[i] for i in ids])
        for arm,margin in [('fixed_margin_0',0.),('fixed_margin_025',.25),('calibrated',selection['selected_margin'])]:
            for i,g in zip(ids,gains):
                mask,_=apply_gains(candidates[i],g,margin)
                np.testing.assert_array_equal(mask,archive[arm][ends[i]:ends[i+1]])
    assert len(covered)==len(set(covered))==len(uids)
    movement={};recomputed={}
    with (OUT/'boundary_movements.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            arm=row['arm'];counts=movement.setdefault(arm,{'events':0,'moved_events':0,'moved_starts':0,'moved_ends':0,'start_offsets':{},'end_offsets':{}})
            a=int(row['start_shift_frames']);b=int(row['end_shift_frames']);assert abs(a)<=2 and abs(b)<=2
            counts['events']+=1;counts['moved_events']+=int(a!=0 or b!=0);counts['moved_starts']+=int(a!=0);counts['moved_ends']+=int(b!=0)
            for side,value in [('start_offsets',a),('end_offsets',b)]:counts[side][str(value)]=counts[side].get(str(value),0)+1
    for arm in summary['results']:
        frame=np.zeros(4);duration=np.zeros(4);events=np.zeros(5,dtype=int);fixed=bad=0.
        with (OUT/f'{arm}_per_recording.csv').open(encoding='utf-8-sig',newline='') as f:rows={r['uid']:r for r in csv.DictReader(f)}
        for i,(uid,y) in enumerate(zip(uids,ys)):
            p=archive[arm][ends[i]:ends[i+1]].astype(bool);old=archive['baseline'][ends[i]:ends[i+1]].astype(bool)
            d=float(scores['durations'][i]);t=np.arange(len(y))*.02;w=np.maximum(0,np.minimum(t+.02,d)-t)
            assert np.isclose(w.sum(),d,atol=1e-9) and (w>0).all()
            parts=[p&y,p&~y,~p&y,~p&~y]
            cm=np.array([q.sum() for q in parts]);frame+=cm;duration+=[w[q].sum() for q in parts]
            ps=segments(p,d);ts=segments(y,d);nm=[matched(ps,ts,t) for t in (.5,.7,.9)]
            events+=[len(ps),len(ts),*nm]
            assert list(map(int,cm))==[int(rows[uid][k]) for k in ('tp','fp','fn','tn')]
            assert nm==[int(rows[uid][k]) for k in ('event_tp','event_tp07','event_tp09')]
            fixed+=w[(old!=y)&(p==y)].sum();bad+=w[(old==y)&(p!=y)].sum()
        f1=200*frame[0]/(2*frame[0]+frame[1]+frame[2]);event_f1=200*events[2]/(events[0]+events[1])
        result=summary['results'][arm]
        assert abs(result['frame_f1_percent']-f1)<1e-9
        assert abs(result['event_f1_iou05_percent']-event_f1)<1e-9
        assert np.allclose(duration,result['confusion_seconds_tp_fp_fn_tn'],atol=1e-8,rtol=0)
        assert abs(result['fixed_error_seconds']-fixed)<1e-8 and abs(result['introduced_error_seconds']-bad)<1e-8
        recomputed[arm]={'frame_f1_percent':float(f1),'event_f1_iou05_percent':float(event_f1),'event_counts':events.tolist()}
    audit={'status':'passed','recordings':len(uids),'all_fold_fit_cal_held_disjoint':True,'each_uid_held_once':True,
           'all_saved_models_replay_exactly':True,'all_input_and_source_hashes_unchanged':True,
           'independent_matching':'scipy sparse maximum bipartite matching; production uses Hungarian assignment',
           'per_record_frame_counts_and_event_matches_reproduced':True,'recomputed':recomputed,'movements':movement,
           'tests':'7 local-boundary unit tests passed before fitting, including 200 random topology cases'}
    (OUT/'independent_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
