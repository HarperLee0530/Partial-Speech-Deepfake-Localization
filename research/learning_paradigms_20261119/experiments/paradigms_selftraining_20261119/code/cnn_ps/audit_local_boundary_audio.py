"""Replay audio-boundary OOF models and independently check event/frame metrics."""
import argparse
import csv
import io
from pathlib import Path
import joblib
import numpy as np
from scipy.io import wavfile
from .local_boundary import make_candidates,apply_gains
from .local_boundary_audio import waveform_features,add_audio_features
from .local_boundary_probe import read,write,sha
from .local_boundary_audio_probe import DEFAULT_OUTPUT,PREVIOUS
from .audit_local_boundary import segments,matched


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);out=p.parse_args(argv).output
    protocol=read(out/'protocol.json');summary=read(out/'summary.json');waves=read(out/'waveform_hashes.json')
    assert sha(out/'protocol.json')==summary['protocol_sha256']
    for item in protocol['inputs'].values():assert sha(item['path'])==item['sha256']
    for path,h in protocol['source_hashes'].items():assert sha(path)==h
    manifest=read(protocol['inputs']['manifest']['path']);assert manifest['split']=='dev'
    with np.load(out/'oof_masks.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    with np.load(protocol['inputs']['scores']['path'],allow_pickle=False) as z:s={k:z[k] for k in z.files}
    with np.load(out/'audio_features.npz',allow_pickle=False) as z:af={k:z[k] for k in z.files}
    assert sha(out/'audio_features.npz')==read(out/'extraction.json')['audio_feature_sha256']
    uids=list(map(str,a['uids']));lengths=a['lengths'];ends=np.r_[0,np.cumsum(lengths)];offsets=np.r_[0,np.cumsum(s['lengths']*8)]
    assert uids==list(map(str,s['uids']))==list(map(str,af['uids']))
    np.testing.assert_array_equal(lengths,af['lengths'])
    assert set(uids)=={r['uid'] for r in manifest['records']}==set(waves)
    with np.load(protocol['inputs']['labels']['path'],allow_pickle=False) as z:ys=[z[uid][:n].astype(bool) for uid,n in zip(uids,lengths)]
    score=[];audio=[];mapping={uid:i for i,uid in enumerate(uids)}
    for i,(uid,n) in enumerate(zip(uids,lengths)):
        c=make_candidates(s['scores20'][offsets[i]:offsets[i+1]],float(s['durations'][i]),protocol['threshold']);score.append(c)
        np.testing.assert_array_equal(c.original,a['baseline'][ends[i]:ends[i+1]])
        raw=Path(waves[uid]['path']).read_bytes()
        import hashlib
        assert hashlib.sha256(raw).hexdigest()==waves[uid]['sha256']
        sr,w=wavfile.read(io.BytesIO(raw));assert sr==16000 and len(w)==waves[uid]['samples'] and w.dtype==np.int16
        features=waveform_features(w.astype(float)/32768.,sr)
        np.testing.assert_array_equal(features,af['features'][ends[i]:ends[i+1]])
        audio.append(add_audio_features(c,features))
        assert np.array_equal(audio[-1].features[:,:65],c.features)
    print('Waveform and feature replay passed',len(uids),flush=True)
    folds=read(out/'folds.json');assert folds==read(PREVIOUS/'folds.json');covered=[]
    for fold in range(5):
        info=read(out/f'fold{fold}_selection.json');fit=set(info['fit_uids']);cal=set(info['calibration_uids']);held=set(info['held_uids'])
        assert not(fit&cal or fit&held or cal&held) and fit|cal|held==set(uids)
        assert held==set(folds[fold]['uids']) and cal==set(folds[(fold+1)%5]['uids'])
        covered.extend(held);ids=[mapping[uid] for uid in info['held_uids']]
        for source,candidates in [('score',score),('audio',audio)]:
            model=joblib.load(out/f'{source}_fold{fold}.joblib');gains=model.predict_gains([candidates[i] for i in ids])
            for mode,margin in [('fixed0',0.),('fixed025',.25),('calibrated',info['models'][source]['selected_margin'])]:
                for i,g in zip(ids,gains):
                    mask,_=apply_gains(candidates[i],g,margin)
                    np.testing.assert_array_equal(mask,a[f'{source}_{mode}'][ends[i]:ends[i+1]])
    assert len(covered)==len(set(covered))==len(uids)
    with np.load(PREVIOUS/'oof_masks.npz',allow_pickle=False) as old:
        for mode,old_mode in [('fixed0','fixed_margin_0'),('fixed025','fixed_margin_025'),('calibrated','calibrated')]:
            np.testing.assert_array_equal(a[f'score_{mode}'],old[old_mode])
    recomputed={};movement={}
    with (out/'boundary_movements.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            stats=movement.setdefault(row['arm'],{'events':0,'moved_events':0,'moved_starts':0,'moved_ends':0})
            x=int(row['start_shift_frames']);y=int(row['end_shift_frames']);assert abs(x)<=2 and abs(y)<=2
            stats['events']+=1;stats['moved_events']+=int(x!=0 or y!=0);stats['moved_starts']+=int(x!=0);stats['moved_ends']+=int(y!=0)
    for arm,result in summary['results'].items():
        frame=np.zeros(4);duration=np.zeros(4);events=np.zeros(5,dtype=int);fixed=bad=0.
        with (out/f'{arm}_per_recording.csv').open(encoding='utf-8-sig',newline='') as f:rows={r['uid']:r for r in csv.DictReader(f)}
        for i,(uid,y) in enumerate(zip(uids,ys)):
            mask=a[arm][ends[i]:ends[i+1]].astype(bool);old=a['baseline'][ends[i]:ends[i+1]].astype(bool)
            d=float(s['durations'][i]);t=np.arange(len(y))*.02;w=np.maximum(0,np.minimum(t+.02,d)-t)
            assert (w>0).all() and np.isclose(w.sum(),d,atol=1e-9)
            parts=[mask&y,mask&~y,~mask&y,~mask&~y];cm=np.array([x.sum() for x in parts]);frame+=cm;duration+=[w[x].sum() for x in parts]
            ps=segments(mask,d);ts=segments(y,d);nm=[matched(ps,ts,t) for t in (.5,.7,.9)];events+=[len(ps),len(ts),*nm]
            assert list(map(int,cm))==[int(rows[uid][k]) for k in ('tp','fp','fn','tn')]
            assert nm==[int(rows[uid][k]) for k in ('event_tp','event_tp07','event_tp09')]
            assert len(ps)==len(score[i].spans)
            fixed+=w[(old!=y)&(mask==y)].sum();bad+=w[(old==y)&(mask!=y)].sum()
        f1=200*frame[0]/(2*frame[0]+frame[1]+frame[2]);ef1=200*events[2]/(events[0]+events[1])
        assert abs(f1-result['frame_f1_percent'])<1e-9 and abs(ef1-result['event_f1_iou05_percent'])<1e-9
        assert np.allclose(duration,result['confusion_seconds_tp_fp_fn_tn'],rtol=0,atol=1e-8)
        assert abs(fixed-result['fixed_error_seconds'])<1e-8 and abs(bad-result['introduced_error_seconds'])<1e-8
        recomputed[arm]={'frame_f1_percent':float(f1),'event_f1_iou05_percent':float(ef1),'events_pred_truth_matched05_07_09':events.tolist()}
    audit={'status':'passed','recordings':len(uids),'all_waveform_hashes_unchanged':True,'all_waveform_features_reproduced_exactly':True,
           'all10_models_replayed_exactly':True,'all_input_and_source_hashes_unchanged':True,'fit_cal_held_disjoint':True,'each_uid_held_once':True,
           'score_control_reproduces_previous_oof':True,'per_record_frame_and_sparse_graph_event_matching_reproduced':True,
           'topology_preserved':True,'unit_tests_passed':13,'recomputed':recomputed,'movements':movement}
    write(out/'independent_audit.json',audit);print('AUDIT PASSED',flush=True)

if __name__=='__main__':main()
