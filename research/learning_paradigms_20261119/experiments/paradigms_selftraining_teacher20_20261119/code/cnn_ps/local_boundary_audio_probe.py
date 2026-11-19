"""Paired dev-only boundary experiment: score context vs score+waveform."""
import argparse
import io
import hashlib
from pathlib import Path
import joblib
import numpy as np
from scipy.io import wavfile
from sklearn.model_selection import KFold
from . import local_boundary,local_boundary_audio,metrics20
from .local_boundary import LocalBoundaryRefiner,make_candidates,apply_gains,MODEL_OPTIONS
from .local_boundary_audio import AudioBoundaryRefiner,waveform_features,add_audio_features,FEATURE_NAMES
from .local_boundary_probe import INPUTS,MARGINS,read,write,sha,dump_csv,load,evaluate,bootstrap

DEFAULT_OUTPUT=Path('D:/SAL_精简实验/result/ps_local_boundary_audio_20261116_run1')
PREVIOUS=Path('D:/SAL_精简实验/result/ps_local_boundary_20261116_run1')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args(argv);out=args.output;out.mkdir(parents=True,exist_ok=False)
    protocol={'status':'declared_before_extraction_and_fit','inputs':{k:{'path':str(p),'sha256':sha(p)} for k,p in INPUTS.items()},
              'source_hashes':{str(p):sha(p) for p in [Path(__file__),Path(local_boundary.__file__),Path(local_boundary_audio.__file__),
                                                    Path(metrics20.__file__),Path(__file__).with_name('local_boundary_probe.py')]},
              'threshold':read(INPUTS['baseline_metrics'])['threshold'],'max_shift_frames':2,'model_options':MODEL_OPTIONS,
              'audio_features':list(FEATURE_NAMES),'audio_grid':'20ms=320samples. RMS/ZCR actual cell.25ms Hann FFT512 centered at cell midpoint, zero edges.',
              'audio_context':'At original edge:80ms left/right means and difference, immediate difference, local std; proposed edge left/right/difference; moved-cell mean.126 extra features.',
              'feature_count':{'score':65,'audio':191},'global_normalization':'none; each record processed independently',
              'split':'Same5 shuffled UID folds(seed20261116): held k, cal(k+1)%5, fit other3. No test access.',
              'margins':MARGINS,'calibration':'Same as prior experiment: calibration frameF1 and eventF1 >= original baseline; FPR<=max(5%,cal baseline); maximize mean of two F1, ties prefer conservative margin; None is identity.',
              'primary':'audio_calibrated minus score_calibrated on paired held-out recordings; also report both vs original frozen detector.',
              'secondary':['score_fixed0','score_fixed025','audio_fixed0','audio_fixed025'],
              'feature_model_search':'One predeclared waveform feature set, identical tree parameters, no model search or post-hoc threshold tuning.',
              'limitations':['Detector and threshold were already selected on dev; exploratory even though refiner predictions are held-out.',
                             'UID grouping does not guarantee source/speaker independence; folds reused across previous experiments.',
                             'Bootstrap only resamples frozen fold outputs; it excludes retraining/calibration uncertainty.',
                             'No creation/deletion/merging of events; original detector/checkpoint/default inference remain unchanged.']}
    write(out/'protocol.json',protocol)
    records=load();threshold=protocol['threshold'];originals=[r['scores']>=threshold for r in records]
    score=[make_candidates(r['scores'],r['duration'],threshold) for r in records]
    baseline,base_rows=evaluate(records,originals,originals)
    old=read(PREVIOUS/'summary.json')['results'];assert baseline['frame_f1_percent']==old['baseline']['frame_f1_percent']
    manifest=read(INPUTS['manifest']);assert manifest['split']=='dev' and all(r['split']=='dev' for r in manifest['records'])
    mapping={r['uid']:r['audio'] for r in manifest['records']};audio=[];all_features=[];wave_hashes={}
    for i,(r,c) in enumerate(zip(records,score)):
        path=Path(mapping[r['uid']]);raw=path.read_bytes();sr,w=wavfile.read(io.BytesIO(raw))
        if sr!=16000 or w.ndim!=1 or w.dtype!=np.int16:raise ValueError('Expected mono16kHz PCM16 dev audio')
        wave=w.astype(np.float64)/32768.
        assert abs(len(wave)/sr-r['duration'])<1e-8
        features=waveform_features(wave,sr);all_features.append(features)
        audio.append(add_audio_features(c,features))
        assert np.array_equal(audio[-1].features[:,:65],c.features) and audio[-1].actions==c.actions
        wave_hashes[r['uid']]={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'samples':len(wave),'sample_rate':int(sr)}
        if (i+1)%400==0:print('Audio extraction',i+1,'/',len(records),flush=True)
    np.savez_compressed(out/'audio_features.npz',uids=np.array([r['uid'] for r in records]),lengths=np.array([len(a) for a in all_features]),features=np.concatenate(all_features))
    write(out/'waveform_hashes.json',wave_hashes)
    write(out/'extraction.json',{'recordings':len(records),'audio_seconds':sum(r['duration'] for r in records),
                               'audio_feature_sha256':sha(out/'audio_features.npz'),'feature_columns':list(FEATURE_NAMES)})
    folds=[held for _,held in KFold(5,shuffle=True,random_state=20261116).split(records)]
    foldfile=[{'fold':k,'uids':[records[i]['uid'] for i in ids]} for k,ids in enumerate(folds)]
    assert foldfile==read(PREVIOUS/'folds.json');write(out/'folds.json',foldfile)
    outputs={f'{source}_{mode}':[None]*len(records) for source in ('score','audio') for mode in ('fixed0','fixed025','calibrated')}
    fold_rows=[];cal_rows=[];moves=[]
    for k in range(5):
        held=folds[k];cal=folds[(k+1)%5];fit=np.concatenate([folds[j] for j in range(5) if j not in (k,(k+1)%5)])
        assert not(set(fit)&set(cal) or set(fit)&set(held) or set(cal)&set(held))
        selection={'fold':k,'fit_uids':[records[i]['uid'] for i in fit],'calibration_uids':[records[i]['uid'] for i in cal],
                   'held_uids':[records[i]['uid'] for i in held],'models':{}}
        cr=[records[i] for i in cal];co=[originals[i] for i in cal];cb,_=evaluate(cr,co,co)
        hr=[records[i] for i in held];ho=[originals[i] for i in held];hb,_=evaluate(hr,ho,ho)
        for source,candidates,cls in [('score',score,LocalBoundaryRefiner),('audio',audio,AudioBoundaryRefiner)]:
            model=cls().fit([candidates[i] for i in fit],[records[i]['labels'] for i in fit]);joblib.dump(model,out/f'{source}_fold{k}.joblib')
            cg=model.predict_gains([candidates[i] for i in cal]);hg=model.predict_gains([candidates[i] for i in held]);choices=[]
            for margin in MARGINS:
                masks=[apply_gains(candidates[i],g,margin)[0] for i,g in zip(cal,cg)]
                result,_=evaluate(cr,masks,co)
                eligible=(result['frame_f1_percent']>=cb['frame_f1_percent']-1e-10 and
                          result['event_f1_iou05_percent']>=cb['event_f1_iou05_percent']-1e-10 and
                          result['real_fpr_percent']<=max(5.,cb['real_fpr_percent'])+1e-10)
                choices.append((eligible,.5*(result['frame_f1_percent']+result['event_f1_iou05_percent']),100 if margin is None else margin,margin))
                cal_rows.append(dict(source=source,fold=k,margin=margin,eligible=eligible,frame_f1=result['frame_f1_percent'],
                                     event_f1=result['event_f1_iou05_percent'],real_fpr=result['real_fpr_percent']))
            selected=max(choice for choice in choices if choice[0])[3]
            selection['models'][source]={'selected_margin':selected,'fit_summary':model.fit_summary}
            for mode,margin in [('fixed0',0.),('fixed025',.25),('calibrated',selected)]:
                arm=f'{source}_{mode}';masks=[]
                for i,g in zip(held,hg):
                    p,spans=apply_gains(candidates[i],g,margin);outputs[arm][i]=p;masks.append(p)
                    for event,((a,b),(aa,bb)) in enumerate(zip(candidates[i].spans,spans)):
                        moves.append(dict(arm=arm,fold=k,uid=records[i]['uid'],event=event,start_frame=int(a),end_frame=int(b),
                                          start_shift_frames=int(aa-a),end_shift_frames=int(bb-b)))
                result,_=evaluate(hr,masks,ho)
                fold_rows.append(dict(arm=arm,fold=k,margin=margin,frame_f1=result['frame_f1_percent'],event_f1=result['event_f1_iou05_percent'],
                                      frame_delta=result['frame_f1_percent']-hb['frame_f1_percent'],event_delta=result['event_f1_iou05_percent']-hb['event_f1_iou05_percent']))
            print('Fold',k,source,'margin',selected,'held delta frame/event',fold_rows[-1]['frame_delta'],fold_rows[-1]['event_delta'],flush=True)
        write(out/f'fold{k}_selection.json',selection)
    results={'baseline':baseline};per={'baseline':base_rows};dump_csv(out/'baseline_per_recording.csv',base_rows)
    archive={'uids':np.array([r['uid'] for r in records]),'lengths':np.array([len(r['labels']) for r in records]),'baseline':np.concatenate(originals).astype(np.uint8)}
    with np.load(PREVIOUS/'oof_masks.npz',allow_pickle=False) as previous:
        for source,oldname in [('score_fixed0','fixed_margin_0'),('score_fixed025','fixed_margin_025'),('score_calibrated','calibrated')]:
            assert np.array_equal(np.concatenate(outputs[source]),previous[oldname]),'Score control failed prior-run reproduction'
    for arm,masks in outputs.items():
        assert all(p is not None for p in masks)
        result,rows=evaluate(records,masks,originals);results[arm]=result;per[arm]=rows
        result['delta_frame_f1_pp']=result['frame_f1_percent']-baseline['frame_f1_percent']
        result['delta_event_f1_pp']=result['event_f1_iou05_percent']-baseline['event_f1_iou05_percent']
        result['bootstrap95_vs_baseline']=bootstrap(base_rows,rows)
        assert result['event_counts']['predicted']==baseline['event_counts']['predicted']
        dump_csv(out/f'{arm}_per_recording.csv',rows);archive[arm]=np.concatenate(masks).astype(np.uint8)
    paired={}
    for mode in ('fixed0','fixed025','calibrated'):
        a=results[f'audio_{mode}'];s=results[f'score_{mode}']
        # Bootstrap helper's third field expects repair/harm relative to comparator.
        relative=[]
        for r,am,sm,row in zip(records,outputs[f'audio_{mode}'],outputs[f'score_{mode}'],per[f'audio_{mode}']):
            relative.append(dict(row,fixed_seconds=float(r['widths'][(sm!=r['labels'])&(am==r['labels'])].sum()),
                                 introduced_seconds=float(r['widths'][(sm==r['labels'])&(am!=r['labels'])].sum())))
        paired[mode]={'delta_frame_f1_pp':a['frame_f1_percent']-s['frame_f1_percent'],
                      'delta_event_f1_pp':a['event_f1_iou05_percent']-s['event_f1_iou05_percent'],
                      'net_error_reduction_seconds':a['net_error_reduction_seconds']-s['net_error_reduction_seconds'],
                      'bootstrap95':bootstrap(per[f'score_{mode}'],relative)}
    np.savez_compressed(out/'oof_masks.npz',**archive);dump_csv(out/'boundary_movements.csv',moves);dump_csv(out/'calibration_candidates.csv',cal_rows)
    write(out/'fold_results.json',fold_rows)
    assert all(sha(p)==protocol['inputs'][k]['sha256'] for k,p in INPUTS.items())
    write(out/'summary.json',{'status':'complete','results':results,'audio_vs_score':paired,'score_control_exactly_reproduced':True,
                             'test_accessed':False,'input_hashes_unchanged':True,'protocol_sha256':sha(out/'protocol.json')})
    print('COMPLETE',out,flush=True)
    for arm,r in results.items():print(arm,r['frame_f1_percent'],r['event_f1_iou05_percent'],flush=True)

if __name__=='__main__':main()
