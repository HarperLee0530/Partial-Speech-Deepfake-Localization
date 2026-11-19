"""Verify saved audio-review predictions without fitting or reading test data."""
import argparse
from pathlib import Path
import numpy as np
from .pos_probe import read_json,write_json,sha
from .second_review_probe import context_features
from .audio_review_probe import candidates
from .metrics160 import intervals,match_events


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run
    protocol=read_json(root/'protocol.json');summary=read_json(root/'summary.json')
    assert sha(root/'protocol.json')==summary['protocol_sha256']
    for p,h in protocol['source_hashes'].items():assert sha(p)==h
    for v in protocol['inputs'].values():assert sha(v['path'])==v['sha256']
    with np.load(root/'oof_review.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    with np.load(root/'audio_features.npz',allow_pickle=False) as z:audio={k:z[k] for k in z.files}
    np.testing.assert_equal(a['uids'],audio['uids']);np.testing.assert_equal(a['lengths'],audio['lengths'])
    assert sha(root/'audio_features.npz')==summary['extraction']['feature_sha256']
    models={(m['fold'],m['arm']):m for m in read_json(root/'fitted_reviewers.json')}
    fold_results=read_json(root/'fold_results.json')
    per=read_json(root/'per_recording.json');threshold=protocol['first_pass_threshold_logit']
    arms=('original','raw_add','context_add','audio_add')
    totals={arm:dict(fp=0.,fn=0.,matched=0,predicted=0,truth=0,frame_tp=0,frame_fp=0,frame_fn=0) for arm in arms}
    offset=0;added_outside=0
    for ri,(uid,length) in enumerate(zip(a['uids'],a['lengths'])):
        length=int(length);sl=slice(offset,offset+length);offset+=length
        raw=a['scores'][sl];y=a['labels'][sl].astype(bool);w=a['widths'][sl];fold=int(a['folds'][ri])
        mask=candidates(raw,threshold);features=context_features(raw,w);duration=float(w.sum())
        np.testing.assert_equal(a['original'][sl],raw>=threshold)
        for arm in arms[1:]:
            if arm=='raw_add':expected_risk=raw
            else:
                model=models[fold,arm];x=np.column_stack([features,audio['features'][sl]]) if arm=='audio_add' else features
                expected_risk=((x-np.array(model['mean']))/np.array(model['scale']))@np.array(model['coefficient'])+model['intercept']
            np.testing.assert_allclose(a[arm+'_risk'][sl],expected_risk,atol=1e-8,rtol=0)
            cut=fold_results[fold]['arms'][arm]['threshold']
            expected=a['original'][sl]|(mask&(a[arm+'_risk'][sl]>=cut))
            np.testing.assert_equal(a[arm][sl],expected)
            added_outside+=int(np.sum((a[arm][sl]!=a['original'][sl])&~mask))
            assert np.all(a[arm][sl][a['original'][sl]])
        true_events=intervals(y,duration,.02)
        for arm in arms:
            pred=a[arm][sl];pred_events=intervals(pred,duration,.02)
            matches,_=match_events(pred_events,true_events,.5);saved=per[arm][ri]
            assert saved['uid']==str(uid) and saved['event_tp']==len(matches) and saved['event_pred']==len(pred_events)
            total=totals[arm];total['matched']+=len(matches);total['predicted']+=len(pred_events);total['truth']+=len(true_events)
            total['fp']+=float(w[pred&~y].sum());total['fn']+=float(w[~pred&y].sum())
            total['frame_tp']+=int(np.sum(pred&y));total['frame_fp']+=int(np.sum(pred&~y));total['frame_fn']+=int(np.sum(~pred&y))
    assert added_outside==0
    for arm,c in totals.items():
        r=summary['results'][arm]
        for k in ['matched','predicted','truth']:assert c[k]==r['event_counts'][k]
        assert abs(c['fp']-r['fp_seconds'])<1e-7 and abs(c['fn']-r['fn_seconds'])<1e-7
        assert abs(200*c['frame_tp']/(2*c['frame_tp']+c['frame_fp']+c['frame_fn'])-r['frame_f1_percent'])<1e-8
    for fold in fold_results:
        for choice in fold['arms'].values():assert choice['selected_real_seconds']<=choice['budget_seconds']+1e-7
    # All sample-level crop bounds and target positions must align with the global grid.
    wave=read_json(root/'waveform_hashes.json');requests=read_json(root/'crop_requests.json')
    for crop in requests:
        assert crop['start_sample']%2560==0
        assert crop['start_sample']<=crop['target_start']*320<crop['end_sample']
        assert crop['end_sample']>=min(crop['target_end']*320,wave[crop['uid']]['samples'])
        assert crop['end_sample']<=wave[crop['uid']]['samples']
    report=dict(status='passed',recordings=len(a['uids']),crop_requests=len(requests),
                checks=['Input and source hashes unchanged','All saved reviewer outputs reconstructed from fitted coefficients',
                        'Only original-real candidate frames can change; every original positive retained',
                        'Global sample/frame crop alignment checked','All native event matches, duration errors and frame F1 recomputed',
                        'Calibration cuts respect the predefined additional false-alarm budget'],no_refit=True,no_test_data_access=True)
    write_json(root/'verification.json',report);print(report)


if __name__=='__main__':main()
