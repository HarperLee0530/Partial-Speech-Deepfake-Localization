"""Independent duration-budget and decision audit for total-FPR calibration."""
import argparse
from pathlib import Path
import numpy as np
from .pos_probe import read_json,write_json,sha
from .metrics160 import intervals,match_events


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    root=p.parse_args().run;protocol=read_json(root/'protocol.json');summary=read_json(root/'summary.json')
    assert sha(root/'protocol.json')==summary['protocol_sha256']
    for v in protocol['source_inputs'].values():assert sha(v['path'])==v['sha256']
    for path,h in protocol['source_hashes'].items():assert sha(path)==h
    with np.load(root/'decisions.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    offsets=np.r_[0,np.cumsum(a['lengths'])];y=a['labels'].astype(bool);w=a['widths']
    totals={};changes={};real_seconds=float(w[~y].sum())
    for mode,block in [('pooled','pooled_calibration_results'),('held','separate_fold_results')]:
        for arm in protocol['arms']:
            pred=a[f'{mode}_{arm}'];r=summary[block][arm]
            fp=float(w[pred&~y].sum());fn=float(w[~pred&y].sum())
            tp=int(np.sum(pred&y));fp_n=int(np.sum(pred&~y));fn_n=int(np.sum(~pred&y))
            assert abs(fp-r['fp_seconds'])<1e-7 and abs(fn-r['fn_seconds'])<1e-7
            assert abs(200*tp/(2*tp+fp_n+fn_n)-r['frame_f1_percent'])<1e-9
            event_tp=event_p=event_t=0
            for i in range(len(a['uids'])):
                sl=slice(offsets[i],offsets[i+1]);d=float(w[sl].sum())
                predicted=intervals(pred[sl],d,.02);truth=intervals(y[sl],d,.02)
                pairs,_=match_events(predicted,truth,.5)
                event_tp+=len(pairs);event_p+=len(predicted);event_t+=len(truth)
            assert event_tp==r['event_counts']['matched'] and event_p==r['event_counts']['predicted'] and event_t==r['event_counts']['truth']
            if mode=='pooled':
                cut=summary['pooled_thresholds'][arm]['threshold'];scores=a['scores_'+arm]
                np.testing.assert_equal(pred,scores>=cut)
                assert fp<=real_seconds*.05+1e-8
                negative=scores[~y];negative_w=w[~y];excluded=negative[negative<cut]
                if len(excluded):
                    boundary=excluded.max();next_group=float(negative_w[negative==boundary].sum())
                    assert fp+next_group>real_seconds*.05-1e-9
            else:
                choices=read_json(root/'fold_thresholds.json')
                for i,fold in enumerate(a['folds']):
                    sl=slice(offsets[i],offsets[i+1]);cut=choices[int(fold)]['arms'][arm]['threshold']
                    np.testing.assert_equal(pred[sl],a['scores_'+arm][sl]>=cut)
                    assert choices[int(fold)]['arms'][arm]['real_fpr_percent']<=5+1e-9
            totals[f'{mode}_{arm}']=dict(fp_seconds=fp,fn_seconds=fn,real_fpr_percent=100*fp/real_seconds)
            base=a[f'{mode}_baseline'];add=pred&~base;remove=base&~pred
            changes[f'{mode}_{arm}']=dict(recovered_fn_seconds=float(w[add&y].sum()),
                 lost_tp_seconds=float(w[remove&y].sum()),added_fp_seconds=float(w[add&~y].sum()),
                 removed_fp_seconds=float(w[remove&~y].sum()))
    report=dict(status='passed',recordings=len(a['uids']),checks=['All source/result input hashes unchanged',
           'Each pooled arm is within total 5% genuine-duration FPR; next tied group would exceed budget',
           'Each held-out prediction uses only its calibration-fold cut',
           'All frame F1, duration errors and event matching counts recomputed'],
           totals=totals,decision_changes_vs_baseline=changes,no_fit=True,no_test_access=True)
    write_json(root/'verification.json',report);print(__import__('json').dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
