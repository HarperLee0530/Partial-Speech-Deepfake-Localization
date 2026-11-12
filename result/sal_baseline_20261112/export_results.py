"""Export frozen SAL scores, intervals and descriptive errors; never fit or infer.

Usage: python export_results.py --project-root PATH --output NEW_DIRECTORY
Dependencies: numpy, scipy, matplotlib. Original caches are opened read-only.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN = 'baseline_pipeline_20261112_122302_654678'
TRAIN = 'sal_train_reduced_20261112_122313_966079'
EPS = 1e-8


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv_write(path, rows, fields=None):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def spans(mask, dt, seconds):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    x = np.minimum(edges.reshape(-1, 2) * dt, seconds)
    return x[(x[:, 1] - x[:, 0]) > EPS]


def match_events(truth, pred):
    n, m = len(truth), len(pred)
    overlap = np.maximum(0, np.minimum(truth[:, None, 1], pred[None, :, 1])
                         - np.maximum(truth[:, None, 0], pred[None, :, 0]))
    union = (truth[:, 1] - truth[:, 0])[:, None] + (pred[:, 1] - pred[:, 0])[None, :] - overlap
    iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
    matches = {}
    if n and m:
        eligible = iou >= .5 - EPS
        rr, cc = linear_sum_assignment(-np.where(eligible, min(n, m) + 1 + iou, 0))
        matches = {int(r): int(c) for r, c in zip(rr, cc) if eligible[r, c]}
    return overlap, iou, matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    project, out = args.project_root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Never overwrite an earlier export.
    (out / 'predictions').mkdir()
    (out / 'diagnostics').mkdir()
    (out / 'figures').mkdir()
    (out / 'training').mkdir()
    metric_rows, all_summaries, checks = [], {}, {}
    states = json.loads((project / 'result' / RUN / 'pipeline_status.json').read_text(encoding='utf-8'))
    assert states['status'] == 'complete'
    ckpt_sha = states['frozen_checkpoint']['sha256']
    for role in ('dev', 'train_edit', 'train_review'):
        source = project / 'result' / (RUN + '_' + role)
        dest = out / 'predictions' / role
        dest.mkdir()
        metrics = json.loads((source / 'metrics.json').read_text(encoding='utf-8'))
        manifest = json.loads((project / f'code/manifests/reduced/{role}.json').read_text(encoding='utf-8'))
        assert metrics['checkpoint_sha256'] == ckpt_sha
        assert metrics['manifest_sha256'] == manifest['manifest_sha256']
        for name in ('metrics.json', 'config.json', 'checkpoint_metadata.json', 'predictions.sqlite', 'status.json'):
            shutil.copy2(source / name, dest / name)
        shutil.copy2(project / f'code/manifests/reduced/{role}.json', dest / 'manifest.json')
        dt = metrics['resolution_seconds']
        assert dt == .16
        db = sqlite3.connect((source / 'predictions.sqlite').as_uri() + '?mode=ro', uri=True)
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        ids, confusion = [], np.zeros((2, 2), dtype=np.int64)
        records, events, example_pools = [], [], {}
        interval_rows = []
        with (dest / 'intervals.jsonl').open('w', encoding='utf-8') as jf, (dest / 'frames.csv').open('w', encoding='utf-8-sig', newline='') as ff:
            fw = csv.writer(ff)
            fw.writerow(['uid','frame_index','start_s','end_s','valid_duration_s','true_class_0_fake_1_real',
                         'predicted_class_0_fake_1_real','logit_fake','logit_real','margin_real_minus_fake','softmax_fake_uncalibrated'])
            for uid, raw_p, raw_y, seconds in db.execute('SELECT uid,logits,labels,seconds FROM predictions ORDER BY ordinal'):
                ids.append(uid)
                logits = np.frombuffer(raw_p, dtype=np.float32).reshape(-1, 2)
                y = np.frombuffer(raw_y, dtype=np.uint8)
                assert len(logits) == len(y) and np.isfinite(logits).all() and np.isin(y,[0,1]).all()
                pred = logits.argmax(axis=1)  # Ties select class 0, exactly as SAL.
                confusion += np.bincount(y * 2 + pred, minlength=4).reshape(2,2)
                edges = np.minimum(np.arange(len(y)+1) * dt, seconds)
                widths = np.maximum(0, np.diff(edges))
                valid = widths > EPS
                t, p = spans(y == 0, dt, seconds), spans(pred == 0, dt, seconds)
                covered_end = min(seconds, len(y)*dt)
                obj = dict(uid=uid, audio_seconds=seconds, resolution_seconds=dt, scored_end_s=covered_end,
                           unscored_tail=[[covered_end, seconds]] if seconds-covered_end>EPS else [],
                           predicted_fake=p.tolist(), predicted_real=spans(pred==1,dt,seconds).tolist(),
                           true_fake=t.tolist(), true_real=spans(y==1,dt,seconds).tolist())
                if role != 'train_review':
                    obj['false_negative_intervals'] = spans((y==0)&(pred==1),dt,seconds).tolist()
                    obj['false_positive_intervals'] = spans((y==1)&(pred==0),dt,seconds).tolist()
                jf.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + '\n')
                for key in ('predicted_fake','predicted_real','true_fake','true_real','unscored_tail'):
                    for a,b in obj[key]:
                        interval_rows.append(dict(uid=uid,kind=key,start_s=a,end_s=b,duration_s=b-a))
                margin = logits[:,1].astype(float)-logits[:,0].astype(float)
                prob = 1/(1+np.exp(np.clip(margin,-700,700)))
                fw.writerows((uid,i,float(edges[i]),float(edges[i+1]),float(widths[i]),int(y[i]),int(pred[i]),
                              float(logits[i,0]),float(logits[i,1]),float(margin[i]),float(prob[i])) for i in range(len(y)))
                # Check that the interval export reconstructs every positive-width frame exactly.
                mid=(edges[:-1]+edges[1:])/2
                rebuilt=np.zeros(len(y),dtype=bool)
                for a,b in p: rebuilt |= (mid>=a)&(mid<b)
                assert np.array_equal(rebuilt[valid],(pred==0)[valid]),uid
                if role == 'train_review':
                    continue  # Preserve selector role: export predictions, do not select error examples.
                ov,iou,matched=match_events(t,p)
                tg=(ov>EPS).sum(axis=1); pg=(ov>EPS).sum(axis=0)
                kind='all_real' if np.all(y[valid]==1) else ('all_fake' if np.all(y[valid]==0) else 'partial')
                fn=float(widths[(y==0)&(pred==1)].sum()); fp=float(widths[(y==1)&(pred==0)].sum())
                internal=edge_miss=whole_miss=0.
                start_errors=[]; end_errors=[]
                for ti,(a,b) in enumerate(t):
                    missed=b-a-float(ov[ti].sum())
                    idx=np.flatnonzero(ov[ti]>EPS)
                    if not len(idx): whole_miss += missed
                    else:
                        first=max(a,p[idx[0],0]); last=min(b,p[idx[-1],1])
                        edge=(first-a)+(b-last)
                        edge_miss+=edge; internal+=max(0,missed-edge)
                    pi=matched.get(ti)
                    ds=float(p[pi,0]-a) if pi is not None else None
                    de=float(p[pi,1]-b) if pi is not None else None
                    if pi is not None: start_errors.append(ds); end_errors.append(de)
                    events.append(dict(uid=uid,scope=kind,true_event_index=ti,start_s=a,end_s=b,duration_s=b-a,
                        overlaps=int(tg[ti]),complete_miss=int(tg[ti]==0),fragmented=int(tg[ti]>=2),
                        matched_tiou_0_5=int(pi is not None),matched_prediction_index=pi,
                        tiou=float(iou[ti,pi]) if pi is not None else None,start_error_s=ds,end_error_s=de))
                assert abs(internal+edge_miss+whole_miss-fn)<1e-5
                row=dict(uid=uid,scope=kind,audio_seconds=seconds,scored_seconds=float(widths.sum()),
                    unscored_tail_seconds=max(0,seconds-covered_end),fake_seconds=float(widths[y==0].sum()),
                    real_seconds=float(widths[y==1].sum()),false_negative_seconds=fn,false_positive_seconds=fp,
                    internal_miss_seconds=internal,boundary_miss_seconds=edge_miss,whole_event_miss_seconds=whole_miss,
                    true_events=len(t),predicted_events=len(p),matched_events=len(matched),complete_misses=int((tg==0).sum()),
                    fragmented_true_events=int((tg>=2).sum()),merged_predicted_events=int((pg>=2).sum()),
                    spurious_predicted_events=int((pg==0).sum()),
                    boundary_shifted_matches=sum(abs(a)>EPS or abs(b)>EPS for a,b in zip(start_errors,end_errors)))
                records.append(row)
                categories=[]
                if row['complete_misses']: categories.append('complete_miss')
                if row['fragmented_true_events']: categories.append('fragmentation')
                if row['merged_predicted_events']: categories.append('merge')
                if row['boundary_shifted_matches']: categories.append('boundary_offset')
                if row['spurious_predicted_events']: categories.append('spurious_false_alarm')
                if kind=='all_fake' and fn>EPS: categories.append('all_fake_miss')
                if kind=='all_real' and fp>EPS: categories.append('all_real_false_alarm')
                if fn+fp<EPS: categories.append('correct_control')
                for category in categories: example_pools.setdefault(category,[]).append(uid)
        assert ids == manifest['utterance_ids']
        assert confusion.tolist() == metrics['confusion_true_rows_pred_columns_0_spoof_1_bonafide']
        csv_write(dest/'intervals.csv',interval_rows)
        tn,fn,fp,tp=confusion.ravel().tolist()  # fake-positive interpretation; labels retain original encoding.
        assert abs(100*2*tn/(2*tn+fn+fp)-metrics['f1_spoof_percent'])<1e-6
        metric_rows.append(dict(role=role,recordings=len(ids),eer_percent=metrics['eer_percent'],
            fake_f1_percent=metrics['f1_spoof_percent'],real_f1_percent=metrics['f1_bonafide_percent'],
            accuracy_percent=metrics['accuracy_percent'],false_negative_frames=fn,false_positive_frames=fp,
            fake_frames=tn+fn,real_frames=tp+fp,checkpoint_sha256=ckpt_sha))
        checks[role]=dict(ids_match=True,confusion_matches_saved_metrics=True,interval_roundtrip=True,
                          database_integrity='ok',recordings=len(ids))
        if role != 'train_review':
            diag=out/'diagnostics'/role; diag.mkdir()
            csv_write(diag/'recording_errors.csv',records)
            csv_write(diag/'true_event_errors.csv',events)
            totals={}
            for kind in ('all','partial','all_real','all_fake'):
                rr=records if kind=='all' else [r for r in records if r['scope']==kind]
                values={key:sum(r[key] for r in rr) for key in records[0] if key not in ('uid','scope')}
                values['recordings']=len(rr)
                values['event_f1_tiou_0_5_percent']=100*2*values['matched_events']/(values['true_events']+values['predicted_events']) if values['true_events']+values['predicted_events'] else None
                totals[kind]=values
            short={}
            for name,lo,hi in [('up_to_160ms',0,.16),('160_to_320ms',.16,.32),('320_to_640ms',.32,.64),('over_640ms',.64,float('inf'))]:
                ee=[e for e in events if e['duration_s']>lo+EPS and e['duration_s']<=hi+EPS]
                short[name]=dict(events=len(ee),complete_misses=sum(e['complete_miss'] for e in ee),
                                 matched=sum(e['matched_tiou_0_5'] for e in ee),fragmented=sum(e['fragmented'] for e in ee))
            summary=dict(role=role,checkpoint_sha256=ckpt_sha,scopes=totals,event_duration_groups=short,
                         fixed_decision='original binary-head argmax, ties -> fake',diagnostic_only=True,
                         categories_overlap=True,boundary_resolution_seconds=dt,confidence_intervals_computed=False)
            save(diag/'summary.json',summary); all_summaries[role]=summary
            selected=[]
            for cat,pool in sorted(example_pools.items()):
                # Stable hash example among every eligible recording; not a worst-case search.
                uid=min(pool,key=lambda u:hashlib.sha256((role+'|'+cat+'|'+u).encode()).hexdigest())
                selected.append(dict(category=cat,uid=uid,eligible_recordings=len(pool),selection='minimum SHA256(role|category|uid)'))
            save(diag/'examples.json',selected)
            fig,axes=plt.subplots(max(1,len(selected)),1,figsize=(12,max(3,len(selected)*2.2)),squeeze=False)
            for ax,example in zip(axes[:,0],selected):
                raw_p,raw_y,seconds=db.execute('SELECT logits,labels,seconds FROM predictions WHERE uid=?',(example['uid'],)).fetchone()
                yy=np.frombuffer(raw_y,dtype=np.uint8); pp=np.frombuffer(raw_p,dtype=np.float32).reshape(-1,2).argmax(1)
                for mask,base,color in [(yy==0,.55,'#2489a6'),(pp==0,.1,'#ac4a85')]:
                    for a,b in spans(mask,dt,seconds): ax.broken_barh([(a,b-a)],(base,.25),facecolors=color)
                ax.set_yticks([.225,.675],['Predicted fake','True fake']); ax.set_xlim(0,seconds); ax.set_ylim(0,1)
                ax.set_title(f'{example["category"]} | {example["uid"]}',loc='left',fontsize=9)
                ax.set_xlabel('Time (s); blank within scored range = real; 160 ms grid')
                if len(yy)*dt<seconds-EPS: ax.axvspan(len(yy)*dt,seconds,color='gray',alpha=.4)
            fig.suptitle(f'{role}: fixed SAL decisions, descriptive examples (not prevalence)',fontsize=12)
            fig.tight_layout(rect=[0,0,1,.98]); fig.savefig(out/'figures'/f'{role}_error_examples.png',dpi=160); plt.close(fig)
        db.close()
    csv_write(out/'metrics_summary.csv',metric_rows)
    save(out/'verification.json',checks)
    for name in ('resolved_config.json','status.json'):
        shutil.copy2(project/'result'/TRAIN/name,out/'training'/name)
    shutil.copy2(project/'result'/TRAIN/'logs/version_0/metrics.csv',out/'training/epoch_metrics.csv')
    shutil.copy2(project/'result'/(TRAIN+'.json'),out/'training/training_record.json')
    shutil.copy2(project/'result'/RUN/'train.log',out/'training/train.log')
    shutil.copy2(project/'result'/RUN/'frozen_checkpoint.json',out/'training/frozen_checkpoint.json')
    shutil.copy2(project/'result'/RUN/'pipeline_status.json',out/'training/pipeline_status.json')
    with (out/'training/epoch_metrics.csv').open(encoding='utf-8',newline='') as f: curve=[r for r in csv.DictReader(f) if r.get('val/eer')]
    fig,ax=plt.subplots(figsize=(9,4)); ax.plot([int(r['epoch'])+1 for r in curve],[float(r['val/eer']) for r in curve],marker='.',color='#247a9c')
    ax.scatter([25],[1.8525282144546509],color='#bd447b',label='Selected: epoch 25'); ax.set(xlabel='Epoch (one-based)',ylabel='dev EER (%)',title='Reduced SAL: dev checkpoint selection'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'figures/dev_eer_curve.png',dpi=160); plt.close(fig)
    save(out/'diagnostics/summary.json',all_summaries)
    print(json.dumps(dict(metrics=metric_rows,checks=checks),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
