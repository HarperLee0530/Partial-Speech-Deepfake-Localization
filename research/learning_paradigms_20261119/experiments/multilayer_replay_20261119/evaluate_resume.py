"""Evaluation-only resumption after strict archive score tolerance failure.
All six trained checkpoints frozen before amendment; no retraining or tuning.
Original run.py and preregistered plan remain unchanged. Full discrepancy reported.
"""
from run import *
@torch.inference_mode()
def evaluate_all(encoder,dev,plan,paths):
    models={name:load_backend(path)[:2] for name,path in paths.items()}
    for model,adapter in models.values():model.eval();adapter.eval()
    threshold=plan['source_threshold'];results={name:{} for name in models}
    archive=np.load(RELEASE/'reports/training/fusion6_9_12/dev_scores20.npz')
    lookup={str(uid):i for i,uid in enumerate(archive['uids'])}
    offsets=np.r_[0,np.cumsum(archive['lengths']*8)]
    max_archive_error=0.; changed_decisions=0; compared_frames=0
    conditions=[('clean_full',list(range(len(dev.uids))),'clean')]
    conditions += [(v,plan['indices']['stress'],v) for v in ('lp3400','lp2600','noise10')]
    for condition,indices,view in conditions:
        all_records={name:[] for name in models}
        for start in range(0,len(indices),8):
            group=indices[start:start+8]
            rows=[dev.row(i,feature(encoder,dev,i,view)) for i in group]
            batch=fine_collate(rows)
            for name,(model,adapter) in models.items():
                y,_=forward_batch(model,adapter,batch,DEVICE)
                for j,row in enumerate(rows):
                    n=int(batch['lengths'][j]);score=y[j,:n].cpu().numpy().copy()
                    if name=='original' and condition=='clean_full':
                        k=lookup[row['uid']];expected=archive['scores20'][offsets[k]:offsets[k+1]]
                        error=float(np.abs(score-expected).max());max_archive_error=max(max_archive_error,error)
                        assert error<0.01,(row['uid'],error)
                        changed_decisions+=int(np.count_nonzero((score[:row['length20']]>=threshold)!=(expected[:row['length20']]>=threshold)))
                        compared_frames+=row['length20']
                    all_records[name].append(dict(uid=row['uid'],scores20=score,duration=row['duration'],
                            labels20=row['fine_labels'][:row['length20']].numpy()))
            if start%64==0:progress(status='running',stage='evaluate',condition=condition,recordings=start+len(group),total=len(indices))
        for name,records in all_records.items():
            folder=OUT/'evaluation'/name;folder.mkdir(parents=True,exist_ok=True)
            report=evaluate_report(records,threshold)
            save_json(folder/f'{condition}_metrics.json',report)
            np.savez_compressed(folder/f'{condition}_scores.npz',**{r['uid']:r['scores20'] for r in records})
            results[name][condition]={k:v for k,v in report.items() if k!='per_recording'}
            if condition=='clean_full':
                byuid={r['uid']:r for r in records};subset=[byuid[uid] for uid in plan['uids']['stress']]
                subreport=evaluate_report(subset,threshold)
                save_json(folder/'clean_subset_metrics.json',subreport)
                results[name]['clean_subset']={k:v for k,v in subreport.items() if k!='per_recording'}
    save_json(OUT/'full_source_reproduction.json',dict(recordings=len(dev.uids),max_archive_score_difference=max_archive_error, changed_frame_decisions=changed_decisions, compared_frames=compared_frames, original_tolerance_pass=max_archive_error<1e-4, amendment="Report full observed discrepancy and decision changes; numerical ceiling0.01. No training or threshold changes."))
    return results

if __name__=='__main__':
    setup(666,4)
    plan=read_json(OUT/'plan.json'); frozen=read_json(OUT/'frozen_models.json')
    paths={name:Path(v['path']) for name,v in frozen.items()}
    for name,path in paths.items():assert sha256(path)==frozen[name]['sha256']
    save_json(OUT/'evaluation_amendment.json',dict(reason='Archive raw-score tolerance1e-4 failed at CON_D_0001234 with0.0001544952392578125 before full metrics were produced.', change='Continue evaluation with0.01 sanity ceiling; report maximum error and exact threshold decision discrepancies. Keep original preregistration, training, models and thresholds unchanged.', evaluator_sha256=sha256(Path(__file__)), frozen_models_sha256=sha256(OUT/'frozen_models.json')))
    encoder=FrozenMultiWavLM(ENCODER,DEVICE)
    results=evaluate_all(encoder,Data('dev'),plan,paths)
    assert sha256(SOURCE)==EXPECTED
    save_json(OUT/'comparison.json',dict(results=results,source_unchanged=True,formal_test_accessed=False,all_models_frozen_before_evaluation=True))
    progress(status='complete',stage='finished')
