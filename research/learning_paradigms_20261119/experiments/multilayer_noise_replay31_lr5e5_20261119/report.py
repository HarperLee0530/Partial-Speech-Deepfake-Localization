"""Paired regression/robustness report; never tunes model or threshold."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from cnn_ps.io import read_json,save_json,sha256
from cnn_ps.metrics20 import decision_mask

CONDITIONS=['clean_full','clean_subset','noise10','noise0','pink10','lp2600']
KEYS=['event_f1','frame_f1','real_fpr','short_recall','duration_f1']
PER=['tp_seconds','fp_seconds','fn_seconds','tn_seconds','event_tp','event_pred','event_truth','short_tp','short_truth']
NAMES={'original':'原模型','old_only':'只继续训练旧条件','new_only':'仅白噪声增强','replay':'3:1回放学习率5e-5','previous_replay':'3:1回放学习率1e-4'}


def stats(a):
    dtp,dfp,dfn,dtn,etp,ep,et,st,stn,ftp,ffp,ffn,ftn=np.moveaxis(a,-1,0)
    ratio=lambda x,y:100*np.divide(x,y,out=np.zeros_like(x,dtype=float),where=y!=0)
    return np.stack([ratio(2*etp,ep+et),ratio(2*ftp,2*ftp+ffp+ffn),ratio(dfp,dfp+dtn),
                     ratio(st,stn),ratio(2*dtp,2*dtp+dfp+dfn)],axis=-1)


def main():
    out=ROOT/'results';plan=read_json(out/'plan.json');frozen=read_json(out/'frozen_models.json')
    assert read_json(out/'status.json')['status']=='complete'
    assert sha256(plan['source'])==plan['source_sha256']
    assert sha256(ROOT/'run.py')==plan['code_sha256']['run.py']
    meta=read_json(Path('D:/SAL_精简实验/data/cnn_ps_features_v2/dev/complete.json'))
    labels=np.load('D:/SAL_精简实验/data/cnn_ps_features_v2/dev/labels20.npy',mmap_mode='r')
    truth={uid:labels[meta['offsets'][i]:meta['offsets'][i]+meta['lengths20'][i]] for i,uid in enumerate(meta['uids'])}
    durations={uid:float(meta['durations'][i]) for i,uid in enumerate(meta['uids'])}
    rows=[];arrays={}
    for name,identity in frozen.items():
        assert sha256(identity['path'])==identity['sha256']
        assert identity['threshold']==plan['source_threshold']
        folder=out/'evaluation'/name
        for condition in CONDITIONS:
            report=read_json(folder/f'{condition}_metrics.json')
            expected=plan['uids']['clean_full'] if condition=='clean_full' else plan['uids']['stress']
            assert [r['uid'] for r in report['per_recording']]==expected
            assert report['threshold']==plan['source_threshold']
            score_file=folder/('clean_full_scores.npz' if condition=='clean_subset' else f'{condition}_scores.npz')
            count_rows=[]
            with np.load(score_file) as z:
                for r in report['per_recording']:
                    uid=r['uid'];y=truth[uid];s=z[uid][:len(y)]
                    assert np.isfinite(s).all()
                    start=np.arange(len(y))*.02
                    active=np.maximum(0.,np.minimum(start+.02,durations[uid])-start)>0
                    y=y[active].astype(bool);p=decision_mask(s[active],plan['source_threshold'])
                    frame=[int((p&y).sum()),int((p&~y).sum()),int((~p&y).sum()),int((~p&~y).sum())]
                    count_rows.append([r[k] for k in PER]+frame)
            a=np.array(count_rows);arrays[(name,condition)]=a
            fields=['event_f1_iou05_percent','frame_f1_percent','real_duration_fpr_percent',
                    'short_event_recall_le320ms_percent','duration_f1_percent']
            np.testing.assert_allclose(stats(a.sum(0)),[report[k] for k in fields],atol=1e-8,rtol=0)
            row=dict(model=name,condition=condition,recordings=len(a),**{k:report[f] for k,f in zip(KEYS,fields)},eer=report['eer_percent'])
            rows.append(row)
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lookup={(r['model'],r['condition']):r for r in rows}
    means={}
    for arm in ['original']+['old_only','new_only','previous_replay']+plan['arms']:
        means[arm]={}
        names=['original'] if arm=='original' else [f'{seed}_{arm}' for seed in plan['seeds']]
        for condition in CONDITIONS:
            means[arm][condition]={k:float(np.mean([lookup[(name,condition)][k] for name in names])) for k in KEYS+['eer']}
    save_json(out/'means.json',means)
    # Bounded temporary memory even for all2484 clean recordings.
    boots={}
    for condition in CONDITIONS:
        n=len(arrays[('original',condition)]);rng=np.random.default_rng(20261123)
        draws=rng.integers(0,n,size=(2000,n))
        for name in frozen:
            a=arrays[(name,condition)]
            boots[(name,condition)]=np.concatenate([stats(a[draws[i:i+100]].sum(1)) for i in range(0,2000,100)])
    pairs=[(name,'original') for name in frozen if name!='original']
    pairs += [(f'{seed}_replay',f'{seed}_{other}') for seed in plan['seeds'] for other in ('new_only','old_only','previous_replay')]
    comparisons={}
    for candidate,control in pairs:
        key=f'{candidate}_vs_{control}';comparisons[key]={}
        for condition in CONDITIONS:
            delta=stats(arrays[(candidate,condition)].sum(0))-stats(arrays[(control,condition)].sum(0))
            lo,hi=np.quantile(boots[(candidate,condition)]-boots[(control,condition)],[.025,.975],axis=0)
            comparisons[key][condition]={k:dict(delta_pp=float(delta[i]),lower95=float(lo[i]),upper95=float(hi[i])) for i,k in enumerate(KEYS)}
    save_json(out/'paired_comparisons.json',dict(comparisons=comparisons,draws=2000,
        limitations='Fixed models, paired UID resampling only; no source/pretraining variance, speaker dependence or multiplicity correction.'))
    screens={}
    for arm in plan['arms']:
        screens[arm]={}
        for seed in plan['seeds']:
            d=comparisons[f'{seed}_{arm}_vs_original']['clean_full']
            checks=dict(event_f1=d['event_f1']['delta_pp']>=-.5,frame_f1=d['frame_f1']['delta_pp']>=-.5,
                        real_fpr=d['real_fpr']['delta_pp']<=.5,short_recall=d['short_recall']['delta_pp']>=-1.)
            uncertainty=dict(event_f1=d['event_f1']['lower95']>=-.5,frame_f1=d['frame_f1']['lower95']>=-.5,
                        real_fpr=d['real_fpr']['upper95']<=.5,short_recall=d['short_recall']['lower95']>=-1.)
            screens[arm][str(seed)]=dict(point_checks=checks,point_pass=all(checks.values()),
                         interval_checks=uncertainty,interval_pass=all(uncertainty.values()))
    save_json(out/'preservation_screen.json',screens)
    initial=[]
    for seed in plan['seeds']:
        for arm in plan['arms']:
            p=read_json(out/f'seed{seed}'/arm/'complete.json');assert p['updates']==128
            initial.append(p['initial_digests'])
    assert all(x==initial[0] for x in initial)
    reproduction=read_json(out/'full_source_reproduction.json')
    assert reproduction['recordings']==2484 and reproduction['max_archive_score_difference']<0.001
    archive=np.load(ROOT.parents[1]/'tmp/publish_sal_logmel_20261118/experiments/wavlm_multilayer_20261117/reports/training/fusion6_9_12/dev_scores20.npz')
    current=np.load(out/'evaluation/original/clean_full_scores.npz')
    offsets=np.r_[0,np.cumsum(archive['lengths']*8)]
    changed=0
    for i,uid in enumerate(archive['uids']):
        n=len(truth[str(uid)])
        changed+=int(np.count_nonzero(decision_mask(current[str(uid)][:n],plan['source_threshold'])!=decision_mask(archive['scores20'][offsets[i]:offsets[i]+n],plan['source_threshold'])))
    reproduction['exact_float64_decision_changes']=changed
    verification=dict(original_checkpoint_unchanged=True,all_backends_start_identical=True,all_updates_128=True,
          thresholds_never_recalibrated=True,source_full2484_reproduced_within_original_tolerance=reproduction['original_tolerance_pass'],source_full2484_threshold_decisions_identical=changed==0,source_reproduction=reproduction,
          native20_frame_and_event_counts_verified=True,all_prediction_uids_match=True,
          all_frozen_checkpoint_hashes_match=True,formal_test_accessed=False)
    save_json(out/'verification.json',verification)
    lines=['多层WavLM：3:1回放学习率5e-5与1e-4对照','',
      '所有新训练方案从同一原始多层WavLM checkpoint出发；以上一轮噪声增强3:1回放、仅噪声、仅旧样本的冻结分数为对照；源模型重新推理验证可比性。',
      '冻结WavLM；融合标量、投影、CNN-BiGRU和完整双分辨率模块更新，原五项损失不变。',
      '256条目标录音轮换15dB/5dB白噪声，256条独立UID旧样本轮换原音频/20dB噪声/EQ。',
      '对照：原模型、仅旧样本、仅噪声增强、两组均旧:新损失3:1，即0.75旧损失+0.25噪声损失；只将学习率1e-4降至5e-5。每步仍各8条，同数据、同更新次数和同算力。',
      '每种训练两个微调种子，4轮、batch8、128更新、本轮AdamW lr5e-5、对照lr1e-4；固定最后一轮和原阈值。',
      '两组学习率方案同算力；相比仅噪声方案仍额外计算旧批次。两个种子不是独立预训练模型。',
      '原性能：全部2484条dev。压力：沿用上轮384条dev，白噪声10dB/0dB、粉红噪声10dB和低通2.6kHz。',
      '10dB是未训练的中间强度，0dB是未训练的更强强度，粉红噪声是未训练的频谱类型。',
      '整个dev曾用于源模型选模；所有压力条件已在上轮查看，较小学习率是在查看前几轮回放结果后选择的后续方案，本轮为探索性重测，非新盲测。',
      '正式eval和LPS未使用。合成噪声不能代表真实未知环境或未知伪造算法。',
      '录音配对bootstrap2000次，95%区间未校正同源依赖或多重比较。',
      '保护线：事件/帧F1降幅<=0.5pp、真实时长误报增幅<=0.5pp、短事件召回降幅<=1pp。',
      '', '结果（%，更新模型为两种子均值）：',
      '方案 | 条件 | 事件F1 | 帧F1 | 真实时长误报 | 短事件召回 | EER']
    for arm,conditions in means.items():
        for condition,d in conditions.items():
            lines.append(f"{NAMES[arm]} | {condition} | {d['event_f1']:.3f} | {d['frame_f1']:.3f} | {d['real_fpr']:.3f} | {d['short_recall']:.3f} | {d['eer']:.3f}")
    lines+=['','原域保护线：',json.dumps(screens,ensure_ascii=False,indent=2),'','回放事件F1差值与95%区间：']
    for seed in plan['seeds']:
        for control in ['original',f'{seed}_new_only',f'{seed}_previous_replay']:
            for condition in CONDITIONS:
                d=comparisons[f'{seed}_replay_vs_{control}'][condition]['event_f1']
                lines.append(f"seed{seed} vs {control}, {condition}: {d['delta_pp']:+.3f} [{d['lower95']:+.3f}, {d['upper95']:+.3f}]pp")
    lines+=['','核验：',json.dumps(verification,ensure_ascii=False,indent=2),
            '原归档分数存在已知微小数值差异；本轮预定最大容差0.001，准确判定差异见核验，不声称逐位复现。',
            '原权重未修改；全部模型与阈值在评估前冻结。',
            '复现：使用D:/SAL_精简实验/code/.venv/Scripts/python.exe运行run.py，完成后运行report.py。']
    (ROOT/'实验报告.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    print(json.dumps(dict(metric_rows=len(rows),preservation=screens,verification=verification),ensure_ascii=False))

if __name__=='__main__':main()
