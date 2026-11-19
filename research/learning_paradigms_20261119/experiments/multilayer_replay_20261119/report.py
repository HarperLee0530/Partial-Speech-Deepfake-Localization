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

CONDITIONS=['clean_full','clean_subset','lp3400','lp2600','noise10']
KEYS=['event_f1','frame_f1','real_fpr','short_recall','duration_f1']
PER=['tp_seconds','fp_seconds','fn_seconds','tn_seconds','event_tp','event_pred','event_truth','short_tp','short_truth']
NAMES={'original':'原模型','old_only':'只继续训练旧条件','new_only':'只微调新条件','replay':'新旧各半回放'}


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
    for arm in ['original']+plan['arms']:
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
    pairs += [(f'{seed}_replay',f'{seed}_{other}') for seed in plan['seeds'] for other in ('new_only','old_only')]
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
    assert reproduction['recordings']==2484 and reproduction['max_archive_score_difference']<0.01
    archive=np.load(ROOT.parents[1]/'tmp/publish_sal_logmel_20261118/experiments/wavlm_multilayer_20261117/reports/training/fusion6_9_12/dev_scores20.npz')
    current=np.load(out/'evaluation/original/clean_full_scores.npz')
    offsets=np.r_[0,np.cumsum(archive['lengths']*8)]
    changed=0
    for i,uid in enumerate(archive['uids']):
        n=len(truth[str(uid)])
        changed+=int(np.count_nonzero(decision_mask(current[str(uid)][:n],plan['source_threshold'])!=decision_mask(archive['scores20'][offsets[i]:offsets[i]+n],plan['source_threshold'])))
    reproduction['exact_float64_decision_changes']=changed
    reproduction['float32_comparison_count_is_not_authoritative']=True
    verification=dict(original_checkpoint_unchanged=True,all_backends_start_identical=True,all_updates_128=True,
          thresholds_never_recalibrated=True,source_full2484_reproduced_within_original_tolerance=reproduction['original_tolerance_pass'],source_full2484_threshold_decisions_identical=changed==0,source_reproduction=reproduction,
          native20_frame_and_event_counts_verified=True,all_prediction_uids_match=True,
          all_frozen_checkpoint_hashes_match=True,formal_test_accessed=False)
    save_json(out/'verification.json',verification)
    lines=['多层WavLM融合方案：旧样本回放小型对照 — 2026-11-19','',
      '直接使用已发布fusion6/9/12第18轮最佳checkpoint。不是此前384条录音从零训练的小CNN。',
      '所有更新方案从完全相同的已训练权重出发。两个种子只改变微调顺序/dropout，不代表两个独立预训练模型。',
      '公开WavLM冻结；三个融合标量、投影、原CNN-BiGRU及完整双分辨率模块均更新。原五项训练损失保持不变。',
      '原始logit阈值-0.17905920743942258在所有模型、条件中固定，完全没有重新校准。',
      '', '设计：',
      '原模型不更新；old_only只继续训练旧条件；new_only只微调新条件；replay新旧损失各半。',
      '新条件是3.4kHz低通。256条目标训练录音、256条独立UID旧记忆录音；原模型训练时已见过这512条录音的原条件。',
      '旧记忆轮换原音频、20dB噪声、轻微EQ。4轮、batch8、128次AdamW更新、lr1e-4。固定最后一轮，无选模。',
      'new_only和replay的目标数据顺序相同；old_only和replay的旧数据顺序/视图相同。回放额外计算旧数据，非等算力对照。',
      '保持原性能：全部2484条dev原音频。适应/泛化检查：另外固定384条dev的3.4kHz低通、未训练2.6kHz低通及未训练10dB噪声。',
      '384条压力检查UID避开了此前两轮课件小实验的校准/留出UID，但整个官方dev曾用于原模型选择，不能称为全新盲测。',
      '评估修订：原始1e-4分数复现门槛触发；保留训练和阈值不变，另行记录全量最大分数误差和float64阈值判定差异，见核验。',
      '正式eval/LPS没有使用；因果低通未补偿群延迟；合成扰动不等于新伪造方法/真实新数据集。',
      '', '预定原域保护线：事件F1、帧F1最多下降0.5pp；真实时长误报最多增加0.5pp；短事件召回最多下降1pp。',
      '分别报告点估计和配对95%区间是否通过，点估计通过不能证明非劣。',
      '', '结果（%）；更新模型为两个微调种子的均值，原模型为一个固定checkpoint：',
      '方案 | 条件 | 事件F1 | 帧F1 | 真实时长误报率 | 短事件召回 | EER']
    for arm,conditions in means.items():
        for condition,d in conditions.items():
            lines.append(f"{NAMES[arm]} | {condition} | {d['event_f1']:.3f} | {d['frame_f1']:.3f} | {d['real_fpr']:.3f} | {d['short_recall']:.3f} | {d['eer']:.3f}")
    lines+=['','原域保护线：',json.dumps(screens,ensure_ascii=False,indent=2),'',
            '回放相对原模型：各微调种子的事件F1差值与配对录音bootstrap95%区间（pp）：']
    for seed in plan['seeds']:
        for condition in CONDITIONS:
            d=comparisons[f'{seed}_replay_vs_original'][condition]['event_f1']
            lines.append(f"seed{seed} {condition}: {d['delta_pp']:+.3f} [{d['lower95']:+.3f}, {d['upper95']:+.3f}]")
    lines+=['','核验：',json.dumps(verification,ensure_ascii=False,indent=2),
       '', '解释限制：同一个原模型、两个微调种子、单个记忆池、固定1:1配比和学习率；不包含完整重训随机性。',
       '配对bootstrap仅按录音重采样，未修正同源依赖与多重比较。性能保持和未见条件收益应分别判断。',
       '只改善训练时见过的3.4kHz条件，只能称目标条件适应，不能直接称广泛泛化增强。',
       '', '代码run.py、完整结果results/summary.csv、区间paired_comparisons.json、保护线preservation_screen.json。',
       '模型在results/seed*/{old_only,new_only,replay}/last.pt，原权重没有替换。',
       '特征缓存位于D:/SAL_精简实验/data/multilayer_replay_pilot_20261119，保留float32三层特征，不压成固定融合特征。',
       '复现：run.py保存原始训练协议；由于归档分数1e-4复现门槛触发，已有冻结模型使用evaluate_resume.py评估，再运行report.py。评估修订见results/evaluation_amendment.json。']
    (ROOT/'实验报告.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    print(json.dumps(dict(metric_rows=len(rows),preservation=screens,verification=verification),ensure_ascii=False))


if __name__=='__main__':main()
