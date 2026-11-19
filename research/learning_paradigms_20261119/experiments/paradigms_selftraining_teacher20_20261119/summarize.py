"""Posthoc reporting; cannot change models, thresholds, or experiment protocol."""
from pathlib import Path
import csv
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from cnn_ps.io import read_json,save_json,sha256

def stats(a):
    tp,fp,fn,tn,matched,pred,truth,short,shorttruth=a
    safe=lambda x,y:100*x/y if y else 0.
    return np.array([safe(2*matched,pred+truth),safe(2*tp,2*tp+fp+fn),
                     safe(fp,fp+tn),safe(short,shorttruth)])

def main():
    out=ROOT/'results'
    assert read_json(out/'status.json')['status']=='complete'
    plan=read_json(out/'plan.json')
    frozen=read_json(out/'frozen_thresholds.json')
    assert all(all(v.values()) for v in read_json(out/'cache_integrity.json').values())
    results=read_json(out/'comparison.json')['results']
    rows=[];intervals={}
    for seed in plan['seeds']:
        folder=out/f'seed{seed}'
        for arm in plan['arms']:
            checkpoint=folder/arm/'last.pt'
            assert sha256(checkpoint)==frozen[f'{seed}_{arm}']['checkpoint_sha256']
            history=read_json(folder/arm/'history.json')
            assert len(history)==plan['branch_epochs'] and history[-1]['updates']==192
            for condition in ('clean','noise20db'):
                r=results[f'{seed}_{arm}'][condition]
                rows.append(dict(seed=seed,arm=arm,condition=condition,
                    frame_f1=r['frame_f1_percent'],event_f1=r['event_f1_iou05_percent'],
                    real_fpr=r['real_duration_fpr_percent'],short_recall=r['short_event_recall_le320ms_percent'],
                    eer=r['eer_percent'],threshold=r['threshold']))
        for condition in ('clean','noise20db'):
            arrays=[]
            for arm in plan['arms']:
                report=read_json(folder/arm/f'{condition}_metrics.json')
                per=report['per_recording']
                assert [r['uid'] for r in per]==plan['uids']['audit']
                arrays.append(np.array([[r[k] for k in ('tp_seconds','fp_seconds','fn_seconds','tn_seconds',
                                  'event_tp','event_pred','event_truth','short_tp','short_truth')] for r in per]))
                recomputed=stats(arrays[-1].sum(0))
                expected=np.array([report[k] for k in ('event_f1_iou05_percent','duration_f1_percent',
                                      'real_duration_fpr_percent','short_event_recall_le320ms_percent')])
                np.testing.assert_allclose(recomputed,expected,atol=1e-8,rtol=0)
                assert report['threshold']==frozen[f'{seed}_{arm}']['threshold']
                with np.load(folder/arm/f'{condition}_scores.npz') as archive:
                    assert archive.files==plan['uids']['audit']
                    assert all(np.isfinite(archive[uid]).all() for uid in archive.files)
            rng=np.random.default_rng(20261119)
            draws=[]
            for _ in range(2000):
                ids=rng.integers(0,len(arrays[0]),len(arrays[0]))
                draws.append(stats(arrays[1][ids].sum(0))-stats(arrays[0][ids].sum(0)))
            diff=stats(arrays[1].sum(0))-stats(arrays[0].sum(0))
            ci=np.quantile(draws,[.025,.975],axis=0)
            intervals[f'{seed}_{condition}']={k:dict(delta_pp=float(diff[i]),lower95=float(ci[0,i]),upper95=float(ci[1,i]))
                for i,k in enumerate(('event_f1','duration_f1','real_fpr','short_recall'))}
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save_json(out/'paired_bootstrap.json',dict(comparisons=intervals,draws=2000,
        scope='Paired UID bootstrap on fixed models, within each seed. Does not include training variance, previous dev reuse, or source dependence.'))
    means={}
    for condition in ('clean','noise20db'):
        means[condition]={}
        for arm in plan['arms']:
            subset=[r for r in rows if r['condition']==condition and r['arm']==arm]
            means[condition][arm]={key:float(np.mean([r[key] for r in subset]))
                                   for key in ('frame_f1','event_f1','real_fpr','short_recall','eer')}
    save_json(out/'mean_results.json',means)
    lines=['课件自训练小型对照实验 — 2026-11-19','',
        '问题：在384条标注录音的限制下，增加768条隐藏标签训练录音的伪标签，是否改善CNN-BiGRU的定位表现？',
        '依据：课件第5—7页Self-Training。双视图筛选和预测边界排除是针对定位任务的实验设计。',
        '本次没有使用正式eval/LPS数据，没有修改或替换原项目模型。',
        '', '协议：',
        f"从随机初始化训练相同911,521参数CNN-BiGRU；两个种子666/667；{plan['warm_epochs']}轮监督热身。",
        '每个种子从同一热身权重分叉：纯监督4轮，半监督4轮。相同有标签顺序、增强、dropout种子和192次更新。',
        '两组均使用原音频/20dB噪声/轻微EQ轮换。半监督每次额外使用16条未标注音频，计算量更大。',
        '伪标签由固定热身教师给出；原音频与EQ预测同类且各自置信度>=0.9；排除任一预测边界前后2帧。',
        '有标签数据保留20ms和160ms真值监督；额外伪标签损失只在被选中20ms帧计算，真假类别均衡，权重0.3。',
        '固定最终轮，不用开发成绩选epoch。256条开发录音单独校准真实时长FPR<=5%的阈值。',
        '全部模型和阈值冻结后，在另外384条开发录音上报告clean和20dB波形噪声表现；噪声沿用clean阈值。',
        '帧F1和事件F1均采用原生20ms口径；事件一对一IoU>=0.5；短事件<=320ms。',
        '', '两个种子的平均结果（百分比；均值不能代表训练稳定性）：']
    for condition,arms in means.items():
        lines.append(condition)
        for arm,m in arms.items():
            lines.append(f"  {arm}: 帧F1 {m['frame_f1']:.3f}; 事件F1 {m['event_f1']:.3f}; 真实时长FPR {m['real_fpr']:.3f}; 短事件召回 {m['short_recall']:.3f}; EER {m['eer']:.3f}")
        a,b=arms['supervised'],arms['selftraining']
        lines.append(f"  半监督-监督: 帧F1 {b['frame_f1']-a['frame_f1']:+.3f}pp; 事件F1 {b['event_f1']-a['event_f1']:+.3f}pp; FPR {b['real_fpr']-a['real_fpr']:+.3f}pp")
    lines+=['','各个种子的事件F1差值与固定模型配对录音bootstrap95%区间：']
    for name,d in intervals.items():
        x=d['event_f1'];lines.append(f"{name}: {x['delta_pp']:+.3f}pp [{x['lower95']:+.3f}, {x['upper95']:+.3f}]")
    lines+=['','伪标签事后诊断（全部训练完成后才读取对应真值，不回灌训练）：']
    for seed in plan['seeds']:
        sel=read_json(out/f'seed{seed}'/'pseudo_selection.json')
        quality=read_json(out/f'seed{seed}'/'pseudo_quality_posthoc.json')
        lines.append(f"seed{seed}: 有效时长覆盖 {sel['coverage_percent']:.3f}%; {json.dumps(quality,ensure_ascii=False)}")
    deltas=[intervals[f'{seed}_clean']['event_f1']['delta_pp'] for seed in plan['seeds']]
    positive_intervals=all(intervals[f'{seed}_clean']['event_f1']['lower95']>0 for seed in plan['seeds'])
    decision=('两个种子clean事件F1有一致正向信号；仍需结合误报和短事件指标扩大验证，不能证明跨域泛化。' if min(deltas)>0 and positive_intervals else
              '两个种子clean事件F1仅有正向点估计，配对区间尚不足以支持一致改善；不建议据此替换主模型。' if min(deltas)>0 else
              '两个种子clean事件F1都未提升，本轮不支持将该伪标签配方加入主方案。' if max(deltas)<=0 else
              '两个种子clean事件F1方向不一致，本轮收益不稳定，不建议加入主方案。')
    lines+=['','结论：'+decision,'','限制：',*plan['limitations'],
        '20dB噪声与训练增强同类，因此只证明/否定该扰动条件的稳健性，不是新合成器或新数据集泛化。',
        '这里的半监督用隐藏已有标签模拟少标注；不意味着优于利用全部真实标签训练的当前正式模型。',
        '两个随机种子、同一小型划分；录音bootstrap不包含来源相关性、训练随机性或历史开发集复用。',
        '', '复现：',r'D:/SAL_精简实验/code/.venv/Scripts/python.exe -X utf8 run_experiment.py',
        r'D:/SAL_精简实验/code/.venv/Scripts/python.exe -X utf8 summarize.py',
        'run_experiment.py校验已冻结协议；已完成的训练阶段自动跳过。依赖本机只读特征缓存。']
    (ROOT/'实验报告.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    save_json(out/'verification.json',dict(cache_hashes_pass=True,checkpoints_match_frozen=True,
        update_counts_match=True,audit_uid_order_match=True,thresholds_not_refit_on_noise=True,
        tests=read_json(out/'checks.json'),decision=decision))
    print(json.dumps(dict(means=means,decision=decision),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
