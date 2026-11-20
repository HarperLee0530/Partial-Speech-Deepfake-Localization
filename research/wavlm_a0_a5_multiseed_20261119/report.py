"""Summarize three paired training seeds without selecting a best seed or retuning."""
from pathlib import Path
import csv,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from cnn_ps.io import read_json,save_json,sha256
from report_multilayer_experiment import paired_bootstrap
ARMS=['A0_full','A5_fixed_pool'];SEEDS=[666,667,668]
KEYS=['event_f1_iou05_percent','event_precision_percent','event_recall_percent','real_duration_fpr_percent','short_event_recall_le320ms_percent','frame_f1_percent','eer_percent']

def main():
    out=ROOT/'results';assert read_json(out/'status.json')['stage']=='complete'
    plan=read_json(out/'plan.json')
    for rel,digest in plan['source_sha256'].items():assert sha256(ROOT/rel)==digest,rel
    rows=[];records={};metrics={};paired={}
    for seed in SEEDS:
        folders={arm:(ROOT/'reference/seed666'/arm if seed==666 else out/f'seed{seed}'/arm) for arm in ARMS}
        protocols={arm:read_json(folders[arm]/'protocol.json') for arm in ARMS}
        for key in ['initial_backbone_sha256','initial_projection_sha256','initial_adapter_sha256','train_uids_sha256','dev_uids_sha256','learning_rate','batch_size','epochs','seed']:
            if seed==666 and key in ['initial_projection_sha256','initial_adapter_sha256']:
                # Reused historical A0 protocol predates these two audit fields.
                # New paired runs must contain and match both fields.
                continue
            assert protocols[ARMS[0]][key]==protocols[ARMS[1]][key],(seed,key)
        for arm,folder in folders.items():
            c=read_json(folder/'complete.json');d=read_json(folder/'dev_metrics.json');m=read_json(folder/'eval_metrics.json');r=read_json(folder/'eval_per_recording.json')
            assert sha256(folder/'best.pt')==c['checkpoint_sha256']
            assert c['updates']==15880 and c['trained_epochs']==20 and protocols[arm]['seed']==seed
            assert c['threshold']==d['threshold']==m['threshold']
            total=lambda key:sum(x[key] for x in r)
            calc=[200*total('event_tp')/(total('event_pred')+total('event_truth')),100*total('fp_seconds')/(total('fp_seconds')+total('tn_seconds')),100*total('short_tp')/total('short_truth')]
            np.testing.assert_allclose(calc,[m[k] for k in ['event_f1_iou05_percent','real_duration_fpr_percent','short_event_recall_le320ms_percent']],atol=1e-8,rtol=0)
            records[(seed,arm)]=r;metrics[(seed,arm)]=m
            rows.append(dict(seed=seed,arm=arm,epoch=c['epoch'],threshold=c['threshold'],dev_event_f1=d['event_f1_iou05_percent'],**{k:m[k] for k in KEYS}))
        pair=paired_bootstrap(records[(seed,ARMS[0])],records[(seed,ARMS[1])],repetitions=2000,seed=20261120)
        pair['difference']='A5_fixed_pool minus A0_full, percentage points'
        paired[str(seed)]=pair
    uids=[r['uid'] for r in records[(666,'A0_full')]]
    assert len(uids)==3562
    for r in records.values():assert [x['uid'] for x in r]==uids
    split_uids={split:set(read_json(Path(plan['data_location'])/split/'complete.json')['uids']) for split in ['train','dev','eval']}
    assert set(uids)==split_uids['eval']
    for left,right in [('train','dev'),('train','eval'),('dev','eval')]:
        assert not (split_uids[left]&split_uids[right]),(left,right,'UID overlap')
    save_json(out/'split_integrity.json',dict(recordings={k:len(v) for k,v in split_uids.items()},pairwise_uid_disjoint=True))
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={arm:{k:dict(mean=float(np.mean([metrics[(s,arm)][k] for s in SEEDS])),sample_sd=float(np.std([metrics[(s,arm)][k] for s in SEEDS],ddof=1)),per_seed=[metrics[(s,arm)][k] for s in SEEDS]) for k in KEYS} for arm in ARMS}
    deltas={k:dict(mean=float(np.mean([metrics[(s,ARMS[1])][k]-metrics[(s,ARMS[0])][k] for s in SEEDS])),sample_sd=float(np.std([metrics[(s,ARMS[1])][k]-metrics[(s,ARMS[0])][k] for s in SEEDS],ddof=1)),per_seed=[metrics[(s,ARMS[1])][k]-metrics[(s,ARMS[0])][k] for s in SEEDS]) for k in KEYS}
    save_json(out/'three_seed_summary.json',dict(arms=summary,paired_deltas=deltas,seeds=SEEDS,historical_seed666_reused=True,not_best_seed_selection=True))
    save_json(out/'paired_bootstrap.json',dict(per_seed=paired,caveat='Intervals condition on each fixed trained pair and resample recordings, not speakers; no multiplicity correction. Three training seeds are insufficient for strong training-distribution inference. Do not pool three predictions per recording as independent observations.'))
    reference_guard={str(s):dict(event_f1_delta=paired[str(s)]['metrics']['event_f1_iou05_percent']['delta_pp'],fpr_delta=paired[str(s)]['metrics']['real_duration_fpr_percent']['delta_pp'],fpr_increase_le_half_pp=paired[str(s)]['metrics']['real_duration_fpr_percent']['delta_pp']<=.5) for s in SEEDS}
    save_json(out/'reference_fpr_guard.json',reference_guard)
    lines=['A0/A5多训练种子对照：seed666历史结果＋seed667/668完整补跑','',
      '源代码固定于GitHub483c0252cf8b7104376399008d21d414fdced0b6，未改变架构或训练配方。',
      'A5唯一改变是固定有效时长平均池化；保留可学习层融合、注意力、门控、边界、一致性和粗辅助损失。',
      '每组6345训练/2484开发，20轮、batch8、15880更新，AdamW lr0.001、wd0.0001、clip5。同seed内公共参数初始化、数据顺序和增强分配相同。',
      '每轮dev真实时长FPR<=5%校准，再按dev事件F1选轮次，epoch0可选、平分取早。四次训练全部完成并冻结后才打开eval特征；不在测试集重新调阈值或选轮次。',
      'eval3562条已被历史项目查看；seed666及A5选择参考历史结果，此次是配对多种子复核，不是全新盲测。未使用LPS。',
      '本机RTX4060，历史seed666来自RTX4080；重新生成完整float32特征，并先验证两历史模型的全dev归档分数。数值差异见source_cache_check.json。',
      '历史A0协议未记录投影/适配器初始化哈希，无法用该归档字段核验；新增667/668两组的全部公共初始化哈希均已核验一致。',
      '', '逐seed结果：', 'seed | arm | selected epoch | eventF1 | real durationFPR | short recall | frameF1 | EER']
    for r in rows:lines.append(f"{r['seed']} | {r['arm']} | {r['epoch']} | {r[KEYS[0]]:.4f} | {r[KEYS[3]]:.4f} | {r[KEYS[4]]:.4f} | {r[KEYS[5]]:.4f} | {r[KEYS[6]]:.4f}")
    lines+=['','三seed均值±样本标准差（%，不挑最佳seed）：']
    for arm in ARMS:
        lines.append(arm+': '+', '.join(f"{k}={v['mean']:.4f}±{v['sample_sd']:.4f}" for k,v in summary[arm].items()))
    lines+=['','A5-A0逐seed配对差值及按录音95%区间（pp）：']
    for seed,p in paired.items():
        for k,v in p['metrics'].items():lines.append(f"seed{seed} {k}: {v['delta_pp']:+.4f} [{v['ci95_pp'][0]:+.4f}, {v['ci95_pp'][1]:+.4f}]")
    lines+=['','原先误报+0.5pp参考线（仅作为取舍报告，不修改训练选择）：',json.dumps(reference_guard,ensure_ascii=False,indent=2),
       '', '不能把逐录音bootstrap区间当作包含训练随机性的区间，不能把同一录音的三份预测当作三倍独立样本。三个训练seed只能提供有限的稳定性证据。',
       '原方案、权重和GitHub默认分支均未修改。完整数值见summary.csv、three_seed_summary.json、paired_bootstrap.json。']
    (ROOT/'实验报告.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    print(json.dumps(dict(means=summary,paired_deltas=deltas,fpr_guard=reference_guard),ensure_ascii=False))

if __name__=='__main__':main()
