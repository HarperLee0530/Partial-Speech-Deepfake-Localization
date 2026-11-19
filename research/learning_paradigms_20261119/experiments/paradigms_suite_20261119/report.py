"""Verify and summarize the frozen suite without feeding outcomes into training."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from cnn_ps.io import read_json,save_json,sha256

NAMES={
 'mel_control':'CNN纯监督', 'selftraining_reference':'此前自训练', 'multitask':'多任务边界监督',
 'wavlm_control':'WavLM迁移', 'cotraining_mel':'协同训练-Mel', 'cotraining_wavlm':'协同训练-WavLM',
 'coral':'CORAL域适应','continual_source':'持续学习-源模型','continual_naive':'持续学习-直接微调',
 'continual_ewc':'持续学习-EWC','continual_replay':'持续学习-回放',
 'meta_erm':'元学习对照ERM','meta_fomaml':'一阶MAML','meta_erm_adapt':'ERM适应后','meta_fomaml_adapt':'一阶MAML适应后'}
PAIRS=[('selftraining_reference','mel_control'),('multitask','mel_control'),('wavlm_control','mel_control'),
       ('cotraining_mel','mel_control'),('cotraining_wavlm','wavlm_control'),('coral','mel_control'),
       ('continual_naive','continual_source'),('continual_ewc','continual_naive'),('continual_replay','continual_naive'),
       ('meta_fomaml','meta_erm'),('meta_erm_adapt','meta_erm'),('meta_fomaml_adapt','meta_fomaml'),
       ('meta_fomaml_adapt','meta_erm_adapt')]
FIELDS=('tp_seconds','fp_seconds','fn_seconds','tn_seconds','event_tp','event_pred','event_truth','short_tp','short_truth')
METRICS=('event_f1','duration_f1','real_fpr','short_recall')

def stats(a):
    tp,fp,fn,tn,matched,pred,truth,short,shorttruth=np.moveaxis(a,-1,0)
    safe=lambda x,y:100*np.divide(x,y,out=np.zeros_like(x,dtype=float),where=y!=0)
    return np.stack([safe(2*matched,pred+truth),safe(2*tp,2*tp+fp+fn),safe(fp,fp+tn),safe(short,shorttruth)],axis=-1)

def main():
    out=ROOT/'results';plan=read_json(out/'plan.json');frozen=read_json(out/'frozen_thresholds.json')
    assert read_json(out/'status.json')['status']=='complete'
    assert all(read_json(out/'wavlm_cache_integrity.json').values())
    rows=[];arrays={}
    for seed in plan['seeds']:
        erm=read_json(out/f'seed{seed}'/'meta_erm'/'complete.json')
        maml=read_json(out/f'seed{seed}'/'meta_fomaml'/'complete.json')
        assert erm['initial_digest']==maml['initial_digest']
        assert erm['updates']==maml['updates']==200
        assert read_json(out/f'seed{seed}'/'wavlm_warm'/'complete.json')['updates']==960
        for arm in plan['arms']:
            key=f'{seed}_{arm}';folder=out/f'seed{seed}'/arm
            assert sha256(folder/'last.pt')==frozen[key]['checkpoint_sha256']
            expected_source=(f'{seed}_continual_source' if arm.startswith('continual_') and arm!='continual_source'
                             else f'{seed}_{arm[:-6]}' if arm.endswith('_adapt') else key)
            assert frozen[key]['threshold_source']==expected_source
            assert frozen[key]['threshold']==frozen[frozen[key]['threshold_source']]['threshold']
            if arm in ('multitask','wavlm_control','cotraining_mel','cotraining_wavlm','coral','continual_naive','continual_ewc','continual_replay'):
                assert read_json(folder/'complete.json')['updates']==192
            if arm.endswith('_adapt'):assert read_json(folder/'complete.json')['updates']==5
            for condition in ('clean','band'):
                report=read_json(folder/f'{condition}_metrics.json')
                assert report['threshold']==frozen[key]['threshold']
                per=report['per_recording'];assert [r['uid'] for r in per]==plan['uids']['audit']
                a=np.array([[r[k] for k in FIELDS] for r in per]);arrays[(seed,arm,condition)]=a
                expected=[report[k] for k in ('event_f1_iou05_percent','duration_f1_percent','real_duration_fpr_percent','short_event_recall_le320ms_percent')]
                np.testing.assert_allclose(stats(a.sum(0)),expected,rtol=0,atol=1e-8)
                with np.load(folder/f'{condition}_scores.npz') as z:
                    assert z.files==plan['uids']['audit'] and all(np.isfinite(z[uid]).all() for uid in z.files)
                rows.append(dict(seed=seed,arm=arm,name=NAMES[arm],condition=condition,
                     frame_f1=report['frame_f1_percent'],event_f1=report['event_f1_iou05_percent'],
                     real_fpr=report['real_duration_fpr_percent'],short_recall=report['short_event_recall_le320ms_percent'],
                     eer=report['eer_percent'],threshold=report['threshold']))
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    means={}
    for arm in plan['arms']:
        means[arm]={}
        for condition in ('clean','band'):
            r=[x for x in rows if x['arm']==arm and x['condition']==condition]
            means[arm][condition]={k:float(np.mean([x[k] for x in r])) for k in ('frame_f1','event_f1','real_fpr','short_recall','eer')}
    save_json(out/'means.json',means)
    rng=np.random.default_rng(20261121);draws=rng.integers(0,384,size=(2000,384))
    boot={key:stats(a[draws].sum(1)) for key,a in arrays.items()}
    comparisons={}
    for candidate,control in PAIRS:
        key=f'{candidate}_vs_{control}';comparisons[key]={}
        for seed in plan['seeds']:
            comparisons[key][str(seed)]={}
            for condition in ('clean','band'):
                x=(seed,candidate,condition);y=(seed,control,condition)
                delta=stats(arrays[x].sum(0))-stats(arrays[y].sum(0))
                ci=np.quantile(boot[x]-boot[y],[.025,.975],axis=0)
                comparisons[key][str(seed)][condition]={m:dict(delta_pp=float(delta[i]),lower95=float(ci[0,i]),upper95=float(ci[1,i])) for i,m in enumerate(METRICS)}
    save_json(out/'paired_comparisons.json',dict(comparisons=comparisons,bootstrap_draws=2000,
         caveat='Exploratory, many comparisons, no multiplicity correction; fixed models only, does not include training variance, historical dev reuse or source dependence.'))
    verification=dict(all_checkpoints_match_frozen=True,all_thresholds_match_declared_sources=True,
         audit_uids_aligned=True,metrics_recomputed_from_recording_counts=True,finite_scores=True,
         formal_test_accessed=False,original_weights_unchanged=True)
    extra=read_json(out/'extra_checks.json')
    assert all(all(v.values()) for v in extra['mel_integrity'].values())
    assert extra['boundary_padding_gradient_zero'] and extra['boundary_valid_gradient_nonzero']
    assert read_json(out/'encoder_provenance_check.json')['wavlm_train_dev_encoder_and_grid_match']
    assert all(v<1e-4 for v in read_json(out/'batch_inference_check.json').values())
    assert read_json(out/'fisher_backend_check.json')['fisher_gradients_finite_nonzero']
    for name,digest in read_json(out/'band_cache/complete.json')['files'].items():
        assert sha256(out/'band_cache'/name)==digest
    assert sha256(ROOT/'suite.py')==plan['source_sha256']['suite.py']
    verification.update(mel_and_wavlm_cache_integrity=True,band_cache_integrity=True,
           boundary_padding_gradients=True,batch_single_inference_consistency=True,fisher_gradients_valid=True,
           meta_initializations_and_update_counts_matched=True)
    old=ROOT.parent/'paradigms_selftraining_teacher20_20261119/results'
    for seed in plan['seeds']:
        for arm in ('warm','supervised','selftraining'):
            assert sha256(old/f'seed{seed}'/arm/'last.pt')==plan['frozen_old_inputs'][f'{seed}_{arm}']
    save_json(out/'verification.json',verification)
    # Reveal hidden training labels only after every model/threshold is frozen.
    cache=Path('D:/SAL_精简实验/data/cnn_ps_features_v2/train')
    metadata=read_json(cache/'complete.json')
    truth=np.load(cache/'labels20.npy',mmap_mode='r');weights=np.load(cache/'weights20.npy',mmap_mode='r')
    pseudo_quality={}
    for seed in plan['seeds']:
        selected=correct=0.;class_seconds=[0.,0.];class_correct=[0.,0.]
        with np.load(out/f'seed{seed}'/'consensus.npz') as z:
            for i in plan['splits']['unlabeled']:
                uid=metadata['uids'][i];a,b=metadata['offsets'][i:i+2]
                pred,mask=z[uid];w=weights[a:b]*mask;y=truth[a:b]
                selected+=float(w.sum());correct+=float(w[pred==y].sum())
                for k in (0,1):
                    class_seconds[k]+=float(w[pred==k].sum());class_correct[k]+=float(w[(pred==k)&(y==k)].sum())
        pseudo_quality[str(seed)]=dict(selected_seconds=selected,
            selected_duration_accuracy_percent=100*correct/selected if selected else None,
            real_precision_percent=100*class_correct[0]/class_seconds[0] if class_seconds[0] else None,
            fake_precision_percent=100*class_correct[1]/class_seconds[1] if class_seconds[1] else None,
            diagnostic_only_after_training=True)
    save_json(out/'consensus_quality_posthoc.json',pseudo_quality)
    lines=['其他学习范式：六类方法小型筛选 — 2026-11-19','',
       '本轮覆盖课件的迁移学习、多任务学习、协同训练、领域适应、持续学习与元学习；领域适应属于迁移学习子类。',
       '每个方案均运行两个随机种子666/667，固定最终训练轮次，不按留出结果挑选模型。',
       '384条标注训练、768条隐藏标签训练、256条校准开发、384条留出开发。元学习另外使用32条独立训练录音的标签作目标任务支持集。',
       '本轮校准/留出UID均与前一轮半监督实验的640条开发UID互斥，但整个官方开发集在历史项目中用过。',
       '正式eval和LPS未读取，原模型不替换。仅UID互斥，没有证明说话人或源录音独立。',
       '', '评价条件：',
       'clean=原音频；band=6阶Butterworth因果3.4kHz低通，采样率和时长不变。未补偿滤波群延迟。',
       'band是人工构造的通道变化，不代表新伪造算法或新真实数据集。',
       '迁移、多任务、协同组没有在下游训练中见过band。CORAL用band未标注训练音频；持续学习用384条band标签；元学习适应用32条band标签。',
       '这些目标域信息权限不同，不能把全部方法当成同一公平排行榜。',
       '通用WavLM的预训练包含额外数据和参数；与Log-Mel的差异属于系统级比较，不是单因素预训练消融。',
       '', '方法和对照：',
       '迁移：冻结通用WavLM + 768到80维可学习投影 + 原CNN-BiGRU。教师20轮、后续4轮，与CNN监督训练预算对应。',
       '多任务：复用CNN教师20轮，追加4轮真假监督+0.1边界辅助损失，边界正类权重10。推理不使用辅助边界头修正结果。',
       '协同：Mel/WavLM两个教师共同筛选高置信度一致伪标签，再分别训练两个学生4轮；独立视角假设未证明，只称协同训练启发方案。',
       '域适应：原监督损失+100倍Deep CORAL协方差对齐，额外输入clean源特征和band未标注目标特征。',
       '持续学习：相同CNN源模型，追加4轮band训练；直接微调、EWC(lambda1000，64条旧录音估计Fisher)、回放(目标/clean损失各半)。',
       'EWC以每条clean旧录音的20ms+160ms监督损失梯度平方平均，作为经验Fisher近似；没有搜索正则强度。',
       '元学习：另外一对完全相同WavLM小型逐帧MLP。ERM与一阶MAML各200个任务episode，任务为clean/noise/EQ；支持8条、查询8条且互斥。',
       '一阶MAML一步内循环SGD0.01，外循环AdamW0.001；不是二阶MAML。两组随后均用相同32条band支持数据做5步SGD0.01适应。',
       '持续学习全部沿用源模型clean阈值；元学习适应后沿用各自适应前clean阈值；其余方法各自在clean校准。band上不重调阈值。',
       '适应前后可能出现误报变化，必须与F1一起解释。回放、协同、CORAL有额外前向计算，未声称严格等算力。',
       '', '两个种子的均值（%）：',
       '方法 | 条件 | 帧F1 | 事件F1 | 真实时长误报率 | <=320ms短事件召回 | EER']
    for arm,conditions in means.items():
        for condition,m in conditions.items():
            lines.append(f"{NAMES[arm]} | {condition} | {m['frame_f1']:.2f} | {m['event_f1']:.2f} | {m['real_fpr']:.2f} | {m['short_recall']:.2f} | {m['eer']:.2f}")
    lines+=['','预定配对比较：以下为各种子的事件F1变化及配对录音bootstrap95%区间，单位百分点。']
    for (candidate,control) in PAIRS:
        lines.append(f'{NAMES[candidate]} 相对 {NAMES[control]}')
        for seed in plan['seeds']:
            for condition in ('clean','band'):
                r=comparisons[f'{candidate}_vs_{control}'][str(seed)][condition]['event_f1']
                lines.append(f"  seed{seed} {condition}: {r['delta_pp']:+.2f} [{r['lower95']:+.2f}, {r['upper95']:+.2f}]")
    lines+=['','协同伪标签事后诊断（不反馈到训练）：',json.dumps(pseudo_quality,ensure_ascii=False,indent=2),'','解释边界：',
        '这是多个方法的探索性筛选；未做多重比较校正，不能据某一个正向置信区间宣称方法已被证明。',
        '两个种子不包含充分的训练随机性估计；bootstrap仅对固定模型按录音重采样，未处理同源依赖。',
        '持续学习只测合成通道的旧任务保持，元学习仅有三个人工任务条件；两者均不是完整真实任务分布的验证。',
        '一个固定超参数配方失败不能否定整个学习范式。',
        '', '核验：',json.dumps(verification,ensure_ascii=False),
        '', '文件：results/summary.csv为全部种子结果；means.json为均值；paired_comparisons.json为完整配对区间。',
        'results/plan.json记录先验协议与源文件hash；frozen_thresholds.json记录模型hash和阈值来源。',
        '每个seed目录保留模型、损失记录、逐录音指标和预测。原始输入缓存只读。',
        '', '复现：使用D:/SAL_精简实验/code/.venv/Scripts/python.exe运行suite.py，再运行report.py。',
        '代码依赖本机只读Log-Mel/WavLM缓存和上一轮warm20权重；并非无依赖的独立发布包。',
        '', '扩展方法原始文献：',*plan['references'].values()]
    (ROOT/'实验报告.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    print(json.dumps(dict(metric_rows=len(rows),paired_comparisons=len(PAIRS),
                          report=str(ROOT/'实验报告.txt'),verification=verification),ensure_ascii=False))

if __name__=='__main__':main()
