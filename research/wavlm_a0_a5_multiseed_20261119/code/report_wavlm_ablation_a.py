"""Audit and report all eleven predeclared arms, including paired intervals."""
from pathlib import Path
import csv
import numpy as np
import torch
from cnn_ps.io import read_json, save_json, sha256
from cnn_ps.wavlm_ablation_a import ARMS, loss_weights
from report_multilayer_experiment import paired_bootstrap, METRICS

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]/'handoff_audit/ablation_a_20261118'
DESCRIPTIONS = {
    'A0_full': '完整方案：6/9/12层可学习融合，全部模块开启',
    'A1_layer12': '仅第12层', 'A2_layer6': '仅第6层', 'A3_layer9': '仅第9层',
    'A4_mean': '6/9/12层固定等权平均', 'A5_fixed_pool': '可学习降采样改为按有效时长平均',
    'A6_no_attention': '关闭局部跨分辨率注意力残差，保留粗分支和门控',
    'A7_fixed_gate': '可学习门控改为固定0.5，保留注意力',
    'A8_no_boundary': '关闭边界辅助头及其损失，保留一致性',
    'A9_no_consistency': '关闭段内一致性损失，保留边界头',
    'A10_no_coarse_aux': '关闭粗分支辅助损失，保留粗分支和主160ms损失',
}


def report():
    comparison = read_json(ROOT/'all_comparison.json')
    if comparison['status'] != 'complete' or set(comparison['models']) != set(ARMS):
        raise ValueError('All eleven arms must have complete evaluation')
    models = comparison['models']
    benchmark = read_json(ROOT/'inference_benchmark.json')
    inference = read_json(ROOT/'single_audio_verification.json')
    if benchmark['status'] != 'complete' or inference['status'] != 'passed':
        raise ValueError('Inference verification must pass')
    changed = [p for p, digest in read_json(ROOT/'existing_code_sha256.json').items()
               if sha256(CODE.parent/p) != digest]
    if changed:
        raise ValueError(f'Pre-existing code changed: {changed}')
    payloads, records = {}, {}
    for arm in ARMS:
        folder = ROOT/'training'/arm
        complete = comparison['frozen']['arms'][arm]['complete']
        for filename, key in [('best.pt', 'checkpoint_sha256'), ('protocol.json', 'protocol_sha256'),
                              ('dev_scores20.npz', 'dev_scores20_sha256'), ('dev_metrics.json', 'dev_metrics_sha256')]:
            if sha256(folder/filename) != complete[key]:
                raise ValueError(f'Frozen artifact changed: {arm}/{filename}')
        p = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        if p['protocol']['losses'] != loss_weights(arm):
            raise ValueError('Unexpected objective')
        for name, digest in p['protocol']['source_sha256'].items():
            if sha256(CODE/'cnn_ps'/name) != digest:
                raise ValueError(f'Training source changed: {name}')
        payloads[arm] = p
        records[arm] = read_json(folder/'eval_per_recording.json')
    intervals = {}
    for arm in list(ARMS)[1:]:
        result = paired_bootstrap(records[arm], records['A0_full'], repetitions=2000, seed=20261118)
        result['difference'] = f'A0_full minus {arm}, percentage points'
        intervals[arm] = result
    save_json(ROOT/'paired_bootstrap.json', dict(comparisons=intervals,
        multiplicity='Exploratory unadjusted 95% intervals across ten comparisons; not confirmatory familywise inference',
        limitation='Paired recording resampling conditional on fixed models; not training-seed variability or speaker-cluster bootstrap'))
    primary = 'event_f1_iou05_percent'
    rank = sorted(ARMS, key=lambda a: models[a][primary], reverse=True)
    summary = dict(status='complete', metrics=models, descriptive_test_ranking=rank,
        original_source_files_changed=changed, reference_reuse=read_json(ROOT/'reference_reuse_verification.json'),
        selected_epochs={a: p['epoch'] for a, p in payloads.items()},
        layer_weights={a: p['layer_weights'] for a, p in payloads.items()},
        bootstrap=intervals, inference=inference, benchmark=benchmark,
        caution='Test ranking describes this experiment only; it is not a newly validated model-selection rule')
    save_json(ROOT/'verification_summary.json', summary)
    fields = ['arm', 'description', 'selected_epoch', 'dev_event_f1', 'threshold'] + [k for k, _ in METRICS] + ['delta_event_f1_vs_full_pp', 'active_parameters', 'mean_inference_ms']
    with (ROOT/'metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for arm in ARMS:
            p, speed = payloads[arm], benchmark['models'][arm]
            writer.writerow(dict(arm=arm, description=DESCRIPTIONS[arm], selected_epoch=p['epoch'],
                dev_event_f1=p[primary], threshold=p['threshold'], **{k: models[arm][k] for k, _ in METRICS},
                delta_event_f1_vs_full_pp=models[arm][primary]-models['A0_full'][primary],
                active_parameters=speed['active_trainable_backend_parameters'], mean_inference_ms=speed['mean_ms']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = list(ARMS)
    fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    values = [models[a][primary] for a in labels]
    ax.barh(labels, values, color=['#b95f27']+['#4774a5']*10)
    ax.axvline(values[0], color='#a34e22', linestyle='--', linewidth=1)
    for y, value in enumerate(values):
        ax.text(value+.3, y, f'{value:.2f}', va='center', fontsize=9)
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel='Test event F1 (%) at IoU >= 0.5', title='A-group component ablations; seed 666')
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(ROOT/'event_f1_comparison.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    for y, arm in enumerate(labels[1:]):
        v = intervals[arm]['metrics'][primary]
        lo, hi = v['ci95_pp']
        ax.plot([lo, hi], [y, y], color='#4774a5', linewidth=2)
        ax.scatter([v['delta_pp']], [y], color='#b95f27', s=24)
    ax.set_yticks(range(10), labels[1:])
    ax.invert_yaxis()
    ax.axvline(0, color='#444444', linestyle='--', linewidth=1)
    ax.set(xlabel='Full model minus ablation: event F1 (percentage points)',
           title='Paired recording bootstrap: exploratory 95% intervals')
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(ROOT/'module_effect_intervals.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, subset in zip(axes, (labels[:5], [labels[0]]+labels[5:])):
        for arm in subset:
            history = read_json(ROOT/'training'/arm/'history.json')
            ax.plot([r['epoch'] for r in history], [r[primary] for r in history], label=arm, linewidth=1.3)
        ax.set(xlabel='Epoch', ylabel='Dev event F1 (%)', ylim=(0, 100))
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(ROOT/'development_curves.png', dpi=180)
    plt.close(fig)
    lines = ['# 冻结WavLM：A组模块消融验证报告', '',
        '本次完成A0—A10共11种设置：A0/A1复用已完成的同条件20轮结果，新增9组均从同一随机初始化训练20轮。'
        '复用前，全开发集分数及CPU/GPU单步更新通过逐元素一致性检查；训练总预算、模型选择规则一致。', '',
        '这里检验的是当前WavLM特征方案及其任务后端、辅助损失的条件贡献。未拆除WavLM编码器内部的Transformer、卷积或位置编码；内部结构属于后续实验。', '',
        '## 实验设置', '',
        '- 固定训练6,345条、开发2,484条、测试3,562条；UID互不重叠。训练使用同一套三视图缓存，开发/测试只用干净视图。',
        '- 编码器94,381,936个参数全部冻结。每组seed=666、20轮、15,880次更新、batch=8、AdamW lr=0.001、weight decay=0.0001、梯度裁剪5。',
        '- 共享参数初始化、数据顺序相同；两个独立进程运行，模型与随机状态不共享。训练耗时受并发影响，不用于推理速度比较。',
        '- 每组单独用开发集校准真实时长FPR≤5%的阈值，再选择开发集事件F1最高的轮次；epoch0可参与，平分保留较早轮次。',
        '- 全部组训练完成后冻结最佳权重及阈值，才读取本次新增模型的测试结果；无测试集调阈值、选轮次或早停。',
        '- 所有原有代码文件保持原样，新代码和结果独立保存。', '',
        '| 组别 | 唯一变化 |', '|---|---|']
    lines += [f'| {arm} | {DESCRIPTIONS[arm]} |' for arm in ARMS]
    lines += ['', '## 测试集指标', '',
        '指标单位为%，差值单位为百分点；事件F1为主要指标，其他指标用于判断精确率、召回率及误报的取舍。', '',
        '| 组别 | 事件F1↑ | 相对A0差值 | 事件P↑ | 事件R↑ | 短事件R↑ | 真实时长FPR↓ | 帧F1↑ | 帧EER↓ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        m = models[arm]
        vals = [m[primary], m[primary]-models['A0_full'][primary]] + [m[k] for k, _ in METRICS[1:]]
        lines.append('| '+arm+' | '+' | '.join(f'{v:+.4f}' if i == 1 else f'{v:.4f}' for i, v in enumerate(vals))+' |')
    lines += ['', '![事件F1比较](event_f1_comparison.png)', '', '## 各模块的条件贡献', '',
        '下表差值统一为“A0完整方案−该消融组”。对事件F1而言，正值支持保留完整方案中的对应设计；负值说明该消融设置在本轮表现更好。'
        '这是其他模块保持启用时的贡献，不能相加，也不等于证明模块在任何搭配下都独立有效。', '',
        '| 对照 | F1差值 | 按录音配对95%区间 | 当前证据 |', '|---|---:|---|---|']
    for arm, interval in intervals.items():
        v = interval['metrics'][primary]
        lo, hi = v['ci95_pp']
        verdict = ('完整方案较高' if lo > 0 else '消融组较高' if hi < 0 else '区间跨0，方向证据不足')
        lines.append(f"| A0−{arm} | {v['delta_pp']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {verdict} |")
    lines += ['', '![模块贡献区间](module_effect_intervals.png)', '',
        '每次以同一批录音同步重采样两个模型，共2,000次；完整短事件召回和误报差值区间见`paired_bootstrap.json`。'
        '十次比较的95%区间未经多重比较校正，只作探索性证据；区间不包含训练随机种子的波动，也不是按说话人聚类的区间。', '',
        '## 开发集选择、层权重与推理成本', '',
        '| 组别 | 最佳轮次 | 开发事件F1 | 冻结阈值 | 6/9/12层权重 | 有效可训练参数 | 推理均值ms | RTF |',
        '|---|---:|---:|---:|---|---:|---:|---:|']
    for arm, p in payloads.items():
        b = benchmark['models'][arm]
        weights = '/'.join(f'{x:.4f}' for x in p['layer_weights'])
        lines.append(f"| {arm} | {p['epoch']} | {p[primary]:.4f} | {p['threshold']:.8f} | {weights} | {b['active_trainable_backend_parameters']:,} | {b['mean_ms']:.2f} | {b['real_time_factor']:.4f} |")
    lines += ['', '训练完成后，GPU上依次对固定32条开发录音进行完整原始音频推理，每组预热3条；计时包含冻结WavLM、传输、投影、CNN/GRU与适配器，排除磁盘解码和模型加载。'
        '所有组仍执行完整12层编码器，单独取第6/9层不等于截短编码器。单次计时受温度及系统负载影响，微小差异不作速度优势结论。', '',
        '有效参数数排除了被关闭并冻结的分支；为保持公共初始化，checkpoint仍保留这些无效参数，因此不能据文件大小判断参数节省。', '',
        '![开发集学习曲线](development_curves.png)', '', '## 验证与适用边界', '',
        '- 68项单元测试通过；11组均完成只使用train/dev的小样本训练冒烟检查。',
        '- A0/A1与旧实现的CPU/GPU输出、损失、梯度和一次AdamW更新完全一致；全2,484条开发录音分数逐元素一致。',
        '- 关闭边界、一致性和粗辅助损失分别只改变对应项；关闭注意力/门控/可学习池化的参数冻结且绕过执行。',
        '- 最佳权重在32条固定开发录音上的原始音频推理与缓存评分检查通过，详见`single_audio_verification.json`。',
        '- 测试集在项目历史中已经查看过，本次结果属于既有测试集的增量研究。测试排序是描述性结果，不能再将按此排序选出的设置当作在全新测试集上验证过的最终模型。',
        '- 没有审计说话人/音频内容近重复或预训练语料重叠；UID不重叠不能替代这些审计。',
        '- 所有贡献均限于当前单种子、20轮和开发集选模规则。下一步应先用多个训练种子确认候选，再用未参与选择的保留集检验。', '',
        '## 产物', '',
        '- `training/<组别>/best.pt`：每组开发集选择的最佳权重。',
        '- `training/<组别>/history.json`、`dev_metrics.json`、`eval_metrics.json`：曲线及全部指标。',
        '- `experiment_plan.json`：训练前固定方案；`frozen_all_arms.json`：测试前冻结证据。',
        '- `metrics.csv`、`verification_summary.json`：表格与可机器读取的完整汇总。',
        '- `paired_bootstrap.json`：三项指标的配对置信区间。',
        '- `inference_benchmark.json`、`single_audio_verification.json`：推理成本和原始音频核对。', '']
    (ROOT/'A组消融验证报告.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Report complete; test event F1 (descriptive):', flush=True)
    for arm in rank:
        print(arm, models[arm][primary], flush=True)


if __name__ == '__main__':
    report()
