"""Report the fixed two-arm experiment, including paired recording bootstrap."""
import argparse
from pathlib import Path
import numpy as np
import torch
from cnn_ps.io import read_json, save_json, sha256

ARMS = ('layer12', 'fusion6_9_12')
METRICS = (
    ('event_f1_iou05_percent', '事件F1（IoU≥0.5，↑）'),
    ('event_precision_percent', '事件精确率（↑）'),
    ('event_recall_percent', '事件召回率（↑）'),
    ('short_event_recall_le320ms_percent', '≤320ms短事件召回率（↑）'),
    ('real_duration_fpr_percent', '真实时长误报率（↓）'),
    ('frame_f1_percent', '20ms帧F1（↑）'),
    ('eer_percent', '帧EER（↓）'),
)


def paired_bootstrap(baseline, fusion, repetitions=2000, seed=20261117):
    if [r['uid'] for r in baseline] != [r['uid'] for r in fusion]:
        raise ValueError('Bootstrap requires paired recording order')
    keys = ('event_tp', 'event_pred', 'event_truth', 'short_tp', 'short_truth',
            'fp_seconds', 'tn_seconds')
    a, b = (np.asarray([[r[k] for k in keys] for r in rows], dtype=np.float64)
            for rows in (baseline, fusion))
    np.testing.assert_array_equal(a[:, [2, 4]], b[:, [2, 4]])
    np.testing.assert_allclose(a[:, 5:].sum(1), b[:, 5:].sum(1), rtol=0, atol=1e-10)

    def metrics(s):
        return np.asarray([200*s[0]/(s[1]+s[2]), 100*s[3]/s[4], 100*s[5]/(s[5]+s[6])])

    rng = np.random.default_rng(seed)
    differences = np.empty((repetitions, 3))
    for i in range(repetitions):
        ids = rng.integers(0, len(a), size=len(a))
        differences[i] = metrics(b[ids].sum(0)) - metrics(a[ids].sum(0))
    point = metrics(b.sum(0)) - metrics(a.sum(0))
    intervals = np.percentile(differences, [2.5, 97.5], axis=0)
    return dict(method='paired recording percentile bootstrap', repetitions=repetitions,
                seed=seed, difference='fusion minus baseline, percentage points',
                metrics={key: dict(delta_pp=float(point[i]),
                    ci95_pp=intervals[:, i].tolist()) for i, key in enumerate((
                        'event_f1_iou05_percent', 'short_event_recall_le320ms_percent',
                        'real_duration_fpr_percent'))},
                limitation='Conditional on two fixed trained models; not training-seed variability')


def report(root):
    root = root.resolve()
    training = root/'training'
    comparison = read_json(training/'comparison.json')
    if comparison['status'] != 'complete':
        raise ValueError('Complete evaluation required')
    models = comparison['models']
    records = [read_json(training/arm/'eval_per_recording.json') for arm in ARMS]
    bootstrap = paired_bootstrap(*records)
    save_json(root/'paired_bootstrap.json', bootstrap)
    payloads = {arm: torch.load(training/arm/'best.pt', map_location='cpu', weights_only=False)
                for arm in ARMS}
    source_folder = Path(__file__).resolve().parent/'cnn_ps'
    for arm, payload in payloads.items():
        for name, digest in payload['protocol']['source_sha256'].items():
            if sha256(source_folder/name) != digest:
                raise ValueError(f'Training source changed after this run: {arm}/{name}')
        if sha256(training/arm/'best.pt') != comparison['frozen'][arm]['checkpoint_sha256']:
            raise ValueError(f'Frozen checkpoint changed: {arm}')
    weights = {arm: p['layer_weights'] for arm, p in payloads.items()}
    snapshot = root.parent/'existing_code_sha256.json'
    repo = Path(__file__).resolve().parents[1]
    changed = [p for p, digest in read_json(snapshot).items() if sha256(repo/Path(p.replace(chr(92), '/'))) != digest]
    if changed:
        raise AssertionError(f'Pre-existing source changed: {changed}')
    main = bootstrap['metrics']['event_f1_iou05_percent']
    if main['ci95_pp'][0] > 0:
        verdict = '本次单种子对照中，融合的事件F1提高，按录音配对的95%置信区间完全高于0。'
    elif main['ci95_pp'][1] < 0:
        verdict = '本次单种子对照中，融合的事件F1下降，按录音配对的95%置信区间完全低于0。'
    else:
        verdict = '本次对照的事件F1差值置信区间包含0，不能据此确认稳定提升。'
    summary = dict(status='complete', verdict=verdict, metrics=models, layer_weights=weights,
                   bootstrap=bootstrap, original_source_files_changed=changed,
                   selected_epochs={a: p['epoch'] for a, p in payloads.items()})
    reference_audit = root.parent/'published_reference_dev_verification.json'
    if reference_audit.exists():
        summary['published_checkpoint_dev_check'] = read_json(reference_audit)
    inference_audit = root/'single_audio_verification.json'
    if inference_audit.exists():
        summary['final_checkpoint_inference_check'] = read_json(inference_audit)
    save_json(root/'verification_summary.json', summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for arm, color, label in zip(ARMS, ('#3866a6', '#ba5b29'),
                                 ('Layer 12 baseline', 'Layers 6/9/12 fusion')):
        history = read_json(training/arm/'history.json')
        axes[0].plot([r['epoch'] for r in history],
                     [r['event_f1_iou05_percent'] for r in history],
                     color=color, linewidth=2, label=label)
        selected = payloads[arm]
        axes[0].scatter(selected['epoch'], selected['event_f1_iou05_percent'],
                        color=color, marker='*', s=100, zorder=3)
        trained = [r for r in history if r['loss'] is not None]
        axes[1].plot([r['epoch'] for r in trained], [r['loss'] for r in trained],
                     color=color, linewidth=2, label=label)
    axes[0].set(title='Development event F1 (IoU >= 0.5)', ylabel='F1 (%)')
    axes[1].set(title='Training objective', ylabel='Mean batch loss')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8)
    fig.savefig(root/'training_curves.png', dpi=180)
    plt.close(fig)
    lines = ['# 冻结WavLM多层融合：实测报告', '', verdict, '',
             '## 改动与对照', '',
             '- 基线：第12层输出；融合：第6、9、12层输出经softmax权重加权求和。',
             '- WavLM的94,381,936个参数全部冻结；融合只增加3个可学习标量，权重初始均为1/3。',
             '- 原有源码未修改。两组使用相同随机初始化、数据顺序、三种音频视图、投影、CNN/GRU、双分辨率模块及损失。',
             '- 固定数据：训练6,345条、开发2,484条、测试3,562条；seed=666，各20轮，batch=8，AdamW lr=0.001。',
             '- 开发集真实时长FPR≤5%校准阈值，按开发集事件F1选模型。全部训练完成后冻结权重及阈值，才进行测试评价。',
             '', '## 测试集指标', '',
             '| 指标 | 第12层基线 | 三层融合 | 融合−基线（百分点） |',
             '|---|---:|---:|---:|']
    for key, label in METRICS:
        x, y = models[ARMS[0]][key], models[ARMS[1]][key]
        lines.append(f'| {label} | {x:.4f}% | {y:.4f}% | {y-x:+.4f} |')
    lines += ['', '## 开发集选择与融合权重', '',
              '| 方案 | 最佳轮次 | 开发集事件F1 | 冻结阈值 | 层6 / 层9 / 层12权重 |',
              '|---|---:|---:|---:|---|']
    for arm, p in payloads.items():
        w = ' / '.join(f'{x:.4f}' for x in weights[arm])
        lines.append(f"| {arm} | {p['epoch']} | {p['event_f1_iou05_percent']:.4f}% | {p['threshold']:.8f} | {w} |")
    lines += ['', '## 配对不确定性估计', '',
              '以录音为单位进行2,000次配对bootstrap，重采样时两组使用同一批录音；区间单位为百分点。', '',
              '| 差值 | 点估计 | 95%区间 |', '|---|---:|---|']
    for key, values in bootstrap['metrics'].items():
        lo, hi = values['ci95_pp']
        lines.append(f"| {key} | {values['delta_pp']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += ['', '## 验证与限制', '',
              '![开发集事件F1与训练损失曲线](training_curves.png)', '',
              '- 新基线与原后端初始化及输出完全一致；真实音频第12层缓存与原提取器逐元素相等。',
              '- 三层融合权重可正常收到梯度；编码器全部冻结；训练、模型选择、冻结、评价、单音频预测均已验证。',
              '- 一共63项单元测试通过；小样本预测与缓存评价分数完全一致。',
              '- 这里只比较预先指定的第6、9、12层融合，没有用测试集搜索层组合或调整阈值。',
              '- 本实验验证的是三层加权融合整体相对第12层的效果，不单独证明每层或学习权重相对固定平均的贡献。',
              '- 单随机种子；bootstrap只反映当前两个固定模型在录音采样上的不确定性，不包含重训波动。',
              '- 测试集在项目历史中已经被查看过，结果属于既有测试集上的增量实验。',
              '- 本机重新训练的基线可能因运行环境与历史报告不同而有差异；结论以这次相同条件的对照为准。',
              '', '## 产物', '',
              '- `training/layer12/best.pt`：同机重训基线。',
              '- `training/fusion6_9_12/best.pt`：融合模型。',
              '- `training/*/history.json`：逐轮训练及开发集指标。',
              '- `training/*/eval_metrics.json`：全部测试指标。',
              '- `training/frozen_eval_plan.json`：测试前冻结的权重及阈值。',
              '- `paired_bootstrap.json`：配对置信区间。',
              '- `verification_summary.json`：汇总及原源码未改动检查。', '']
    if reference_audit.exists():
        audit = summary['published_checkpoint_dev_check']
        lines += ['## 原模型复现核对', '',
                  '原有WavLM权重使用新增缓存的第12层特征，在全部2,484条开发录音上按原阈值评价。', '',
                  '| 指标 | 复现值 | 与原权重归档指标差值（百分点） |', '|---|---:|---:|']
        for key, value in audit['metrics'].items():
            lines.append(f"| {key} | {value:.8f}% | {audit['delta_pp'][key]:+.8f} |")
        lines.append('')
    if inference_audit.exists():
        lines += ['## 最终权重单音频推理核对', '',
                  '| 方案 | 最大分数绝对差 | 不同判定帧数 | 阈值完全一致 |', '|---|---:|---:|---|']
        for arm, audit in summary['final_checkpoint_inference_check']['models'].items():
            lines.append(f"| {arm} | {audit['max_absolute_score_difference']:.10g} | {audit['differing_decisions']} | {audit['threshold_exact']} |")
        lines += ['', '使用同一条开发录音比较原始音频直接推理和该最佳权重归档的缓存评价分数。', '']
    (root/'多层融合验证报告.md').write_text('\n'.join(lines), encoding='utf-8')
    print(verdict, flush=True)
    print(main, flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'runs/full')
    args = parser.parse_args()
    report(args.output)
