"""Report completed B interventions; refuses missing/incomplete results."""
import argparse
from pathlib import Path
from cnn_ps.io import read_json, save_json, sha256
from cnn_ps.wavlm_ablation_b import ARMS, NEW_ARMS
from cnn_ps.train_wavlm_ablation_b import freeze_eval_plan
from report_multilayer_experiment import METRICS, paired_bootstrap
from run_wavlm_ablation_b import DEFAULT_OUTPUT


def report(output):
    root = Path(output)
    comparison = read_json(root/'all_comparison.json')
    if comparison['status'] != 'complete' or set(comparison['models']) != set(ARMS):
        raise ValueError('All arms must be complete')
    frozen = freeze_eval_plan(root/'training', list(NEW_ARMS))
    if frozen != comparison['frozen']:
        raise ValueError('Comparison differs from frozen weights')
    for arm in ARMS:
        expected = comparison['baseline'] if arm == 'B0_full' else frozen[arm]
        if sha256(root/'training'/arm/'best.pt') != expected['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')
        if read_json(root/'training'/arm/'eval_metrics.json') != comparison['models'][arm]:
            raise ValueError('Evaluation artifact changed')
    inference = read_json(root/'single_audio_verification.json')
    if inference['status'] != 'passed':
        raise ValueError('Raw-audio inference verification must pass')
    records = {a: read_json(root/'training'/a/'eval_per_recording.json') for a in ARMS}
    bootstrap = {}
    for arm in NEW_ARMS:
        value = paired_bootstrap(records[arm], records['B0_full'], repetitions=2000, seed=20261119)
        value['difference'] = f'B0_full minus {arm}, percentage points'
        bootstrap[arm] = value
    save_json(root/'paired_bootstrap.json', bootstrap)
    summary = dict(status='complete', metrics=comparison['models'], bootstrap=bootstrap,
        inference=inference, benchmark=read_json(root/'inference_benchmark.json'),
        limitation='Single seed; unadjusted exploratory intervals; pretrained-model interventions, not retraining WavLM pretraining')
    save_json(root/'verification_summary.json', summary)
    lines = ['# B组：冻结WavLM内部位置机制消融报告', '',
        '参照为A0完整后端；B1关闭卷积位置嵌入，B2将输入依赖的位置门控固定为1且保留偏置，B3将相对位置偏置置零。'
        '三组分别重新缓存特征、从配对随机初始化训练完整后端20轮，WavLM参数不更新。', '',
        '| 指标（%） | B0完整 | B1无卷积位置 | B2门控固定1 | B3无相对偏置 |', '|---|---:|---:|---:|---:|']
    for key, label in METRICS:
        lines.append('| '+label+' | '+' | '.join(f"{comparison['models'][a][key]:.4f}" for a in ARMS)+' |')
    lines += ['', '## 条件贡献区间', '', '正值表示完整模型较高；单位为百分点。', '',
        '| 对照 | 事件F1差值 | 95%配对区间 |', '|---|---:|---|']
    for arm, b in bootstrap.items():
        v = b['metrics']['event_f1_iou05_percent']
        lo, hi = v['ci95_pp']
        lines.append(f"| B0−{arm} | {v['delta_pp']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += ['', '## 实验边界', '',
        '- 阈值及轮次仅由开发集选择；三组完成并冻结后才进行测试评价。',
        '- B0复用通过核验的A0，不因A5在历史测试集表现较好而更换后端。',
        '- 这些干预只能说明当前预训练权重对位置机制的依赖，不能推断移除机制后重新预训练的性能。',
        '- 单种子、按录音重采样2,000次、三个区间未经多重比较校正；不覆盖重训波动或说话人聚类相关。',
        '- 测试集曾被项目历史查看，属于既有基准上的探索性消融。',
        '- 原始音频推理核对和可训练参数/推理时间见single_audio_verification.json及inference_benchmark.json。', '']
    (root/'B组消融验证报告.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    report(parser.parse_args().output)
