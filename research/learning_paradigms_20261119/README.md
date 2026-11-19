# 课件学习范式研究交接 · 2026-11-19

这是供组员研究的**独立实验资料**，不是默认模型升级。此分支只增加 `research/learning_paradigms_20261119/`，不修改根目录方法说明、推理/训练入口、权重或已有实验。不要把研究分支的结果当作已经验收的正式主方案。

## 先看结论

- 正式方法继续保留。以下研究基于已发布的多层 WavLM 第6/9/12层融合候选；仓库默认 `wavlm_frozen` 入口与多层融合候选不是同一个入口，本提交没有切换它们。
- **低通适应＋回放**值得保留：原dev事件F1从83.41%到84.00%，原性能区间保护检查通过。但直接低通微调也有效，收益不能全部归因于回放。
- **噪声增强＋旧:新损失3:1、lr=1e-4**是较均衡的抗噪研究候选。10dB白噪声事件F1从48.64%到58.64%；原域事件F1为83.15%，仅点估计通过保护线，区间尚未通过。
- 5:1未显示整体优势；3:1降低学习率至5e-5改善部分原域指标，但0dB误报明显增加。半监督、协同训练、额外边界多任务、CORAL、EWC、一阶MAML的当前小型配方没有稳定整体优势。
- **没有使用正式eval或LPS进行本轮训练、调参或评价。** 但dev历史上已用于源模型选择，而且后续方案反复参考了同一dev。所有结论都是探索性证据，不能当作独立盲测或真实未知域泛化证明。

## 给组员的阅读顺序

1. 先看下面的实验目录和结果总表，再打开各实验的 `结论摘要.txt` / `实验报告.txt`。
2. 看 `results/plan.json` 了解数据、种子、阈值和训练预算；看 `results/summary.csv`、`means.json`、`paired_comparisons.json` 和 `preservation_screen.json`。
3. 阅读对应 `run.py` / `suite.py` / `run_experiment.py`，以及其独立 `code/cnn_ps/` 快照。历史代码是逐字归档，不要直接在其中改路径或覆盖结果。
4. 按 [复现说明](REPRODUCING.md) 核验结果、配置新运行；[研究建议](TEAM_RESEARCH_GUIDE.md) 说明值得继续的方向和评估边界。

## 八轮实验

| 目录（均在experiments/下） | 内容 | 入口 |
|---|---|---|
| [paradigms_selftraining_20261119](experiments/paradigms_selftraining_20261119/) | 半监督伪标签，8轮教师＋4轮分支 | run_experiment.py / summarize.py |
| [paradigms_selftraining_teacher20_20261119](experiments/paradigms_selftraining_teacher20_20261119/) | 半监督复查，20轮教师＋4轮分支 | run_experiment.py / summarize.py |
| [paradigms_suite_20261119](experiments/paradigms_suite_20261119/) | 迁移、多任务、协同训练、CORAL、持续学习回放/EWC、元学习 | suite.py / report.py |
| [multilayer_replay_20261119](experiments/multilayer_replay_20261119/) | 实际多层模型：低通适应、旧样本、1:1回放对照 | run.py / evaluate_resume.py / report.py |
| [multilayer_noise_replay_20261119](experiments/multilayer_noise_replay_20261119/) | 噪声增强、旧样本、1:1回放与上轮候选对照 | run.py / report.py |
| [multilayer_noise_replay31_20261119](experiments/multilayer_noise_replay31_20261119/) | 噪声回放3:1，lr=1e-4 | run.py / report.py |
| [multilayer_noise_replay51_20261119](experiments/multilayer_noise_replay51_20261119/) | 噪声回放5:1，lr=1e-4 | run.py / report.py |
| [multilayer_noise_replay31_lr5e5_20261119](experiments/multilayer_noise_replay31_lr5e5_20261119/) | 噪声回放3:1，lr=5e-5 | run.py / report.py |

前期初筛是小数据、小模型。不要拿其事件F1与后面的实际多层融合模型直接排名。后期两个seed只改变同一源模型的微调顺序/dropout，不是两个独立预训练模型。

## 实际多层模型的噪声对照

以下均为事件F1（%），更新方案取两个微调种子均值；原域为2484条dev，压力条件为同一384条dev。原阈值固定，未重新校准。

| 方案 | 原域 | 白噪声10dB | 白噪声0dB | 粉红噪声10dB |
|---|---:|---:|---:|---:|
| 原多层模型 | 83.41 | 48.64 | 15.85 | 58.75 |
| 噪声增强＋1:1回放 | 82.71 | 58.33 | 28.39 | 61.80 |
| 噪声增强＋3:1回放 | 83.15 | 58.64 | 27.94 | 62.67 |
| 噪声增强＋5:1回放 | 83.26 | 58.43 | 27.87 | 62.84 |
| 噪声增强＋3:1，低学习率 | 83.11 | 58.42 | 26.85 | 62.99 |

原性能保护线：事件/帧F1下降≤0.5pp、真实时长误报增加≤0.5pp、短事件召回下降≤1pp。噪声3:1、5:1及低学习率方案均仅点估计通过，事件F1区间检查未通过。0dB误报率分别约53.63%、55.50%、64.92%；不能只看事件F1就忽略这些代价。

3:1和5:1是**损失权重比**；每步仍是旧8条＋噪声8条，彼此同算力。相对仅噪声训练，回放则多计算旧批次。

## 包含什么

- 运行时的Python代码快照、完整文本报告、训练历史、固定协议、逐种子CSV、逐录音指标、置信区间、验证记录和保存预测NPZ。
- [archive_manifest.json](archive_manifest.json)：1132个归档文件的大小与SHA256，以及未包含权重的文件名/哈希。归档不做路径替换，历史本机路径仅是出处记录。
- [tools/inspect_results.py](tools/inspect_results.py)：仅用Python标准库即可核验归档与逐录音聚合指标。
- [tools/run_replay.py](tools/run_replay.py)：四个噪声回放配方的新运行入口，配置外部数据、原多层checkpoint与WavLM；输出必须在归档目录之外。

**没有上传**原始训练/测试音频、标签/特征缓存、公开WavLM编码器、66个本轮后端checkpoint，也没有上传整份课件PDF。代码和保存的数值结果可以直接阅读；训练需要另行配置具有使用权限的数据与模型。原多层源checkpoint位于仓库已有的 `experiments/wavlm_multilayer_20261117/`，不由本提交修改。

## 快速核验（不需要GPU或数据）

```bash
python research/learning_paradigms_20261119/tools/inspect_results.py --verify
```

预期：1132份归档文件哈希通过，327组逐录音事件F1/时长误报聚合通过。此工具核验的是保存证据的一致性，不重新训练模型，也不把历史bootstrap区间转用到新的运行。

原模型与更早归档分数存在已记录的最大0.000679数值差异，434150帧有1帧阈值判断不同；第一轮低通实验保留了触发检查及评估修订。后续同机对照的原模型预测完全一致。详细限制和复现环境见 [REPRODUCING.md](REPRODUCING.md)。
