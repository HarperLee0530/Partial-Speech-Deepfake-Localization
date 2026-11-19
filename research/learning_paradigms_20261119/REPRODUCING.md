# 复现与代码使用

## 1. 不训练即可复核结果

在仓库根目录执行：

```bash
python research/learning_paradigms_20261119/tools/inspect_results.py --verify
```

Python3.10及以上、仅标准库。它核验归档文件SHA256、逐录音事件F1/真实时长FPR聚合并输出索引。原始 `summary.csv` 可直接在表格软件打开，NPZ包含保存分数。

历史 `report.py` / `summarize.py` 还会检查本机权重哈希、数据路径及标签，不能在缺失权重/数据时直接运行。不要删除那些断言来让归档报告“通过”。

## 2. 新运行：四个噪声回放配方

新入口只复用对应历史文件中的实际训练函数，保持架构、优化器设置、固定轮次和数据划分。它不会修改历史文件或默认方法。支持：

| profile | 旧:噪声损失 | 学习率 |
|---|---:|---:|
| noise11 | 1:1 | 1e-4 |
| noise31 | 3:1 | 1e-4 |
| noise51 | 5:1 | 1e-4 |
| noise31_lr5e5 | 3:1 | 5e-5 |

每个profile从原始多层checkpoint出发，训练两个seed的回放模型，并新评估原模型和这两个模型。为了避免混用新结果与旧区间，不自动导入其他历史对照，也不自动生成新bootstrap区间。新运行会输出完整逐录音指标、保存预测、summary.csv及冻结模型清单。历史多臂对照和区间仍保存在归档中。

### 环境

原运行：Windows、Python3.10、PyTorch2.2.1+cu121、CUDA GPU。Python依赖版本见 `requirements-research.txt`；PyTorch/torchaudio请使用匹配驱动的官方安装方式，历史CUDA12.1环境可用：

```bash
python -m pip install torch==2.2.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r research/learning_paradigms_20261119/requirements-research.txt
```

完整训练需要CUDA；`--help`不导入深度学习库，`--check`需要Python依赖和数据，但不训练、不创建输出目录。跨GPU/软件版本可能出现浮点偏差，不保证逐位复现。

### 数据和模型（必须另行准备）

- 原PS训练/开发缓存 `cnn_ps_features_v2/{train,dev}`：原 `complete.json`，以及 `labels20.npy`、`labels160.npy`、`weights20.npy`、`weights160.npy`；元数据所引用的16kHz单声道原音频。前期LogMel/迁移筛选还需要其特征缓存。
- 固定公开编码器 `wavlm_base_plus_hf`，包含HF配置及 `pytorch_model.bin`。权重SHA256：`3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51`。
- 仓库已有的多层模型 `experiments/wavlm_multilayer_20261117/weights/best.pt`，SHA256：`160a4b3cacbee48398d11f218e3ca2dff76615670ae05255f6775119dd37b50e`。
- 该源模型目录中的 `reports/training/fusion6_9_12/dev_scores20.npz`，用于复现核验。若文件是Git LFS指针，先取得真实文件。

原始数据、缓存与公开编码器均未包含在此提交。需要从项目拥有者取得相同缓存元数据及有权限使用的数据。元数据SHA必须与历史plan一致；若要使用新划分/新数据，应明确建立新的实验协议，而不是把它标成历史复现。不要将数据上传到此分支。

复制 `config.example.json` 为本地配置（例如此目录下被忽略的 `local_config.json`），填写路径。原元数据若引用另一台机器的音频路径，可使用前缀映射，保持原元数据本身及哈希不变：

```json
"audio_path_map": [
  {"from": "D:/old/audio/root/", "to": "/mnt/team/audio/root/"}
]
```

`from`应按实际 `complete.json` 中记录填写，不是固定值。所有配置路径建议用绝对路径。

```bash
python research/learning_paradigms_20261119/tools/run_replay.py --profile noise31 --config /absolute/local_config.json --output /absolute/new_runs/noise31_seedpair --check
python research/learning_paradigms_20261119/tools/run_replay.py --profile noise31 --config /absolute/local_config.json --output /absolute/new_runs/noise31_seedpair
```

输出目录必须尚不存在，且位于归档目录之外；不覆盖旧结果、不修改原checkpoint。此入口会创建三层float32特征缓存，需要额外磁盘空间。中断后保留部分输出供检查；若要重跑，使用新的输出目录。原数据只读。

交付前已完成：源码语法检查、归档哈希及指标聚合核验、四个profile配置/模型/数据预检。新入口未额外重跑四套完整训练；历史训练结果来自归档代码运行，不能把入口预检说成重新训练验证。

## 3. 其他课件方法

半监督：`paradigms_selftraining_20261119/run_experiment.py` 与teacher20版本；初筛：`paradigms_suite_20261119/suite.py`；低通回放：`multilayer_replay_20261119/run.py`。

它们作为原始、带哈希的实验源码保留，本次没有把所有方法重写成通用CLI。若继续开发，请在新工作目录复制代码，配置缓存和源权重路径，并保存新的plan/哈希/输出，不修改归档。建议先阅读常量与依赖关系：

- 半监督：`CACHE`、`RESULT`；教师8轮/20轮固定，已有标签隐藏模拟无标注，不是真正新采集无标注数据。
- 六方向初筛：`common.CACHE`、`WROOT`、`WWEIGHTS`、`OLD`、`OUT`；依赖teacher20的历史参考、冻结WavLM特征缓存。不能仅运行suite.py就假定缺失权重会自动恢复。
- 低通回放：`DATA`、`ENCODER`、`SOURCE`、`RELEASE`、`BANK`、`OUT`。首轮原始脚本的1e-4归档复现门槛触发，随后通过单独 `evaluate_resume.py` 记录修订续跑；不要隐去这一过程。
- 后续噪声回放：`PREVIOUS`引用上一轮冻结对照；其缓存分数与新跑原模型逐项比对一致后才复用。不同目录里的 `previous_replay` 所指配方不同，请以对应报告为准。

训练权重本次未上传，完整清单及SHA保存在 `archive_manifest.json` 的omitted中；若确需历史checkpoint直接推理，应另行向项目拥有者取得对应权重，而不是拿默认推理入口的另一套checkpoint替代。
