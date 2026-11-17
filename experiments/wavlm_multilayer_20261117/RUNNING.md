# 代码与运行说明

以下命令从本目录的 `code/` 子目录执行。该目录有独立的 `cnn_ps` 包，避免导入仓库旧版本。不要把本目录覆盖复制到仓库根目录。

## 1. 环境

实验使用Python 3.10.18、PyTorch 2.2.1+cu121、transformers 4.51.3、RTX 4080 Laptop GPU。重新训练需要CUDA；单音频预测也支持CPU。具体依赖固定在 `requirements-release.txt` 及其引用的两个requirements文件中。

```powershell
cd experiments/wavlm_multilayer_20261117/code
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-release.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe verify_release.py
```

已有兼容Python环境可直接使用，无需再次创建。Linux将 `.\.venv\Scripts\python.exe` 替换为 `.venv/bin/python`。复现报告需要matplotlib 3.10.1，该依赖已列入发布要求。

## 2. 下载固定的公开WavLM编码器

```powershell
.\.venv\Scripts\python.exe -m cnn_ps.download_wavlm --output ../../../data/models/wavlm-base-plus
```

该命令只下载公开模型，不上传音频；固定revision为 `4c66d4806a428f2e922ccfa1a962776e232d487b`，下载后会验证模型SHA256。已有同版本模型可直接复用。

## 3. 直接使用最佳模型预测

```powershell
.\.venv\Scripts\python.exe -m cnn_ps.predict_wavlm_multilayer `
  --audio "你的16kHz单声道音频.wav" `
  --checkpoint ../weights/best.pt `
  --model-folder ../../../data/models/wavlm-base-plus `
  --output prediction.json --device cuda --threads 4
```

CPU推理使用 `--device cpu`。输出包含20ms假音频logit、冻结阈值和预测伪造区间。不能将该阈值当成概率阈值，也不要在测试音频上重新调整阈值。输出文件已存在时会拒绝覆盖，请使用新文件名。

## 4. 复现完整对照实验

默认从仓库根目录的 `data/` 读取原有数据，结果只写入本独立目录的 `runs/`。所需结构：

```text
<project-root>/data/reduced/{train_detector,train_edit,dev,eval}/audio/<UID>.wav
<project-root>/data/reduced/<role>/segment_labels/{train,dev,eval}_seglab_0.16.npy
<project-root>/data/downloads/database_segment_labels_v1.2.tar.gz
<project-root>/data/models/wavlm-base-plus/
```

各role只需对应split的标签文件。训练UID固定在 `code/manifests/ps_fixed_ids.json`；不得重新抽取划分后仍声称复现本次结果。官方标签归档会验证SHA256。

```powershell
.\.venv\Scripts\python.exe run_multilayer_experiment.py `
  --project-root ../../.. --output ../runs/full

.\.venv\Scripts\python.exe report_multilayer_experiment.py --output ../runs/full
```

执行顺序：准备清单及标签 → train/dev参考缓存 → 冻结WavLM三层缓存 → 第12层基线20轮 → 融合20轮 → 冻结所选权重及阈值 → eval缓存及评价。数据配置保持原实验设置，batch=8，seed=666，AdamW lr=0.001，4个CPU线程。

这会创建数十GB缓存，需要充足磁盘空间。已完成的阶段会复用；不覆盖已有训练目录，也不提供训练中途的优化器断点恢复。全新实验请指定新的输出目录。状态在 `runs/full/status.json`，各阶段日志在 `runs/full/logs/`，训练进度在各组 `progress.json`。

`--prepare-only` 只准备清单、标签以及train/dev参考缓存，不训练模型。`--project-root` 可以指向包含同样 `data/` 结构的其他目录。

报告脚本面向新生成的完整 `runs/full/`，需要两组训练的全部归档；发布包的 `reports/` 是已有结果档案，只附最佳融合权重。核验发布档案请使用 `verify_release.py`，无需重新训练。

## 5. 小样本验证与源码

- `cnn_ps/wavlm_multilayer.py`：冻结编码器、三层特征缓存、加权融合、缓存读取。
- `cnn_ps/train_wavlm_multilayer.py`：单组训练、配对冻结、固定阈值评价。
- `cnn_ps/predict_wavlm_multilayer.py`：最佳权重单音频预测。
- `tests/test_wavlm_multilayer.py`：基线等价、仅多3个参数、随机数一致性、融合计算和梯度。

历史小样本冒烟脚本保留为 `verify_multilayer_smoke.py`，需要预先准备的4/4/2条小样本mel/ssl缓存。它不是性能实验，也不下载数据：

```powershell
.\.venv\Scripts\python.exe verify_multilayer_smoke.py `
  --reference-root "小样本缓存根目录" `
  --model-folder ../../../data/models/wavlm-base-plus `
  --output ../runs/smoke_new
```

一般代码检查直接运行第1节的63项测试，不依赖这些历史缓存。

## 6. Windows防休眠8小时

```powershell
.\.venv\Scripts\python.exe prevent_sleep_8h.py
```

默认8小时，可用 `--hours` 修改；`--status-file ../runs/keep_awake.json` 可记录PID和到期时间。脚本通过Windows `SetThreadExecutionState` 阻止因闲置自动休眠，允许屏幕关闭，不永久修改电源计划。到期或Ctrl+C会释放请求；手动休眠、合盖策略、关机不受该请求保证。

## 7. 对结果的解释

完整指标及置信区间见[验证报告](reports/多层融合验证报告.md)，数据泄露及评估边界见[README](README.md#数据泄露与结论边界)。当前结果使用单随机种子，且eval历史上已经被查看，不应描述为完全独立盲测。
