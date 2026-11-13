# 2026-11-13 缩减 SAL：PS eval 与 LPS 测试

复用 [2026-11-12 冻结基线](../sal_baseline_20261112)选定的同一模型，完成两个固定缩减测试集的全部预测。没有重新训练、重新选 checkpoint、调测试阈值或加入后处理。

| 测试集 | 录音数 | EER % ↓ | 伪造类帧级 F1 % ↑ | 真实类帧级 F1 % ↑ | 帧级准确率 % ↑ |
|---|---:|---:|---:|---:|---:|
| PS eval | 3,562 | 4.503134 | 95.383669 | 95.634361 | 95.512512 |
| LPS crossfade | 3,800 | 37.909355 | 62.441524 | 57.675701 | 60.200779 |

完整精度见 [metrics_summary.json](metrics_summary.json)。本次结果显示该冻结模型在 LPS 上的表现明显弱于 PS eval；不能据此宣称新的方法改善了跨数据集泛化。LPS 曾有历史探索，不称为新的盲测。

## 模型与数据身份

- checkpoint：`best-epoch=024.ckpt`，即第 25 轮；由同伴按最小 dev EER 选择，训练于第 40 轮结束。
- 原始权重 SHA256：`6b05b7406e20b719c7a55ea40c90905fda7f9d4f01ea7d3d50604d7c40c7d4ff`。下载分片、完整重组文件均已校验；模型严格加载全部匹配。
- 训练代码提交：`77b2dc3` 加同伴记录的兼容性修复；本次测试入口提交：`0d8a20e966d717257270713103d0d8f6db5a051f`，工作区干净，推理逻辑保持原样。
- 固定 `reduced` 清单：eval 为 `203792d5718e75e54614ad005386a3037cd9ddb83dd6dc7a93266dfc9c4252b1`；LPS 为 `7e4ff246af20815b7cd65ec0c0821cc1ba3b2a8de984df3261f50925222d3f21`。
- 全部六个数据角色的音频与标签内容哈希检查通过，见 [data_audit.json](data_audit.json)。PS 原生网格为 160 ms，LPS 为 20 ms。
- RTX 4060 Laptop GPU；FP32、batch=1、完整录音推理。全部 7,362 条在 GPU 上完成，CPU OOM 回退为 0。

EER 使用连续 `logit_real - logit_fake` 排序，按原实现最近 FPR/FNR 交点计算。F1 与准确率使用原始二分类 argmax，平分选类别 0；`0=伪造，1=真实`。EER 交点阈值没有被用于调节分类或区间结果。这里 F1 为帧级指标，不是事件级指标。

## 文件与核验

- [eval 原始结果与缓存](predictions/sal_reduced_eval_20261113)、[LPS 原始结果与缓存](predictions/sal_reduced_lps_20261113)：各含 `metrics.json`、`predictions.sqlite`、`config.json`、`checkpoint_metadata.json`、`status.json`。SQLite 保留逐录音 logits、标签、时长、设备和推理耗时。
- [verification.json](verification.json)：本机完成后的核验；[packaged_verification.json](packaged_verification.json)：整理成仓库目录后再次核验。
- [evaluation_plan.json](evaluation_plan.json)、[records](records)、[eval_console.log](eval_console.log)、[lps_console.log](lps_console.log)：原始运行计划、状态记录和日志。绝对路径仅用于溯源，换机请使用下方相对路径命令。
- [files_manifest.json](files_manifest.json)：本目录逐文件大小与 SHA256，清单自身除外；[verify_results.py](verify_results.py) 使用仓库中的固定清单核验缓存，无需音频、权重或 GPU。

核验检查 SQLite 完整性、全部录音 ID/顺序、条数、有限 logits、二值标签、设备统计、缓存元数据中的冻结权重身份，并独立重算混淆矩阵、两类 F1 和准确率。百分比差异均小于 `1e-4` 个百分点。**EER 沿用原推理入口的计算值，本次核验没有独立重算 EER。**

在项目根目录，使用已有项目 Python 环境执行：

```powershell
.\code\.venv\Scripts\python.exe result/sal_frozen_test_20261113/verify_results.py
```

该脚本只需 Python 与 NumPy，结果输出到 stdout，不修改缓存。原始 JSON、日志、SQLite 均按字节保留；空的运行锁文件未纳入仓库。

## 重新运行推理

按[同伴的权重说明](../sal_baseline_20261112/README.md#下载训练权重)取得并重组 Release 权重，放入原模型目录。准备固定数据包后，在 `code` 目录运行以下命令；输出名称必须尚不存在：

```powershell
.\.venv\Scripts\python.exe sal.py --mode test --split eval --checkpoint "../data/models/sal_train_reduced_20261112_122313_966079/best-epoch=024.ckpt" --run-name sal_reduced_eval_recheck
.\.venv\Scripts\python.exe sal.py --mode test --split lps --checkpoint "../data/models/sal_train_reduced_20261112_122313_966079/best-epoch=024.ckpt" --run-name sal_reduced_lps_recheck
```

本次上传复用已发布的权重与数据包，只新增测试结果、缓存和核验材料。
