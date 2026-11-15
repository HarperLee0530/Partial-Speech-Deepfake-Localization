# CNN 权重：训练起点与选中成品

当前主方案是 **CNN-BiGRU + 信道增强直接微调**。本目录只保留两个完整 CNN checkpoint：一个作为训练起点和原 CNN 对照，一个作为已经训练完成的选中模型。两者可独立加载，不需要额外判别头。

这两个文件与训练电脑归档逐字节一致，共 7,847,120 字节（约 7.85 MB）。普通 Git clone、Download ZIP 或单文件 Download raw 都可取得，无需 Git LFS。

| 文件 | 用途 | 当前原生 20ms 阈值 |
|---|---|---|
| **`original_cnn.pt`** | **训练起点**；复现最初原 CNN-BiGRU 对照，或开始新的直接微调实验 | `4.5316390991210946` |
| **`cnn_finetuned_seed666.pt`** | **选中成品**；完成信道增强全参数微调的 CNN，第 8 轮，用于主方案推理 | `0.2922912240028382` |

两个阈值均来自 **同一 PS dev 2,484 条固定子集，真实音频时长误报率上限 5%** 的统一校准规则；分别校准后冻结，用于对应模型的 eval 和推理。分数是“越大越假”的 CNN logit，`score >= threshold` 判假。比较使用 float64，保留同分边界阈值精度；不要将表中数值截短或先转换为 float32。

文件大小、SHA256、所选轮次、当前 20ms 阈值及历史 160ms 阈值见 [manifest.json](manifest.json)。这次整理没有重新训练或改选 checkpoint。此前包含八个权重的加载与推理核验属于[历史版本验证记录](https://github.com/HarperLee0530/Partial-Speech-Deepfake-Localization/blob/51b20490d8b795fad13d83375ade72b25042b764/docs/verification_published_weights.json)，不代表当前目录仍提供八个文件。

## 原 CNN 的身份

`original_cnn.pt` 对应旧文件名 `cnn_bigru_train_20261109_203335_588602_best.pt`。

```text
SHA256: 36899e66bcf84b1f5aa8a6f2d57128eb80b077bc28eaa94d9bb8fbb713a2a428
参数: 911,521
seed: 666
训练: 完整 PS train 25,380 条
验证: 完整 PS dev 24,844 条
所选轮次: 7
```

原模型训练时使用的数据范围，与本次统一评估使用的 dev 2,484 条子集不同。`cnn_finetuned_seed666.pt` 从该原模型出发，在 PS train 6,345 条子集上直接微调，使用 dev 2,484 条子集选择第 8 轮。微调模型的 SHA256 为 `ed3bf849955ae0d00f4d2ca7c92f7d9eb719b0d07dd5fb866dfa4b547eab9157`。

原 checkpoint 保留历史 `protocol`、路径和 `interval_threshold=0.6` 等元数据。0.6 属于旧区间导出设置，不是当前 20ms 阈值。微调 checkpoint 仍保留当时的开发集模型选择规则 `J = 0.5 × 时长 F1 + 0.5 × 事件 F1`；统一重评没有根据 eval 重新选择权重。

## 加载与推理

在仓库 `code/` 目录运行，输入要求为 16kHz 单声道音频。

```powershell
# 校验两个权重
Get-FileHash ../weights/original_cnn.pt -Algorithm SHA256
Get-FileHash ../weights/cnn_finetuned_seed666.pt -Algorithm SHA256

# 原 CNN 对照，原生 20ms，使用对应的冻结 dev 阈值
python -m cnn_ps.predict --audio ../data/example.wav --checkpoint ../weights/original_cnn.pt --resolution 0.02 --threshold 4.5316390991210946 --output ../result/original_prediction.json

# 选中的直接微调 CNN，原生 20ms
python -m cnn_ps.predict --audio ../data/example.wav --checkpoint ../weights/cnn_finetuned_seed666.pt --resolution 0.02 --threshold 0.2922912240028382 --output ../result/finetuned_prediction.json
```

Python 中可直接加载选中的成品：

```python
from cnn_ps.backbone import load_backbone

cnn, frontend = load_backbone("../weights/cnn_finetuned_seed666.pt", device="cpu")
```

`load_backbone` 默认将 CNN 置于评估、冻结状态。开始新的微调实验时，使用 `original_cnn.pt`；`finetune.py` 会显式开放全部参数。checkpoint 不含优化器续训状态，新的微调运行会建立新的优化器。

## 历史 160ms 结果

160ms 仅用于回放历史辅助结果：原 CNN 使用阈值 `0.0`，微调 CNN 使用阈值 `-0.8814183473587036`。必须同时设置 `--resolution 0.16` 和对应阈值，不能混用 20ms 的阈值或结果。

音频、标签等数据仍需按根目录 README 准备。当前主流程只使用上述两个 CNN 权重。
