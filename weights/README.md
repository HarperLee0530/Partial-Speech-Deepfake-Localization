# CNN 权重：下载后用哪一个

这些是实际产生既有 PS 结果的 checkpoint 原件，与训练电脑归档逐字节一致。普通 Git clone、Download ZIP 或单文件 Download raw 都可取得，无需 Git LFS、SAL Release 或另外寻找 `result/` 文件夹。八个文件共约 10.78 MB。

| 文件 | 用途 |
|---|---|
| **`original_cnn.pt`** | **最初的原 CNN-BiGRU 基线。复现基线、准备冻结特征、训练小修正头、开始新的直接微调实验，都从这个文件开始。** |
| `cnn_finetuned_seed666.pt` | 已完成信道增强全参数微调的 CNN，第8轮；用于复现该方案推理。不要把它当原基线。 |
| `head_aug_erm_seed666.pt` / `667` / `668` | 信道增强训练的小修正头；必须搭配 **original_cnn.pt**。它们不包含完整 CNN。 |
| `head_clean_erm_seed666.pt` / `667` / `668` | 原声训练的小修正头对照；同样搭配 **original_cnn.pt**。 |

文件大小、SHA256、所选轮次、各时间尺度的历史阈值见 [manifest.json](manifest.json)。所有权重已通过当前代码加载和真实音频推理核验，见 [验证记录](../docs/verification_published_weights.json)。

## 原 CNN 的身份

`original_cnn.pt` 对应旧文件名 `cnn_bigru_train_20261109_203335_588602_best.pt`，不是网格搜索中的其他 CNN，也不是随机初始化模型。

```text
SHA256: 36899e66bcf84b1f5aa8a6f2d57128eb80b077bc28eaa94d9bb8fbb713a2a428
参数: 911,521
seed: 666
训练: 完整 PS train 25,380 条
验证: 完整 PS dev 24,844 条
所选轮次: 7
```

原 checkpoint 保留历史 `protocol`、路径和 `interval_threshold=0.6` 等元数据。其中0.6仅属于旧区间导出设置；当前原CNN的160ms历史基线表用 **logit阈值0**。不能自动把这些旧字段当成新评估协议。

## 加载与推理

在仓库 `code/` 目录运行。以下阈值只用于复现已报告的历史结果；未来统一评估协议需要按它自己的开发集规则选阈值。

```powershell
# 校验原权重
Get-FileHash ../weights/original_cnn.pt -Algorithm SHA256

# 原CNN的160ms历史基线，阈值0
python -m cnn_ps.predict --audio ../data/example.wav --checkpoint ../weights/original_cnn.pt --resolution 0.16 --threshold 0 --output ../result/original_prediction.json

# 原CNN + 信道增强小修正头，seed666，20ms
python -m cnn_ps.predict --audio ../data/example.wav --checkpoint ../weights/original_cnn.pt --head ../weights/head_aug_erm_seed666.pt --resolution 0.02 --threshold 0.848010540008545 --output ../result/head_prediction.json

# 直接微调后的CNN，20ms；不再叠加小修正头
python -m cnn_ps.predict --audio ../data/example.wav --checkpoint ../weights/cnn_finetuned_seed666.pt --resolution 0.02 --threshold 0.2922912240028382 --output ../result/finetuned_prediction.json
```

若要复现160ms区间结果，必须同时切换 `--resolution` 和对应阈值，不能沿用20ms阈值。输入要求16kHz单声道。

Python中可直接使用：

```python
from cnn_ps.backbone import load_backbone
from cnn_ps.head import load_head

cnn, frontend = load_backbone("../weights/original_cnn.pt", device="cpu")
head = load_head("../weights/head_aug_erm_seed666.pt")
```

`load_backbone` 默认将CNN置于评估、冻结状态。`finetune.py` 会显式开放全部参数；不需要组员自己猜应该解冻哪些层。checkpoint 不含优化器续训状态，重新微调会建立新的优化器。

权重已经提供；音频、标签和大缓存仍需按根目录README准备。历史权重的模型选择规则没有被重写为新的事件F1协议。
