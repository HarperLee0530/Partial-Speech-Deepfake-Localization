# PS 部分伪造语音定位：CNN-BiGRU＋信道增强直接微调

**唯一主方案：从最初原 CNN-BiGRU 权重开始，在增强音频上直接微调原网络全部参数，输出20 ms真假分数与伪造区间。** 原CNN保留为训练起点和对照。主分支仅维护这条训练、推理和评估流程。

## 方法

```text
训练音频 → 原声 / 20 dB白噪声 / 轻微EQ（逐轮轮换）
         → Log-Mel80（20 ms）→ CNN（16/32/64）→ 两层BiGRU
         → 原线性分类层 → 每20 ms的伪造分数
```

- 直接更新原模型全部 **911,521** 个参数，包括CNN、BiGRU和原分类层。
- 增强发生在输入波形：噪声按整段RMS缩放至20 dB；EQ为 `x[n]+0.15*(x[n]-x[n-1])`，首个previous取首采样。时长和真假标签保持一致。
- 每条录音每轮出现一次，三种信道固定分配并逐轮轮换；开发集和测试集使用原声。
- 损失为 **真实20 ms时长加权BCE＋160 ms时长加权BCE**，系数各为1。160 ms分数是8个原生logit的均值；主要定位输出保留20 ms。
- 已完成设置：seed666，8轮，整段音频，batch8，AdamW学习率0.0001、weight decay0.0001，梯度裁剪5。
- 为复现已发布权重，当前训练仍按开发集 `J=0.5×时长F1+0.5×事件F1` 选模，epoch0可选，平局取更早轮。本次保留的权重选中第8轮。后续改为纯事件F1选模的方案见评价规范，尚未作为本次成绩的训练规则。

## 已完成的同口径结果

固定 **PS dev2484 / eval3562**。使用真实20 ms标签，假类为正；每个模型分别在同一dev按真实时长FPR≤5%确定阈值，冻结后评估eval。事件IoU≥0.5，一对一匹配；短事件≤320 ms；不增加平滑、补空隙或短事件删除。

| 指标 | 最初原CNN | 保留方案：信道增强直接微调 |
|---|---:|---:|
| **事件F1 ↑（主指标）** | 31.7508% | **50.2401%** |
| 事件Precision ↑ | 25.5042% | 42.5432% |
| 事件Recall ↑ | 42.0495% | 61.3372% |
| ≤320 ms事件召回 ↑ | 44.5932% | **62.9841%** |
| 真实时长误报率 ↓ | **4.5105%** | 6.0467% |
| 全真录音误报率 ↓ | 37.8082% | **18.9041%** |
| 20 ms Frame F1 ↑（补充） | 61.1897% | 74.9082% |
| 20 ms EER ↓（补充） | 26.5678% | 21.1291% |

事件F1提高 **18.4894个百分点**，短事件召回提高 **18.3910个百分点**，同时真实时长误报率增加1.5361个百分点。365条全真录音的报警数量由138降至69；全真报警条数与真实时长误报率是不同指标。开发集5%预算不保证测试集也低于5%。

这是一个适配seed的结果，且原CNN只使用160 ms监督，微调额外使用真实20 ms监督，因此不能把全部提升归因于信道增强。评估为已使用过的3562条子集，并非完整71237条PS测试集或新的盲测集。

[完整重评结果与计数](docs/results_native20_fixed_reevaluation.json) · [独立复算](docs/verification_native20_fixed_reevaluation.json) · [原微调训练记录](docs/results_cnn_finetune_seed666.json) · [统一评价规范](docs/evaluation_protocol_v2.txt)

## 两个权重分别怎么用

| 文件 | 用途 | 本次20 ms开发集阈值 |
|---|---|---:|
| [original_cnn.pt](weights/original_cnn.pt) | 原基线；重新微调的起点 | 4.5316390991210946 |
| [cnn_finetuned_seed666.pt](weights/cnn_finetuned_seed666.pt) | 当前选定的成品模型 | 0.2922912240028382 |

权重随仓库提供，SHA256见 [manifest](weights/manifest.json)，加载说明见 [weights/README.md](weights/README.md)。阈值是分数阈值，不是概率；不能混用两个模型或两种时间尺度的阈值。重新训练后应使用新模型自己的开发集阈值。

## 环境与数据

Python3.10，依赖为PyTorch、torchaudio、NumPy、SoundFile、SciPy、scikit-learn。进入仓库后：

```powershell
cd code
.\Setup.cmd
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

下文从 `code/` 执行。`python` 表示已激活的Python环境；未激活时用 `.\.venv\Scripts\python.exe`。Setup仅安装依赖，不下载数据或启动训练。GPU需要匹配的CUDA版本；可显式选择CPU。

数据不随Git提交，官方来源见 [PS资源清单](code/manifests/ps_sources.json)。组内布局：`data/reduced/<role>/audio/<uid>.wav`，各role的 `segment_labels/` 内保存官方160 ms标签。准备固定PS清单和实际细标签：

```powershell
python -m cnn_ps.manifests reduced --reduced-root ../data/reduced --output ../data/manifests_ps
python -m cnn_ps.manifests labels20 --archive ../data/database_segment_labels_v1.2.tar.gz --output ../data/labels20
```

细标签工具校验官方归档SHA256，输出NPZ为 **1=假、0=真**；官方原始粗标签为1=真。不得将160 ms标签重复8遍代替真实20 ms标签。音频要求16kHz单声道。

原CNN在完整PS train25380/dev24844上训练选模；微调使用固定train6345（train_detector5076＋train_edit1269）、dev2484、eval3562，`train_review`不参与。这里的角色名只用于寻找组内文件。不能将重新在6345条上训练的CNN视为同一个历史基线。

## 使用成品模型

```powershell
python -m cnn_ps.predict --audio ../data/example.wav --checkpoint ../weights/cnn_finetuned_seed666.pt --resolution 0.02 --threshold 0.2922912240028382 --output ../result/example.json
```

输出包含原生分数、预测区间及实际时长裁剪。为保留旧调用兼容性，`predict`默认仍为160 ms和未校准阈值0；**当前方案请按上例显式传20 ms及开发集阈值**。

## 从原CNN直接微调

```powershell
python -m cnn_ps.finetune --checkpoint ../weights/original_cnn.pt --expected-sha256 36899e66bcf84b1f5aa8a6f2d57128eb80b077bc28eaa94d9bb8fbb713a2a428 --train-manifest ../data/manifests_ps/train.json --dev-manifest ../data/manifests_ps/dev.json --train-labels20 ../data/labels20/train.npz --dev-labels20 ../data/labels20/dev.npz --output ../result/cnn_finetune1 --device cuda
```

直接读取波形，不需要小修正头或其特征缓存。训练只访问train/dev；输出保存权重、逐轮开发集记录、开发集分数和阈值，拒绝覆盖非空目录。此命令保留已完成实验的训练与旧J选模行为。

## 统一评估成品与原基线

只有下载的权重也可以评估。以下入口先重新推理原声dev、按5%真实时长预算校准并保存阈值，然后才读取eval。主报告为 `metrics20.json`，160 ms辅助报告另按dev Frame F1校准阈值；不重选checkpoint。

```powershell
python -m cnn_ps.evaluate_cnn --checkpoint ../weights/cnn_finetuned_seed666.pt --dev-manifest ../data/manifests_ps/dev.json --dev-labels20 ../data/labels20/dev.npz --eval-manifest ../data/manifests_ps/eval.json --eval-labels20 ../data/labels20/eval.npz --output ../result/main_eval1 --device cuda
python -m cnn_ps.evaluate_cnn --checkpoint ../weights/original_cnn.pt --dev-manifest ../data/manifests_ps/dev.json --dev-labels20 ../data/labels20/dev.npz --eval-manifest ../data/manifests_ps/eval.json --eval-labels20 ../data/labels20/eval.npz --output ../result/baseline_eval1 --device cuda
```

对于本机新完成的微调目录，也可把 `--checkpoint ...` 替换为 `--run ../result/cnn_finetune1`，核验并沿用该run保存的开发集分数与阈值；两选项互斥。跨设备、库版本的重推理可能有微小浮点差异；已发表的固定权重重评数值使用相应存档分数，可按文档中的哈希复核。

## 代码位置

| 文件（`code/cnn_ps/`） | 职责 |
|---|---|
| `backbone.py`、`baseline.py` | 原CNN-BiGRU、Log-Mel、原基线从头训练与collate |
| `augmentation.py`、`finetune.py` | 波形信道增强、全部CNN参数直接微调 |
| `predict.py`、`evaluate_cnn.py` | 单音频推理、原CNN/成品模型统一评估 |
| `metrics20.py`、`report_native20.py` | 20 ms阈值、事件/时长/录音误报及补充指标 |
| `metrics160.py` | 秒单位事件匹配与160 ms历史辅助指标 |
| `manifests.py`、`io.py` | 固定PS划分、细标签提取、路径与文件校验 |

## 历史对照

小修正头、缓存训练及三方案回放专用代码和六个头权重已移出当前主树。其完整代码、权重可在[清理前提交](https://github.com/HarperLee0530/Partial-Speech-Deepfake-Localization/tree/51b20490d8b795fad13d83375ade72b25042b764)回查。

已验证的对照数字和审计记录保留在 `docs/`，仅供报告引用：[历史记录说明](docs/history_index.txt)。SAL只作为[历史参考结果](docs/native20_sal_reference.txt)保留；它在EER和全真录音误报方面有优势，不应将当前CNN选择描述为所有指标全面超过SAL。

本次整理的[验证记录](docs/verification_direct_finetune_cleanup.json)：19项测试通过；两个权重SHA不变；原CNN/成品的真实录音推理与清理前一致，直接checkpoint评估的开发集校准、阈值冻结顺序和分数复算通过。未重新训练模型。
