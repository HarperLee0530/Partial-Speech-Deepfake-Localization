# PS 部分伪造语音定位：边界辅助双分辨率 CNN-BiGRU

本实验分支已完成[冻结 WavLM 特征对照](docs/results_wavlm_frozen_20261116.txt)。原 Mel 控制组完整复现历史训练记录和选中参数，测试事件 F1 为 **55.65%**；冻结 WavLM＋可训练投影层后为 **62.53%**，20 ms Frame F1 从 **71.14%** 提高到 **82.09%**，但真实时长误报率从 **4.02%** 增至 **6.38%**。原主方案继续保留，实验结果尚未用于替换默认入口。

这是单 seed、额外预训练特征与接入层的整体比较：WavLM-Base-Plus 冻结约 9438 万参数，额外使用 9.4 万小时预训练音频，后端新增 63056 个可训练参数。开发集的 5% 误报约束不保证测试集也低于 5%。[完整结果与区间](docs/results_wavlm_frozen_20261116.json) · [方法与预检](docs/wavlm_frozen_experiment.txt) · [两组后端权重清单](weights/wavlm_manifest.json)。

实验推理使用 [`predict_wavlm`](code/cnn_ps/predict_wavlm.py)，进入 `code/`：

```powershell
pip install -r requirements-wavlm.txt
python -m cnn_ps.download_wavlm --output MODEL_FOLDER
python -m cnn_ps.predict_wavlm --audio example.wav --output prediction.json --checkpoint ../weights/cnn_wavlm_frozen_seed666.pt --model-folder MODEL_FOLDER --device cpu
```

官方 WavLM 权重由下载脚本按固定 revision 和 SHA 校验，不随本仓库的定位后端权重一起存放。原主方案仍使用下方说明中的 `predict_main`。

**当前主方案为 `boundary`：CNN-BiGRU＋信道增强＋双分辨率、局部交叉注意力、分歧门控＋边界辅助组合。** CNN与新增模块全部端到端训练，输出原生20ms伪造区间。

主方案在本轮PS测试上的事件F1为 **55.65%**、≤320ms召回为 **65.30%**、真实时长误报率为 **4.02%**。选择它作为工程主线，是查看现有四组结果后的决定；这次整理不重训、不重选epoch或阈值，不能当作新增盲测。完整[中文方法论](docs/main_methodology_zh.md)包含结构图、公式、损失、训练评价与结论边界。

四组均从相同随机CNN初始化训练20轮，统一测试和独立审计已完成。其余三组保留为对照，数值和原始权重保持不变。

| 组名 | 输入与结构 | 作用 |
|---|---|---|
| `cnn_clean` | CNN-BiGRU，只用原声 | 结构基线控制 |
| `cnn_aug` | 相同CNN，原声／20dB白噪声／轻微EQ轮换 | 信道增强控制 |
| `dual` | 增强＋20/160ms两分支＋局部交叉注意力＋分歧门控＋粗辅助监督 | 结构对照 |
| `boundary` | 在`dual`上加入可学习降采样、边界BCE、同真值片段内部一致性 | **当前主方案** |

`dual`同时加入结构与粗辅助监督；`boundary`是三项改动的组合，首轮不能分别归因于其中一项。当前四组不使用位置偏置或可靠性门控扩展。

## 主方案入口

在已安装依赖的环境中进入`code/`，下载仓库中的主方案权重后可直接定位单条音频：

```powershell
python -m cnn_ps.predict_main --audio 'example.wav' --output 'prediction_main.json' --device cpu
```

默认加载[主方案权重](weights/cnn_boundary_ps_eventf1_seed666.pt)，包括CNN和全部新增模块，使用checkpoint中冻结的开发集logit阈值`0.42304700613021856`。输入须为16kHz单声道；输出实际20ms分数与伪造时间区间，不需要标签或原训练目录。与旧`cnn_ps.predict`区分，新入口不会遗漏adapter，也不使用默认0/0.5阈值。

重新训练主方案使用专门入口，固定选择`boundary`，训练、校准和评价逻辑复用原对照实现。以下示例使用**新输出目录**；缓存准备见下方完整流程。

```powershell
python -m cnn_ps.main_method train --train-cache 'D:\SAL_精简实验\data\cnn_ps_features_v2\train' --dev-cache 'D:\SAL_精简实验\data\cnn_ps_features_v2\dev' --output 'D:\SAL_精简实验\result\ps_boundary_main_newrun' --device cuda
python -m cnn_ps.main_method freeze --output 'D:\SAL_精简实验\result\ps_boundary_main_newrun'
python -m cnn_ps.main_method evaluate --eval-cache 'D:\SAL_精简实验\data\cnn_ps_features_v2\eval' --output 'D:\SAL_精简实验\result\ps_boundary_main_newrun' --device cuda
```

默认seed666、20轮、batch8。只训练主方案时，输出目录包含`boundary/`和单方案冻结计划；已有四组实验继续用`train_four`及其原四组冻结计划。`predict_main`只做单音频推理；正式指标须用带真值的统一评价流程计算。

## 本轮测试结果（seed666）

四组均训练20轮，完整独立审计通过，测试使用各自已经冻结的开发集阈值。下表为同一3562条PS录音的原生20ms评价，主指标为事件F1@IoU0.5。

| 组别 | 角色 | 事件F1↑ | ≤320ms召回↑ | 真实时长FPR↓ | 全真录音误报率↓ | Frame F1↑ |
|---|---|---:|---:|---:|---:|---:|
| `cnn_clean` | 控制 | 53.22% | 63.83% | 6.40% | 25.21% (92/365) | 74.08% |
| `cnn_aug` | 控制 | 50.21% | 56.58% | 4.94% | 12.33% (45/365) | 67.62% |
| `dual` | 结构对照 | 54.98% | 61.39% | 6.03% | 15.62% (57/365) | 73.38% |
| `boundary` | **主方案** | 55.65% | 65.30% | 4.02% | 10.14% (37/365) | 71.14% |

`dual`相对增强控制的事件F1增加4.7709个百分点；按录音配对bootstrap的95%区间为[3.9094, 5.5684]。`boundary`的事件F1、≤320ms召回和上述两类误报率点估计最好，但相对`dual`的事件F1仅增加0.6631个百分点，95%区间[-0.1059, 1.4519]跨0，额外增益尚不确定。该bootstrap只重采样录音，不涵盖训练seed差异；仅有seed666，eval此前已看过。

边界组合也有代价：事件Recall61.35%和Frame F1 71.14%低于`cnn_clean`的62.14%和74.08%；短事件改善主要集中≤80ms，80–160ms及160–320ms召回低于`dual`。未匹配预测事件从`dual`的6634增加到6760，不能说所有误报计数都下降或已经证明减少碎片化。仅信道增强本轮降低部分误报，同时降低事件F1，不能沿用旧实验结论。

详见[中文结果分析](docs/results_four_eventf1_20261115.txt)、[完整数值与审计摘要](docs/results_four_eventf1_20261115.json)。四个开发集选中权重位于[weights目录](weights/README.md)，文件SHA已核对。

## 共同条件与评价

- 固定PS **train6345 / dev2484 / eval3562**，seed666，20个训练epoch，batch8；epoch0只验证，也可被选中。
- 结构：Log-Mel80 → CNN(16/32/64) → 两层双向GRU(hidden128) → 逐帧分类；dropout0.4，CNN共911,521参数。四组初始CNN张量完全相同，全部参数从头更新。
- CNN和新增模块统一AdamW lr=`1e-3`、weight decay=`1e-4`、梯度裁剪5，训练数据顺序相同。`cnn_clean`始终取原声，其他组按相同UID分配与epoch轮换信道。
- 新增模块使用**可训练LayerNorm256**，不读取旧CNN特征mean/std或旧normalization文件；标准化后截断至[-5,5]。
- 共同损失为实际时长加权20ms BCE＋最终160ms BCE；两个候选方法另加0.5倍粗辅助BCE；`boundary`再加0.1倍边界BCE＋0.05倍片段内一致性。
- 每轮仅在干净**dev**上按真实时长FPR≤5%确定阈值，再按该阈值下的dev**事件F1@IoU0.5最大**选择权重，平局取更早epoch。测试集绝不用于校准或选模。
- 统一原生20ms评价，辅报≤320ms短事件召回、真实时长误报率、全真录音误报率和Frame F1/EER。5%只约束dev，不保证eval也≤5%。

这是新的统一结构实验，不是旧成品复评。旧成绩不放入当前结果表。当前eval子集此前已使用过，不能称为新盲测；单seed不足以证明稳定增益。详见[本轮规范](docs/dual_resolution_experiment.txt)与[评价定义](docs/evaluation_protocol_v2.txt)。

## 环境和独立输入

Python3.10，依赖见`code/requirements.txt`。进入`code/`后可用`Setup.cmd`建立环境；GPU运行需匹配的PyTorch/CUDA。

- 永久输入：`D:\SAL_精简实验\data\cnn_ps_inputs_v2\manifests\{train,dev,eval}.json`及`labels20\{train,dev,eval}.npz`。
- 新LogMel缓存：`D:\SAL_精简实验\data\cnn_ps_features_v2\{train,dev,eval}`；train三信道，dev/eval仅原声。当前运行的eval缓存是在四组开发集选择冻结之后生成。
- 本轮结果：`D:\SAL_精简实验\result\ps_four_eventf1_20261115_run1`。

清单不依赖旧result缓存，音频与官方160ms标签仍在原数据目录。输入16kHz单声道；实际20ms NPZ为**1=假**，官方160ms NPY为1=真，由读取器转换。不能重复粗标签伪造细标签。

## 四组对照的完整复现流程

同一PowerShell窗口内依次执行。组员修改本机路径，复现时另选新的`$run`，不要覆盖当前结果。

```powershell
$repo = 'D:\Partial-Speech-CNN-cleanup'
$inputs = 'D:\SAL_精简实验\data\cnn_ps_inputs_v2'
$features = 'D:\SAL_精简实验\data\cnn_ps_features_v2'
$run = 'D:\SAL_精简实验\result\ps_four_eventf1_20261115_run1'
Set-Location -LiteralPath (Join-Path $repo 'code')
$python = Join-Path $repo 'code\.venv\Scripts\python.exe'
# 本机也可使用已有环境：D:\SAL_精简实验\code\.venv\Scripts\python.exe
$config = Join-Path $repo 'code\configs\cnn_ps_structure.json'
$trainCache = Join-Path $features 'train'
$devCache = Join-Path $features 'dev'
$evalCache = Join-Path $features 'eval'
& $python -B -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw '测试未通过' }
```

先生成train三信道和dev单原声LogMel。**当前机器这两个新缓存已完成，可跳过生成；生成器拒绝覆盖非空目录。** 这里只读[纯结构配置](code/configs/cnn_ps_structure.json)，不读checkpoint、不运行CNN。此时不要生成eval缓存。

```powershell
& $python -B -m cnn_ps.dual_features --model-config $config --manifest (Join-Path $inputs 'manifests\train.json') --labels20 (Join-Path $inputs 'labels20\train.npz') --output $trainCache --split train --channels 3 --cpu-threads 1 --seed 666
if ($LASTEXITCODE -ne 0) { throw 'train缓存未完成' }
& $python -B -m cnn_ps.dual_features --model-config $config --manifest (Join-Path $inputs 'manifests\dev.json') --labels20 (Join-Path $inputs 'labels20\dev.npz') --output $devCache --split dev --channels 1 --cpu-threads 1 --seed 666
if ($LASTEXITCODE -ne 0) { throw 'dev缓存未完成' }
```

四组独立启动、共享相同训练参数；以下按顺序运行，也可单独选择尚未启动的一组。训练不接收eval缓存、旧权重或归一化文件。

```powershell
$trainArgs = @('-B', '-m', 'cnn_ps.train_four', '--train-cache', $trainCache, '--dev-cache', $devCache, '--output', $run, '--epochs', '20', '--batch-size', '8', '--seed', '666', '--device', 'cuda', '--threads', '4')
& $python @trainArgs --arm cnn_clean
if ($LASTEXITCODE -ne 0) { throw 'cnn_clean未完成' }
& $python @trainArgs --arm cnn_aug
if ($LASTEXITCODE -ne 0) { throw 'cnn_aug未完成' }
& $python @trainArgs --arm dual
if ($LASTEXITCODE -ne 0) { throw 'dual未完成' }
& $python @trainArgs --arm boundary
if ($LASTEXITCODE -ne 0) { throw 'boundary未完成' }
```

**四组全部完成后先冻结，再生成或复用eval缓存，最后统一测试。** 冻结检查还核对四组初始化、训练预算、数据和源码一致。当前机器的eval缓存已完成，复现时可在冻结后复用；已有缓存时跳过下方生成命令，评估入口仍会校验缓存哈希。

```powershell
& $python -B -m cnn_ps.train_four --freeze-only --output $run --device cpu
if ($LASTEXITCODE -ne 0) { throw '冻结未通过，停止读取eval' }
& $python -B -m cnn_ps.dual_features --model-config $config --manifest (Join-Path $inputs 'manifests\eval.json') --labels20 (Join-Path $inputs 'labels20\eval.npz') --output $evalCache --split eval --channels 1 --cpu-threads 1 --seed 666
if ($LASTEXITCODE -ne 0) { throw 'eval缓存未完成' }
& $python -B -m cnn_ps.train_four --evaluate --eval-cache $evalCache --output $run --device cuda --threads 4
if ($LASTEXITCODE -ne 0) { throw '统一评估未完成' }
```

`--evaluate`只使用冻结dev阈值，不搜索测试工作点；拒绝替换不同的冻结计划或覆盖已完成测试报告。

## 每轮存档与权重状态

每组`$run/<arm>/epochs/`保存`epoch_00`至`epoch_20`的完整CNN/新增模块权重、dev阈值、指标和原始分数。`best.pt`、`dev_metrics.json`、`dev_scores20.npz`对应dev选中轮，`complete.json`记录哈希；统一测试后才生成`comparison.json`及各组`eval_metrics.json`。

逐轮保存用于检查学习过程，或在另行明确的**dev-only**协议下重选。不能看eval后挑更好的epoch并称为原协议结果。

旧CNN/B1/dual权重已移出当前交付，旧结果按本轮清理要求移入回收站；原PS数据及SAL、语义、HAD、prosody材料保留。[权重清单](weights/manifest.json)登记四个新选中权重及完整SHA、epoch、各自dev阈值和配置来源，详见[权重使用说明](weights/README.md)。

主方案单音频推理使用`predict_main`，无需原运行目录。本轮checkpoint**不能直接交给旧`predict`或`load_backbone`**。完整复评还需同一数据及`complete.json`、`protocol.json`、dev指标/分数和冻结计划等结果元信息，仅下载`.pt`不足以运行`--evaluate`。阈值是原生20ms、假为正类的**logit阈值**，以float64执行`score >= threshold`；不是sigmoid概率阈值，不默认0或0.5，不使用表格舍入值。

主方案入口：[main_method.py](code/cnn_ps/main_method.py)、[predict_main.py](code/cnn_ps/predict_main.py)。共同实现与对照入口：[train_four.py](code/cnn_ps/train_four.py)、[dual_resolution.py](code/cnn_ps/dual_resolution.py)、[dual_features.py](code/cnn_ps/dual_features.py)。部分旧入口暂留用于共享函数与回溯。
