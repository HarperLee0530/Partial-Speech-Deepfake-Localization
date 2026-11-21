# PS 部分伪造语音定位：冻结 WavLM＋边界辅助双分辨率 CNN-BiGRU

> **本分支说明：** `reproduce/s1-seed666-667-20261121` 仅归档已评价的 S1 研究候选与 A0 配对对照，**不将 S1 设为当前主基线**。代码入口、训练环境、冻结权重、结果及复现限制见 [S1 复现包](reproductions/s1_20261121/README.md)。

**当前默认主方案为 `wavlm_frozen`**：用冻结的 WavLM-Base-Plus 替换 Log-Mel 前端，再训练投影层、CNN-BiGRU 和双分辨率边界辅助模块。训练保留信道增强，输出原生 20 ms 伪造区间。这是离线定位模型。

2026-11-16 按已完成的对照结果切换工程主线；不重训、不重新选择轮次或测试阈值。原 Mel 方案及其权重保留作对照。[中文方法论](docs/main_methodology_zh.md) · [完整结果与审计](docs/results_wavlm_frozen_20261116.json) · [中文结果](docs/results_wavlm_frozen_20261116.txt)。

## 同条件测试结果

PS train 6,345 / dev 2,484 / eval 3,562，seed666，双方20轮、15,880次更新。每轮仅在开发集校准真实时长FPR≤5%，再最大化开发集事件F1；冻结后测试。Mel控制组逐轮开发集归档、选中参数及阈值与历史主方案完全一致。

| 方法 | 事件F1↑ | 事件Precision↑ | 事件Recall↑ | ≤320ms召回↑ | 真实时长FPR↓ | 全真录音误报率↓ | 20ms Frame F1↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原 Mel 边界辅助方案 | 55.65% | 50.91% | 61.35% | 65.30% | **4.02%** | 10.14% | 71.14% |
| **冻结 WavLM 主方案** | **62.53%** | **58.21%** | **67.55%** | **66.86%** | 6.38% | **9.32%** | **82.09%** |

事件F1提高6.89个百分点，配对录音bootstrap的95%区间为[5.63, 8.08]；真实时长误报率增加2.37个百分点。**5%只约束开发集，测试集6.38%如实报告。** 全真录音误报率差值区间跨0，不能宣称稳定下降。目前只有一个训练seed，测试子集此前已看过，这些区间不涵盖重训差异。

WavLM使用外部9.4万小时预训练音频，冻结94,381,936个参数；768→80接入层额外增加63,056个可训练参数，后端共1,000,070个可训练参数。这是预训练特征与接入层的整体收益，不能全部归因于我们自己的结构创新，也不是等参数量比较。

## 默认推理入口

进入 `code/`，安装依赖并下载固定版本的官方编码器。定位后端权重已经在本仓库；官方WavLM约378MB，单独下载并校验SHA，不需要上传音频。

```powershell
pip install -r requirements-wavlm.txt
python -m cnn_ps.download_wavlm --output MODEL_FOLDER
python -m cnn_ps.predict_main --audio example.wav --output prediction.json --model-folder MODEL_FOLDER --device cpu
```

默认加载 [cnn_wavlm_frozen_seed666.pt](weights/cnn_wavlm_frozen_seed666.pt)，包含投影层、CNN和完整边界辅助模块。使用选中epoch19的原生20ms **logit** 阈值 `0.35863286256790167`，不是概率阈值。输入16kHz单声道，连续假帧合并，不平滑、不填空隙、不过滤短事件。无需标签或训练目录。`--device cuda`可用GPU。

原Mel模型仍可运行：`python -m cnn_ps.predict_mel --audio example.wav --output prediction_mel.json --device cpu`。两种入口和权重格式不能混用。[权重清单](weights/README.md)。

## 主方案重新训练

[main_method](code/cnn_ps/main_method.py) 固定选择WavLM方案；[train_wavlm](code/cnn_ps/train_wavlm.py)提供WavLM与Mel配对实验。以下使用新的输出目录。`FEATURES`是下方流程准备的原始标签/时长及Mel参考缓存，`SSL`为WavLM缓存；后端训练读取WavLM特征，不把Mel特征混入输入。

```powershell
python -m cnn_ps.wavlm_frontend --reference FEATURES/train --model-folder MODEL_FOLDER --output SSL/train
python -m cnn_ps.wavlm_frontend --reference FEATURES/dev --model-folder MODEL_FOLDER --output SSL/dev
python -m cnn_ps.main_method train --train-cache FEATURES/train --dev-cache FEATURES/dev --train-ssl SSL/train --dev-ssl SSL/dev --output NEW_RUN --device cuda
python -m cnn_ps.main_method freeze --output NEW_RUN
# 完成开发集选模并冻结后，才准备eval参考缓存和WavLM特征
python -m cnn_ps.wavlm_frontend --reference FEATURES/eval --model-folder MODEL_FOLDER --output SSL/eval --frozen-plan NEW_RUN/frozen_eval_plan.json
python -m cnn_ps.main_method evaluate --eval-cache FEATURES/eval --eval-ssl SSL/eval --output NEW_RUN --device cuda
```

默认seed666、20轮、batch8。投影、CNN与新增模块全部更新，WavLM始终冻结。旧四组用 `train_four`；旧Mel单方案用 `mel_main_method`。已有配对实验的冻结计划不可改成单方案计划，应继续用原 `train_wavlm` 复评。缓存拒绝覆盖已有产物，详见[前端与对齐说明](docs/wavlm_frozen_experiment.txt)。

以下四组为历史Mel结构对照，不能用它们证明WavLM上的各个模块都独立有效。下一步需多seed与WavLM骨干上的模块消融。

## 历史 Mel 四组对照（seed666）

四组均训练20轮，完整独立审计通过，测试使用各自已经冻结的开发集阈值。下表为同一3562条PS录音的原生20ms评价，主指标为事件F1@IoU0.5。

| 组别 | 角色 | 事件F1↑ | ≤320ms召回↑ | 真实时长FPR↓ | 全真录音误报率↓ | Frame F1↑ |
|---|---|---:|---:|---:|---:|---:|
| `cnn_clean` | 控制 | 53.22% | 63.83% | 6.40% | 25.21% (92/365) | 74.08% |
| `cnn_aug` | 控制 | 50.21% | 56.58% | 4.94% | 12.33% (45/365) | 67.62% |
| `dual` | 结构对照 | 54.98% | 61.39% | 6.03% | 15.62% (57/365) | 73.38% |
| `boundary` | Mel 对照 | 55.65% | 65.30% | 4.02% | 10.14% (37/365) | 71.14% |

`dual`相对增强控制的事件F1增加4.7709个百分点；按录音配对bootstrap的95%区间为[3.9094, 5.5684]。`boundary`的事件F1、≤320ms召回和上述两类误报率点估计最好，但相对`dual`的事件F1仅增加0.6631个百分点，95%区间[-0.1059, 1.4519]跨0，额外增益尚不确定。该bootstrap只重采样录音，不涵盖训练seed差异；仅有seed666，eval此前已看过。

边界组合也有代价：事件Recall61.35%和Frame F1 71.14%低于`cnn_clean`的62.14%和74.08%；短事件改善主要集中≤80ms，80–160ms及160–320ms召回低于`dual`。未匹配预测事件从`dual`的6634增加到6760，不能说所有误报计数都下降或已经证明减少碎片化。仅信道增强本轮降低部分误报，同时降低事件F1，不能沿用旧实验结论。

详见[中文结果分析](docs/results_four_eventf1_20261115.txt)、[完整数值与审计摘要](docs/results_four_eventf1_20261115.json)。四个开发集选中权重位于[weights目录](weights/README.md)，文件SHA已核对。

## 历史 Mel 四组共同条件与评价

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

历史Mel单音频推理使用`predict_mel`，无需原运行目录。下列历史checkpoint**不能直接交给旧`predict`或`load_backbone`**。完整复评还需同一数据及`complete.json`、`protocol.json`、dev指标/分数和冻结计划等结果元信息，仅下载`.pt`不足以运行`--evaluate`。阈值是原生20ms、假为正类的**logit阈值**，以float64执行`score >= threshold`；不是sigmoid概率阈值，不默认0或0.5，不使用表格舍入值。

历史Mel入口：[mel_main_method.py](code/cnn_ps/mel_main_method.py)、[predict_mel.py](code/cnn_ps/predict_mel.py)。共同实现与对照入口：[train_four.py](code/cnn_ps/train_four.py)、[dual_resolution.py](code/cnn_ps/dual_resolution.py)、[dual_features.py](code/cnn_ps/dual_features.py)。部分旧入口暂留用于共享函数与回溯。
