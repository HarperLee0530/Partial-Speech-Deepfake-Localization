# PS 部分伪造语音定位：四组从随机初始化训练

**本轮四组模型从同一个随机初始化的 CNN-BiGRU 开始训练。正式结果尚未产生，当前没有发布成品权重。** 前两组是控制组，最后两组是候选方法，不加载旧 CNN、B1 或双分辨率权重。

| 组名 | 输入与结构 | 作用 |
|---|---|---|
| `cnn_clean` | CNN-BiGRU，只用原声 | 结构基线控制 |
| `cnn_aug` | 相同CNN，原声／20dB白噪声／轻微EQ轮换 | 信道增强控制 |
| `dual` | 增强＋20/160ms两分支＋局部交叉注意力＋分歧门控＋粗辅助监督 | 候选方法一 |
| `boundary` | 在`dual`上加入可学习降采样、边界BCE、同真值片段内部一致性 | 候选方法二 |

`dual`同时加入结构与粗辅助监督；`boundary`是三项改动的组合，首轮不能分别归因于其中一项。当前四组不使用位置偏置或可靠性门控扩展。

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
- 新LogMel缓存：`D:\SAL_精简实验\data\cnn_ps_features_v2\{train,dev}`，train三信道、dev仅原声，当前机器已生成。
- 本轮结果：`D:\SAL_精简实验\result\ps_four_eventf1_20261115_run1`。

清单不依赖旧result缓存，音频与官方160ms标签仍在原数据目录。输入16kHz单声道；实际20ms NPZ为**1=假**，官方160ms NPY为1=真，由读取器转换。不能重复粗标签伪造细标签。

## Windows运行顺序

同一PowerShell窗口内依次执行。组员修改本机路径，复现时另选新的`$run`，不要覆盖当前结果。

```powershell
$repo = 'D:\Partial-Speech-DualResolution-experiment'
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

**四组全部完成后先冻结，再生成eval缓存，最后统一测试。** 冻结检查还核对四组初始化、训练预算、数据和源码一致。

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

旧CNN/B1/dual权重已移出当前交付，旧结果按本轮清理要求移入回收站；原PS数据及SAL、语义、HAD、prosody材料保留。[权重清单](weights/manifest.json)当前`weights=[]`。正式训练、冻结和评估完成前不填写新成绩，不沿用旧成绩或旧阈值。

当前入口：[train_four.py](code/cnn_ps/train_four.py)、[dual_resolution.py](code/cnn_ps/dual_resolution.py)、[dual_features.py](code/cnn_ps/dual_features.py)。部分旧入口暂留用于共享函数与回溯，本轮只使用上述四组流程。
