# A组消融代码与运行说明

本组只拆解冻结WavLM的特征选择/融合、当前任务后端和损失。编码器内部结构保持不变；没有执行B组实验。

## 文件

- `cnn_ps/wavlm_ablation_a.py`：11组开关、模型与独立损失系数。
- `cnn_ps/train_wavlm_ablation_a.py`：独立训练、开发集选择、冻结和测试入口。
- `tests/test_wavlm_ablation_a.py`：公共初始化、旧实现等价、开关隔离与层选择检查。
- `verify_wavlm_ablation_a.py`：11组train/dev冒烟训练，以及A0/A1全开发集复用核对。
- `run_wavlm_ablation_a.py`：固定矩阵编排；最多两个独立进程；全部完成才允许测试。
- `benchmark_wavlm_ablation_a.py`：完整原始音频推理核对及顺序计时。
- `report_wavlm_ablation_a.py`：完整指标、配对bootstrap区间、图和中文报告。

以上均为新增文件，旧版训练和预测入口没有改动。

## 固定矩阵

| 组别 | 相对于A0的变化 |
|---|---|
| A0_full | 全开：6/9/12层softmax加权，所有当前模块和损失开启 |
| A1_layer12 | 只取第12层 |
| A2_layer6 | 只取第6层 |
| A3_layer9 | 只取第9层 |
| A4_mean | 6/9/12层固定各1/3，不学习层权重 |
| A5_fixed_pool | 可学习降采样改为有效时长加权平均 |
| A6_no_attention | 不计算局部跨分辨率注意力，残差置零；粗分支和门控保留 |
| A7_fixed_gate | 门控固定0.5；注意力保留 |
| A8_no_boundary | 边界头绕过并冻结；边界损失权重0；一致性保留 |
| A9_no_consistency | 一致性权重0；边界头/损失保留 |
| A10_no_coarse_aux | 粗分支辅助损失权重0；粗分支和主160ms损失保留 |

完整损失权重：main20=1、main160=1、coarse_aux=0.5、boundary=0.1、consistency=0.05。
除对应开关外均保持原值。为保持公共初始化，先创建完整适配器，然后冻结并绕过无效分支，checkpoint仍保留这些参数。

各组不是从A0已训练权重微调，而是从配对随机初始化开始完整训练。A0/A1可复用上一轮同配置、同预算结果，但必须先通过旧新实现CPU/CUDA单步以及全开发集逐元素一致性核对；复用检查记录保存在结果目录。

## 本机运行

在项目`code`目录使用已有`.venv/Scripts/python.exe`。依赖沿用现有环境，未改变版本。

```powershell
$env:PYTHONIOENCODING='utf-8'
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -u verify_wavlm_ablation_a.py
.venv/Scripts/python.exe -u run_wavlm_ablation_a.py --workers 2
```

验证脚本会创建新的冒烟目录，已有结果时不应重复运行。完整编排同样拒绝覆写未完成组：若异常中断，先保留日志、定位原因，再将该失败组明确移到本实验目录内的隔离目录，从头按同一预算重训；不将部分训练当作20轮对照，也不自动恢复优化器。

缓存复用工作区已有`handoff_audit/multilayer_20261117/full/reference/{train,dev,eval}`和`layers/{train,dev,eval}`。编码器来自项目`data/models/wavlm-base-plus`，不联网下载。路径由脚本位置确定。

结果位于工作区`handoff_audit/ablation_a_20261118`：

- `status.json`：总体阶段、活动进程、完成及待运行组。
- `logs/<组别>.log`、`training/<组别>/progress.json`：运行进度。
- `training/<组别>/best.pt`：开发集选出的最佳权重；不是测试集最优轮次。
- `training/<组别>/history.json`：epoch0至epoch20的开发集结果和训练损失。
- `experiment_plan.json`：训练前固定方案。
- `frozen_all_arms.json`：11组全部完成后、测试前的冻结记录。
- `A组消融验证报告.md`：全部验证结果、指标解释和局限。
- `metrics.csv`、`paired_bootstrap.json`：表格及条件贡献区间。
- `inference_benchmark.json`、`single_audio_verification.json`：32条固定开发音频的完整推理和缓存一致性检查。

编排器运行期间通过Windows线程电源请求阻止系统自动睡眠，结束或异常时释放；不修改永久电源计划。

## 单独复现实验

独立训练入口要求全新的输出子目录；不要复用已经归档的本轮实验目录。

```powershell
.venv/Scripts/python.exe -m cnn_ps.train_wavlm_ablation_a `
  --arm A6_no_attention `
  --train-cache ../../handoff_audit/multilayer_20261117/full/reference/train `
  --dev-cache ../../handoff_audit/multilayer_20261117/full/reference/dev `
  --train-ssl ../../handoff_audit/multilayer_20261117/full/layers/train `
  --dev-ssl ../../handoff_audit/multilayer_20261117/full/layers/dev `
  --output ../../handoff_audit/ablation_a_reproduction/training `
  --epochs 20 --batch-size 8 --seed 666 --threads 4
```

默认CUDA。所有组训练完成之后，再用相同入口`--evaluate --arms <全部需要比较的新训练组>`，并给出`--eval-cache`、`--eval-ssl`和对应`--output`。该入口核验完成状态、预算、源码与checkpoint摘要后才读取eval。A0/A1复用记录由编排器单独审计，不能将其旧schema目录直接传给新训练入口的冻结器。

## 加载最佳模型

读取`best.pt`中的`model_config`，用`AblationCNN(config, arm, seed)`及`make_adapter(ARMS[arm], seed, device)`创建结构，然后严格加载`backbone_state_dict`、`adapter_state_dict`。组名必须与文件夹匹配，特别是复用的A0/A1 checkpoint内部保留原始组名作为来源证据。

编码器保持`eval()`且`requires_grad_(False)`。输入为16kHz单声道完整录音，按原160ms存储网格补零；冻结编码器使用左右40采样点对齐。第6/9/12层分别是Transformer第6/9/12层的输出，隐藏状态0不属于其中。单层组选择对应768维特征，多层组按`T×3×768`顺序展平为2304维，交给模型完成融合。

阈值直接读取checkpoint，不在预测音频或测试集上重校准。按`float64(score) >= float64(threshold)`判定，连续正帧组成事件，无额外平滑、填缝或短事件删除。完整可执行加载、原始音频推理示例见`benchmark_wavlm_ablation_a.py`。

## 结论边界

主要比较事件F1@IoU0.5，辅助查看精确率、召回率、≤320ms事件召回、真实时长误报率、20ms帧F1和EER。测试EER作为阈值无关的汇总指标计算，不用于部署阈值选择。

“A0−关闭一个模块”表示该模块在其余配置保留时的条件贡献，不代表各贡献可相加。单种子配对bootstrap只估计录音采样不确定性，不能替代多种子重训。本轮不依据测试结果继续调整模型，也不自动执行B组或上传GitHub。
