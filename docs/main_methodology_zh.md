# 主方案：冻结 WavLM 与边界辅助双分辨率 CNN-BiGRU

2026-11-16 将已完成的 `wavlm_frozen` 设为默认主线。保留原始权重与开发集阈值，工程切换没有新增训练或测试。选择依据是已查看的PS实验，不能称为新增盲测。

## 输入如何转换

16kHz单声道 → 训练时原声/20dB白噪声/轻微EQ轮换 → 冻结WavLM-Base-Plus最后一层768维特征 → 可训练LayerNorm768与Linear(768,80) → CNN-BiGRU → 双分辨率及边界辅助 → 每20ms一个假语音logit → 连续假帧合并成事件。

WavLM始终eval且不计算梯度，不使用分词器或转写文本。官方预处理不作波形标准化。其卷积总步长320样本、感受野400样本；整条录音先补到160ms存储网格，再两侧各补40样本，特征中心为159.5+320t样本。只对实际音频计算时长和监督，尾帧保留实际不足20ms的时长。

这80维是学习得到的特征坐标，并非Mel频带。WavLM与双向GRU均使用上下文，整个系统属于离线定位，不能称为流式系统。

## 哪些部分训练

WavLM冻结94,381,936参数。接入层63,056参数、CNN-BiGRU911,521参数、边界辅助模块25,493参数，合计1,000,070个可训练参数。CNN从随机初始化训练，不是冻结旧CNN再只训修正器。

保留20ms细分支、160ms可学习粗聚合、局部四头交叉注意力和分歧门控。融合的是相对CNN原分数的残差，粗分支不直接覆盖细分支；边界头预测相邻有效帧真假切换，同真值片段内约束相邻概率一致。当前不用相对位置偏置、可靠性门控、能量特征或短事件额外加权。

新增模块的详细公式与实际时长掩码沿用[Mel后端方法的第3、4节](mel_methodology_zh.md)，代码见 [dual_resolution.py](../code/cnn_ps/dual_resolution.py)。边界辅助与段内一致性只在训练中使用真值，推理不需要边界标签。

## 训练与评价

固定PS train6345/dev2484/eval3562，seed666；20轮，batch8，共15,880次更新。AdamW学习率0.001，weight decay0.0001，整体梯度裁剪5。每条训练录音每轮按确定性轮换使用一种信道；开发集和测试集只用原声。

损失为 `L20_final + L160_final + 0.5 L160_coarse + 0.1 Lboundary + 0.05 Lconsistency`。BCE按实际时长加权；完整定义保留在[后端说明](mel_methodology_zh.md)。没有直接优化可微事件F1。

每轮仅用开发集真实时长FPR≤5%确定阈值，再以该阈值下事件F1最大选择轮次。epoch0可选，平局取更早轮次；冻结后才测试。当前选中epoch19，logit阈值为 `0.35863286256790167`，使用float64比较 `score >= threshold`。测试集不重新校准。

主指标为事件F1@IoU≥0.5，一对一匹配后汇总计数。连续20ms假帧合并，末端裁到真实时长，不平滑、不补空隙、不移除短事件。另报事件P/R、≤320ms短事件召回、真实时长FPR、全真录音FAR、Frame F1、时长F1和EER。[评价标准](evaluation_protocol_v2.txt)。

## 结果和限制

| 指标（%） | 原Mel对照 | WavLM主方案 |
|---|---:|---:|
| 事件F1 | 55.65 | 62.53 |
| 事件Precision / Recall | 50.91 / 61.35 | 58.21 / 67.55 |
| ≤320ms短事件召回 | 65.30 | 66.86 |
| 真实时长误报率 | 4.02 | 6.38 |
| 全真录音误报率 | 10.14 | 9.32 |
| 20ms Frame F1 | 71.14 | 82.09 |
| 时长F1 | 70.97 | 82.00 |
| EER | 24.71 | 14.53 |

Mel控制组精确重现历史逐轮开发集归档及所选参数，排除了此前滤波器实验的控制组变动。事件F1提高6.89个百分点，但真实时长误报率增加2.37个百分点；开发集5%约束不会保证测试集≤5%。全部指标与审计见[结果JSON](results_wavlm_frozen_20261116.json)。

WavLM来自[微软公开模型](https://huggingface.co/microsoft/wavlm-base-plus)，使用9.4万小时外部预训练音频；固定revision与SHA见[权重清单](../weights/wavlm_manifest.json)。收益包含预训练表示和新增投影层，不能写成纯粹结构创新或等参数量比较。目前仅一个seed，测试子集已被分析，配对录音bootstrap不反映不同训练seed的不确定性；尚未验证跨数据集泛化或各模块在WavLM上的独立贡献。

## 实现入口

- 默认训练/冻结/评价：[main_method.py](../code/cnn_ps/main_method.py)，委托已审计的 [train_wavlm.py](../code/cnn_ps/train_wavlm.py)。
- 默认单音频推理：[predict_main.py](../code/cnn_ps/predict_main.py)，加载发布的WavLM后端与固定官方编码器。
- 冻结特征：[wavlm_frontend.py](../code/cnn_ps/wavlm_frontend.py)；投影与CNN：[wavlm_backend.py](../code/cnn_ps/wavlm_backend.py)。
- 原Mel对照入口：[predict_mel.py](../code/cnn_ps/predict_mel.py)、[mel_main_method.py](../code/cnn_ps/mel_main_method.py)。

安装、下载、训练命令见[README](../README.md)。公开WavLM编码器单独下载；仓库后端权重包含投影、CNN和全部新增模块，不能省略编码器或只加载CNN。
