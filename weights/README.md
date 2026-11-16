# 主方案与对照权重

当前主方案为 **wavlm_frozen**：[cnn_wavlm_frozen_seed666.pt](cnn_wavlm_frozen_seed666.pt)，epoch19，原生20ms logit阈值 `0.35863286256790167`。SHA256：`b52127f2c5564e64fbd3589a2203f6d8719be60a1c1546d59e79f5b295d747b2`。

后端文件含可训练投影、CNN和全部边界辅助模块，schema为 `ps_frozen_wavlm_cnn_v1`。它不含冻结的公开WavLM编码器；进入code目录，安装 `requirements-wavlm.txt`，执行：

```powershell
python -m cnn_ps.download_wavlm --output MODEL_FOLDER
python -m cnn_ps.predict_main --audio example.wav --output prediction.json --model-folder MODEL_FOLDER --device cpu
```

固定官方模型revision、SHA、后端SHA及配对Mel权重见 [wavlm_manifest.json](wavlm_manifest.json)。配对Mel文件 [cnn_mel_control_wavlm_trial_seed666.pt](cnn_mel_control_wavlm_trial_seed666.pt) 使用 `predict_wavlm --checkpoint ...`；历史原Mel文件仍用 `predict_mel`。不同schema的权重不要混用。

工程切换未改变任何权重或阈值。原始审计结果与实验计划保留历史记录；其中“main_unchanged”描述实验当时状态，不是当前默认状态。

## 历史四组 Mel 权重（保留）

本目录提供四组新权重，已核对源best.pt、complete.json、frozen_eval_plan.json、独立审计和交付文件的SHA一致。四组均来自本轮重新训练。

| 文件 | 组别 | dev选中epoch | 完整logit阈值 |
|---|---|---:|---:|
| [cnn_clean_ps_eventf1_seed666.pt](cnn_clean_ps_eventf1_seed666.pt) | `cnn_clean` | 19 | `-0.07645973563194274` |
| [cnn_aug_ps_eventf1_seed666.pt](cnn_aug_ps_eventf1_seed666.pt) | `cnn_aug` | 16 | `0.11703586578369142` |
| [cnn_dual_ps_eventf1_seed666.pt](cnn_dual_ps_eventf1_seed666.pt) | `dual` | 17 | `0.029058158397674564` |
| [cnn_boundary_ps_eventf1_seed666.pt](cnn_boundary_ps_eventf1_seed666.pt) | `boundary` | 19 | `0.42304700613021856` |

完整SHA256、字节数、每组dev指标、参数量、结构配置、训练源码及数据产物哈希见[manifest.json](manifest.json)。运行源码为`df6ccf9`，主线`a50cc19`合入同一实现；Git的CRLF/LF转换改变了部分源码文件字节SHA，但规范化换行后内容完全一致，权重仍按原始字节SHA核对。只交付每组按开发集规则选中的一个权重，不发布所有逐轮权重。

四组从相同随机CNN-BiGRU开始，seed666，20轮。`cnn_clean`与`cnn_aug`是控制，`dual`与`boundary`是候选；新增模块用LayerNorm，不依赖旧模型或外部特征均值/标准差。选择规则为每轮dev真实时长FPR≤5%校准，再最大化dev事件F1@IoU0.5，平局取更早轮；eval完全使用冻结结果。

**阈值空间是原生20ms、fake为正类的logit**。执行`float64(score) >= float64(threshold)`以保留同分边界；阈值不是sigmoid概率，不能直接当概率阈值，也不能换成0、0.5或展示舍入值。每组必须使用自己的阈值。

checkpoint的schema是`ps_four_random_cnn_v1`，包含`backbone_state_dict`、可选`adapter_state_dict`、`model_config`、`adapter_options`、`protocol`、所选epoch及dev阈值/指标，不含优化器续训状态。应按`cnn_ps.train_four.evaluate_arms`的加载逻辑重建CNN和adapter，然后strict加载各自state。`cnn_clean/cnn_aug`的adapter为None；`dual/boundary`不能丢掉adapter只加载CNN。

历史Mel方案提供独立单音频入口，在`code/`下运行：

```powershell
python -m cnn_ps.predict_mel --audio 'example.wav' --output 'prediction_main.json' --device cpu
```

它默认加载本目录的历史Mel方案文件，完整加载CNN和adapter，读取其开发集阈值；无需标签或原始run元信息。支持`--checkpoint`指定同格式的boundary权重，输入为16kHz单声道。该入口不重新校准，不计算无真值的“准确率”。方法说明见[Mel方法论](../docs/mel_methodology_zh.md)。

**这四个文件不能直接用于旧`predict`或`load_backbone`。** `train_four --evaluate`也不是任意checkpoint单文件接口：它校验结果目录中的`best.pt`、`complete.json`、`protocol.json`、`dev_metrics.json`、`dev_scores20.npz`及冻结计划，还需要同一PS数据和单原声eval缓存。只有这四个`.pt`和manifest不足以走完整复评CLI；需要配套结果元信息，或按根README重新运行完整实验。复评使用开发集已冻结的阈值。

本轮完整数字与bootstrap见[结果摘要](../docs/results_four_eventf1_20261115.json)，复现训练/冻结/测试顺序见[根README](../README.md)。目前只有seed666，eval此前已使用，不是新盲测或多seed结论。
