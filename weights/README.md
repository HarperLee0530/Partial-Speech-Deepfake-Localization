# 本轮权重状态

**当前没有发布权重，`manifest.json`中的`weights`为空。** 旧CNN、B1和双分辨率权重已按本轮清理要求移除，不再作为训练起点。

当前四组为`cnn_clean`、`cnn_aug`、`dual`、`boundary`。所有组以seed666创建相同随机CNN-BiGRU，端到端训练20轮；不加载预训练checkpoint。新增模块采用可训练LayerNorm，不依赖旧CNN特征统计。

`code/configs/cnn_ps_structure.json`只含网络和前端参数，不是权重。新LogMel缓存使用`--model-config`生成。

本轮产物保存位置为：

```text
D:\SAL_精简实验\result\ps_four_eventf1_20261115_run1\<arm>\
  epochs\epoch_00.pt ... epoch_20.pt
  epochs\epoch_XX_metrics.json
  epochs\epoch_XX_scores20.npz
  best.pt
  dev_metrics.json
  dev_scores20.npz
  complete.json
```

每轮`.pt`包含`backbone_state_dict`、可选`adapter_state_dict`、结构/adapter配置、dev指标与阈值、协议。`best.pt`仅按dev真实时长FPR≤5%条件下的事件F1选择，epoch0可选，平局取更早轮。存档不含优化器续训状态。

四组完成后执行`train_four --freeze-only`，再生成eval缓存并`--evaluate`。本轮checkpoint应由`train_four`加载，不能套用旧`load_backbone`或旧成品推理命令。

正式结果尚未产生，验证完成后再登记新权重及SHA256。不要沿用旧模型阈值。完整命令见[根目录说明](../README.md)。
