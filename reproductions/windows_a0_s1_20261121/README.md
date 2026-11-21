# Windows 上的 A0/S1 配对复现

基准提交为 `50d994ba3d1637f413cc88e2fb785087526e96f2`。S1 移除了主 160 ms BCE 损失。四组均用固定 UID，从头训练 20 轮，每组更新 15,880 次。根据 DEV 上真实语音时长 FPR ≤ 5% 的约束选择 checkpoint，将其原始 20 ms logit 阈值冻结后评估 EVAL。

| Seed | A0 DEV | S1 DEV | S1−A0（百分点） | A0 EVAL | S1 EVAL | S1−A0（百分点） |
|---|---:|---:|---:|---:|---:|---:|
| 666 | 83.4082% | 84.1120% | +0.7039 | 73.6767% | 73.4291% | −0.2475 |
| 667 | 83.9180% | 85.0650% | +1.1470 | 73.4878% | 73.4765% | −0.0113 |

表中指标为 Event-F1@IoU0.5。两个种子的 DEV 均有改善，但这次从头训练未复现原报告的 EVAL 增益。复评原作者发布的四个 checkpoint 时，四组 EVAL 成绩均与原报告一致。Windows 本地 TRAIN、DEV 特征文件的 SHA256 与 AutoDL 记录不同，差异对训练成绩的影响尚未确定。

## 查看工件

- [从头训练结果](fresh_replay/报告.md) 和 [逐模型指标](fresh_replay/results.json) 包含最佳轮、冻结阈值、Precision、Recall 和时长 FPR。各模型子目录提供逐录音结果及 DEV/EVAL 原始分数。
- [原权重在本机的复评](frozen_replay/报告.md) 和 [逐帧核对结果](frozen_replay/results.json) 可用于检查原作者的 checkpoint。
- [训练工件](experiments/) 保存每组的 `best.pt`、初始权重、各轮训练记录、训练协议、最佳轮 DEV 分数和完成记录。[四组权重的哈希与阈值](fresh_frozen.json) 可按模型名查阅。
- [日志](logs/) 对应最终完成的四次运行。A0 seed 667 的首次运行在第 10 轮后写进度文件时遇到 Windows 文件锁，随后按相同种子从头训练。最终成绩来自完整的 20 轮。Windows 包装脚本为 JSON 写入增加了有限次数重试。
- [数据和环境核验](data/manifests/audit.json) 记录 UID、标签和特征哈希，以及本机环境。基准包内有[固定数据与复现说明](../s1_20261121/REPRODUCE.md)。原始音频、WavLM 权重和特征缓存需另行准备。

## 在独立工作目录运行

按本次驱动的路径约定，在项目下创建 `handoff_audit/<新运行目录>`，将本目录的 `reproduce_local.py` 和 `entry667/` 复制到其中。将基准提交检出到新运行目录的 `repo/`，并准备 `.venv/`（本机 Python 3.10.18、PyTorch 2.2.1+cu121）。把固定划分对应的 `layers/` 和 `reference/` 缓存置于新运行目录的 `cache/fixed_ps/` 下，均需含 train、dev 和 eval。在项目的 `partial_speech_project/data/` 下准备驱动引用的 `reduced/`、`downloads/database_segment_labels_v1.2.tar.gz` 和 `models/wavlm-base-plus/pytorch_model.bin`。

从新运行目录执行以下 PowerShell 命令，驱动会核验输入、复评原权重、训练四组并评估。请使用空的输出目录，驱动会跳过已有完成记录的运行。

```powershell
.\.venv\Scripts\python.exe -X utf8 -u reproduce_local.py
```

若只希望用本次发布的四个最佳权重复评，可在新运行目录额外复制本包的 `experiments/`，并执行以下命令。输入数据沿用上面的目录约定，输出将写入新运行目录的 `fresh_replay/`。

```powershell
.\.venv\Scripts\python.exe -X utf8 -c "import reproduce_local as r; r.audit(); r.replay('fresh_replay')"
```
