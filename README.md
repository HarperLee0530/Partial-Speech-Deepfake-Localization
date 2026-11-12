# Partial Speech Deepfake Localization 课程项目

本版本固定使用缩减数据，完成 SAL、CNN、语义分析入口与数据协议迁移。**尚未完成缩减版 SAL 的正式训练，也未证明后续方法改善 LPS 泛化。** 历史成绩只用于研究诊断。

## 目录

- `code/`：`sal.py`、`cnn.py`、`semantic.py`、`data.py`、`evaluate.py`；共享 `project_config.py`、一份 `config.json` 和一份 `test_project.py`。`vendor/SAL` 保留上游实现及许可证。
- `data/`：配套 ZIP 的说明和校验索引。实际音频与模型不提交 Git。
- `result/historical_evidence/`：近期实验的必要统计与独立复核；不是新缩减实验成绩。
- `result/migration/`：数据与代码验收记录。未来运行自动在 `result/` 创建新的 run。
- `visualization/`：保留的诊断图和可视化工具。旧图均有历史数据范围，不能当成缩减后重跑结果。

## 数据交接

从组员取得 `reduced_data_and_wavlm.zip`，**解压到项目根目录**，得到 `data/reduced/` 和 `data/models/wavlm_large.pt`，不要解压成 `data/data/`。包约 3.53 GB，包含 17,460 条音频和通用 WavLM 权重。校验值见 `data/package_manifest.json`。

| 角色 | 原划分比例 | 条数 | 用途 |
|---|---:|---:|---|
| train_detector | train 20% | 5,076 | SAL 与 CNN 检测器训练 |
| train_edit | train 5% | 1,269 | 冻结 SAL 后生成编辑收益样本 |
| train_review | train 5% | 1,269 | 预留给后续复查选择器 |
| dev | dev 10% | 2,484 | 选 checkpoint、阈值及预算 |
| eval | eval 5% | 3,562 | PS 最终评价 |
| lps | crossfade 范围 4.985% | 3,800 | 跨数据集评价 |

三训练角色录音 ID 互斥，各自保留全部 20 位训练说话人。官方 train/dev/eval 边界不变。PS 无可靠原音频衍生映射，因此不宣称来源完全独立；LPS 以文件名推断的来源组整组抽样，不能保证每位说话人严格同比例。dev 沿用已使用的固定清单；LPS 之前已被分析，不应称为全新未观察测试集。原生标签仍是 PS 160 ms、LPS 20 ms，1=真、0=假。

## 首次运行

建议 Windows、Python 3.10、NVIDIA CUDA 环境。在 PowerShell 中进入 `code`：

```powershell
.\Setup.cmd
.\.venv\Scripts\python.exe project_config.py --check
.\.venv\Scripts\python.exe data.py --action audit --deep
.\.venv\Scripts\python.exe test_project.py
```

`Setup.cmd` 安装锁定依赖；不下载完整数据、不自动开始训练。若使用 CPU，在 `config.json` 的 sal/cnn 段修改 device；WavLM 正式训练仍建议 GPU。不要把旧 full/fixed10 权重、风险表或结果指针冒充缩减协议产物。

## SAL 只训练一次 冻结后缓存

```powershell
.\.venv\Scripts\python.exe sal.py --mode check
.\.venv\Scripts\python.exe sal.py --mode train
.\.venv\Scripts\python.exe sal.py --mode validate --checkpoint latest
.\.venv\Scripts\python.exe sal.py --mode predict --split train_edit --checkpoint latest
.\.venv\Scripts\python.exe sal.py --mode predict --split train_review --checkpoint latest
```

模型由通用 WavLM 初始化，不从完整 PS 训练过的 SAL 成品权重续训。dev 选择 checkpoint 后，记录输出 JSON 中的权重路径与 SHA256；后续所有对照显式传同一权重路径，防止新增训练改变 `latest` 指向。每个预测 run 保存逐录音 `predictions.sqlite`、`metrics.json` 和状态记录。

方案及超参数冻结后再运行测试：

```powershell
.\.venv\Scripts\python.exe sal.py --mode test --split eval --checkpoint latest
.\.venv\Scripts\python.exe sal.py --mode test --split lps --checkpoint latest
```

正式对照建议把 `latest` 替换为已登记的具体 checkpoint 路径。SAL 保留 batch=8、min_epochs=40、max_epochs=60；缩减减少每轮工作量，不代表几分钟可以训完。修改候选策略不需要重新训练 SAL。

**实现修正记录**：原 vendor 的 CSM `_mixup_batch` 将同一录音拼接回自身，复核时输入与标签完全不变。本版本保留 vendor 原件，在 `sal.py` 训练实例上覆盖该方法，以不同录音为来源并对齐音频/标签切点；训练和预测元数据记录 `csm_implementation=aligned_cross_recording_v1`。模型结构和损失不变，但这是明确的实现修正，不能声称与未修复上游代码逐字节等价，也没有据此取得新的性能结论。CPU/GPU 推理和跨盘路径也已统一处理。

GPU 训练的 EER 状态在 CPU 上累积和汇总，避免 PyTorch 2.2.1 的 CUDA 累计求和与确定性设置冲突；训练本身仍为 FP32，指标定义保持上游实现。外部文件位于另一盘符时记录绝对路径，项目内 data/result 仍记录可迁移的逻辑相对路径。

## CNN 和语义

```powershell
.\.venv\Scripts\python.exe cnn.py --mode check
.\.venv\Scripts\python.exe cnn.py --mode train
.\.venv\Scripts\python.exe cnn.py --mode dev
.\.venv\Scripts\python.exe cnn.py --mode ps
.\.venv\Scripts\python.exe cnn.py --mode lps
.\.venv\Scripts\python.exe semantic.py check
```

CNN 网格搜索合并在 `cnn.py --mode grid-plan/grid-search`，默认不自动使用旧网格参数。课程项目可先做单个 CNN 对照，不必再次运行完整网格。

语义默认导出不使用真值的 ASR/POS/NER 特征，作为可选消融。`semantic.py --help` 查看 manifest/transcribe/annotate/analyze/features 的单一入口。ASR、spaCy 模型未包含在数据包，需按组员环境另备；可用已有合规转写输入先验证核心流程。旧真假比例/POS风险及同分数风险不能直接复用。风险表只能在 `train_edit` 拟合，不能再回灌该角色训练样本；应用到 train_review/dev/eval/lps 时校验冻结检测器及清单。

## 下一步主线

先阅读交接 Word 与历史证据，审阅并反驳方法假设，再完成缩减 SAL 基线及 LPS 错误诊断。重点辨别内部空洞、碎片、错误合并和全假录音上的声学漏检。之后检查候选的可修正上限，与简单规则对照，再决定是否实施编辑收益网络。

**六种编辑动作、编辑收益网络和复查收益选择器尚未实现。** 新方法目标是在 PS 保持性能的约束下改善 LPS，不能预先写成成功。语义三轮结果均不足以支持稳定增益，保留为可选研究问题；复查选择器等有效错误事件足够再做。

固定 dev 上预先约定 PS 退化容忍值与主要指标，所有方法同一 checkpoint、同一清单、同一候选预算。测试集不参与调参。只修改最终区间时，应报告区间/帧级 F1、IoU、漏检和误报时长，不能把未改变声学排序的后处理说成 EER 改善。
