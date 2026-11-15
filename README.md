# PS 语音真假帧检测：CNN-BiGRU

当前主方案：**保留已训练的 CNN-BiGRU，用信道增强训练一个小型残差式逐帧判别头，提高 PS 上的真假帧识别表现。**

仓库只维护 CNN 路线：原 CNN 从头训练、残差头主方案、直接微调对照、推理和统一评估。旧 SAL、WavLM、语义、候选收益网络、生成器分组、事件长度权重和 LPS 入口已移出当前版本；历史提交仍可查阅。本地数据和历史实验不属于代码清理范围。

**评价标准**见 [PS 统一评估规范 v2](docs/evaluation_protocol_v2.txt)：原生20 ms事件F1为主，同时报告短事件召回、真实时长误报率和全真录音误报率。现已完成三个固定权重的统一口径重评，见下表；训练代码仍使用原有选模规则，尚未切换为v2事件F1选模。

## 三个固定权重的统一20 ms重评

固定PS开发集2484条、测试子集3562条。三个模型分别在同一开发集按真实时长FPR≤5%校准阈值并冻结，再评估测试集；使用实际20 ms标签、IoU≥0.5一对一事件匹配，短事件≤320 ms，不增加平滑或最短事件过滤。两个适配模型均为seed666。

| 指标 | 最初原CNN | 信道增强+小修正头 | 信道增强+直接微调CNN |
|---|---:|---:|---:|
| **事件F1 ↑（主指标）** | 31.7508% | 43.5979% | **50.2401%** |
| 事件Precision ↑ | 25.5042% | 36.8446% | **42.5432%** |
| 事件Recall ↑ | 42.0495% | 53.3823% | **61.3372%** |
| ≤320 ms事件召回 ↑ | 44.5932% | 54.4320% | **62.9841%** |
| 真实时长误报率 ↓ | **4.5105%** | 4.8612% | 6.0467% |
| 全真录音误报率 ↓ | 37.8082% | 25.2055% | **18.9041%** |
| 时长F1 ↑（补充） | 61.0931% | 70.3340% | **74.7840%** |

共有11427个真实伪造事件，其中7694个≤320 ms；三个模型分别找回4805、6100、7009个事件，短事件分别找回3431、4188、4846个。365条全真录音中，分别有138、92、69条至少被误报一次。

直接微调相对原CNN的事件F1提高18.4894个百分点，相对小修正头提高6.6422个百分点；同时真实时长误报率比小修正头增加1.1855个百分点。全真录音误报更少与误报时长更多可以同时发生，二者衡量的问题不同。不能据此称所有指标全面改善。

这是**固定历史权重的重新算分**：回放冻结CNN缓存和原生分数，重新计算开发集阈值与全部指标，未训练或重选checkpoint。两个适配权重仍按旧J选模；原CNN仍是历史固定权重。只有一个适配seed，且原CNN原先只有160 ms监督，不能将增益单独归因于信道增强。测试子集此前已用于开发观察，不属于新的盲测集。

[完整结果、开发集阈值与计数](docs/results_native20_fixed_reevaluation.json)；[重评步骤及来源](docs/native20_fixed_reevaluation.txt)；[独立逐分数复算](docs/verification_native20_fixed_reevaluation.json)。19项回归/边界测试通过，独立核对三个模型的开发集阈值及全部逐录音计数一致。旧160 ms结果在下文单独保留。

## 模型与训练

```text
16 kHz 单声道 → Log-Mel80（20 ms）→ CNN（16/32/64）→ 两层 BiGRU → 上下文 h（256维）
h → 原线性判别层 → 原分数 z
h → 按 train 统计量归一化并截断 → Linear(256,32) → Tanh → Linear(32,1) → 修正量 Δ
最终逐帧分数 = z + Δ → 8 个 logit 均值 → 160 ms 分数
```

- 原 CNN-BiGRU 共 **911,521** 个参数，主方案中全部冻结，包括原判别层；新增头 **8,257** 个参数，最后一层零初始化，初始输出与原 CNN 相同。8 帧始终均匀平均，不学习池化权重。
- 头的上下文输入按原始 train 缓存的均值/标准差归一化，再截断至 `[-10,10]`。保留历史初始化随机数顺序，兼容已保存实验。
- 信道增强发生在**音频输入端**：原声、20 dB 白噪声、轻微 EQ。噪声按整段 RMS 缩放；EQ 为 `x[n] + 0.15*(x[n]-x[n-1])`，首个 previous 等于首采样。不裁剪幅度，不改时长和真假标签。
- 每条训练录音每轮出现一次，三种信道固定分配并逐轮轮换；dev/eval 只用原声。
- 损失：**160 ms 时长加权 BCE + 真实 20 ms 时长加权 BCE**，系数各为 1。20 ms 标签来自官方细粒度标注，不能用重复的粗标签代替。没有分组、类平衡或短事件权重。
- 头默认 CPU、8 轮、batch 2,048 个粗帧块、AdamW 学习率 0.001。原声 dev 选模：真实时长 FPR≤5%，`J=0.5×时长F1+0.5×事件F1`；epoch0 可选，平局取更早轮。随后仅用 dev 选择 160 ms 帧 F1 阈值，不重选 checkpoint。

## 已完成结果与限制

以下为**固定 3,562 条 PS eval 子集**、160 ms、假帧为正类；不是 71,237 条完整测试集成绩。新增头的F1/EER报告seed666/667/668的均值±样本标准差，误报率列为均值。

| 方法 | Frame F1 ↑ | EER ↓ | 真实帧误报率 ↓ |
|---|---:|---:|---:|
| 最初原 CNN，原阈值0 | 78.8702% | 19.7829% | 5.6269% |
| 冻结 CNN+头，原声训练 `clean_erm` | 79.2353±0.4258% | 19.4341±0.1361% | 5.3004% |
| **冻结 CNN+头+信道增强 `aug_erm`** | **81.1063±0.2762%** | **16.3033±0.1274%** | **5.9508%** |

主方案较原 CNN 的帧 F1 提高 **2.2361 个百分点**、EER 降低 **3.4796 个百分点**，同时真实帧误报率增加约0.324个百分点。没有证据支持“实现短事件保护”：≤320 ms事件召回14.3575%，原声头对照14.4572%；20 ms事件指标也未全面改善。残差连接和信道增强本身是已有方法，不能单独宣称为新算法。

[结果明细](docs/results_ps160.json)保存各seed、阈值、checkpoint哈希。[独立回放验证](docs/verification_evaluation.json)已用清理后代码重算全部3562条录音，8个模型/阈值组合的160 ms结果与原报告完全一致。

**直接微调对照已完成一个种子**：不加新头，增强音频后更新原CNN全部参数，同时沿用真实20/160 ms双尺度监督。seed666训练8轮，dev选中第8轮；起始模型核验和训练共19.68分钟。下面单独列出同seed比较，避免与上面的三seed均值混用。

| 方法，固定3562条PS eval | 160 ms Frame F1 ↑ | EER ↓ | 真实帧误报率 ↓ |
|---|---:|---:|---:|
| 最初原CNN，阈值0 | 78.8702% | 19.7829% | 5.6269% |
| 冻结CNN+头+增强，seed666 | 80.9346% | 16.2184% | 5.7480% |
| 全CNN直接微调+增强，seed666 | **82.8403%** | **16.0585%** | 6.8100% |

相对同seed头方案，直接微调的Frame F1提高1.9057个百分点，EER降低0.1599个百分点，但误报率增加1.0620个百分点。160 ms事件F1从29.9108%到31.6143%，短事件召回从14.2449%到14.5308%。

20 ms结果另行比较：

| 同seed方法 | 时长F1 ↑ | 事件F1 ↑ | ≤320 ms事件召回 ↑ | 真实时长误报率 ↓ |
|---|---:|---:|---:|---:|
| 冻结CNN+头+增强 | 70.3340% | 43.5979% | 54.4320% | 4.8612% |
| 全CNN直接微调+增强 | **74.7840%** | **50.2401%** | **62.9841%** | 6.0467% |

这次直接微调改善了召回和定位，但有误报代价，不能称全面胜出。新方案只有一个seed；学习率、batch和可训练参数范围不同，这是训练方案比较。原CNN原先只有160 ms监督，两种适配方案都新增真实20 ms监督；若要把增益单独归因于信道增强，还需要“原声全CNN微调”对照。

[完整新结果及逐轮dev记录](docs/results_cnn_finetune_seed666.json)、[训练前固定的对比协议](docs/protocol_cnn_finetune_seed666.json)、[独立逐帧分数复算](docs/verification_finetune_result.json)。原结构911,521参数不变，无新增头；26个参数张量均有更新。所有模型测试UID完全一致，模型与阈值未根据eval调整。

权重获取和模型用途见 [weights/README.md](weights/README.md)；所有文件SHA256见 [weights/manifest.json](weights/manifest.json)。

## 代码与环境

| 文件（位于 `code/cnn_ps/`） | 职责 |
|---|---|
| `backbone.py`、`baseline.py` | 原 CNN-BiGRU、特征、原基线从头训练 |
| `io.py`、`manifests.py` | 可迁移路径、固定PS清单、官方细粒度标签提取 |
| `augmentation.py`、`prepare.py` | 波形增强、冻结CNN缓存 |
| `head.py`、`cached_data.py`、`train_head.py` | 残差头、主方案及原声头对照 |
| `evaluate.py`、`predict.py` | 基线/头统一评估、单音频推理 |
| `finetune.py`、`evaluate_cnn.py` | 全CNN直接微调及固定阈值评估 |
| `metrics160.py`、`metrics20.py` | 帧、时长、区间匹配指标 |

Python3.10。依赖仅 PyTorch、torchaudio、NumPy、SoundFile、SciPy、scikit-learn。PowerShell进入仓库根目录后：

```powershell
cd code
.\Setup.cmd
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Setup不下载数据或开始训练。GPU需要匹配的PyTorch/torchaudio CUDA安装；CPU也可准备缓存和推理。下文在 `code/` 中执行，`python` 表示已激活的环境；Setup不会激活当前PowerShell环境，未激活时将下文的 `python` 换为 `.\.venv\Scripts\python.exe`。输出命令拒绝覆盖非空实验目录。

## 数据与原权重

**CNN权重现在随仓库提供，位于 [weights/](weights/README.md)**：原CNN、直接微调CNN、三种子增强头和原声头对照。组员训练新方案应从 `weights/original_cnn.pt` 开始。音频、标签和大缓存仍不提交Git，可使用组内已有文件，或按[PS公开资源清单](code/manifests/ps_sources.json)获取官方数据。无需WavLM/SAL权重，也无需寻找旧 `result/` 目录。

历史原权重 `cnn_bigru_train_20261109_203335_588602_best.pt` 的 SHA256 为：
`36899e66bcf84b1f5aa8a6f2d57128eb80b077bc28eaa94d9bb8fbb713a2a428`。

该权重在**完整PS train25380/dev24844**上训练选模。新增头使用固定train6345（train_detector5076+train_edit1269）、dev2484、eval3562；`train_review` 不使用。旧角色名称只用于定位已有音频。各官方划分ID互斥，但不宣称排除了全部来源衍生关系。不能把重新在6345条上训练的CNN当作同一个历史基线。

现有音频布局为 `data/reduced/<role>/audio/<uid>.wav`，粗标签位于该role的 `segment_labels/<train|dev|eval>_seglab_0.16.npy`。生成路径清单并提取真实细标签：

```powershell
python -m cnn_ps.manifests reduced --reduced-root ../data/reduced --output ../data/manifests_ps
python -m cnn_ps.manifests labels20 --archive ../data/database_segment_labels_v1.2.tar.gz --output ../data/labels20
```

标签工具先校验官方归档SHA256，输出固定清单的NPZ（1=假）。直接使用官方原始NPY时1=真，准备缓存必须显式指定编码。字典格式NPY和PyTorch checkpoint只读取可信来源文件。

## 主方案复现

原权重已包含在 `../weights/original_cnn.pt`，核对哈希即可使用。下列命令从 `code/` 目录运行。

```powershell
python -m cnn_ps.prepare --manifest ../data/manifests_ps/train.json --labels20 ../data/labels20/train.npz --labels20-encoding fake --checkpoint ../weights/original_cnn.pt --output ../cache/train --channels --device cuda
python -m cnn_ps.prepare --manifest ../data/manifests_ps/dev.json --labels20 ../data/labels20/dev.npz --labels20-encoding fake --checkpoint ../weights/original_cnn.pt --output ../cache/dev --device cuda
python -m cnn_ps.train_head --train-cache ../cache/train --dev-cache ../cache/dev --channel-cache ../cache/train --output ../result/head_run1 --arms aug_erm clean_erm --seeds 666 667 668

# 冻结训练结果后再准备并评估eval。
python -m cnn_ps.prepare --manifest ../data/manifests_ps/eval.json --labels20 ../data/labels20/eval.npz --labels20-encoding fake --checkpoint ../weights/original_cnn.pt --output ../cache/eval --device cuda
python -m cnn_ps.evaluate --dev-cache ../cache/dev --eval-cache ../cache/eval --heads ../result/head_run1/aug_erm_seed666/b_model.pt ../result/head_run1/aug_erm_seed667/b_model.pt ../result/head_run1/aug_erm_seed668/b_model.pt --output ../result/head_eval1 --native20
```

缓存保留float16上下文及所有8N存储尾帧，真实时长由损失权重处理。跨设备/批大小/库版本重提取可能有微小差异；精确回放历史分数使用组内归档缓存，见[缓存验证](docs/verification_cache.json)。

已有160 ms结果表的EER为sklearn ROC最近交点处平均FPR/FNR。事件IoU≥0.5，先最大化一对一匹配数再最大化IoU；160 ms预测区间对真实20 ms事件，短事件≤320 ms。20 ms时长指标和160 ms帧计数指标不能混用。原CNN从头训练沿用历史genuine-positive EER选模，与结果表EER分开记录。

```powershell
python -m cnn_ps.predict --audio ../data/example.wav --checkpoint ../weights/original_cnn.pt --output ../result/example_original.json
python -m cnn_ps.predict --help
```

使用新增头时必须指定对应时间尺度的dev阈值。输出是logit，不是概率；音频要求16kHz单声道。

## 直接微调CNN的对照

加载同一原CNN并开放全部参数，不加新头。整段训练，不裁剪；同样三种信道及20/160 ms双尺度损失。默认seed666、8轮、batch8条录音、AdamW学习率0.0001，按上述20 ms dev规则选模，epoch0可选。

```powershell
python -m cnn_ps.finetune --checkpoint ../weights/original_cnn.pt --expected-sha256 36899e66bcf84b1f5aa8a6f2d57128eb80b077bc28eaa94d9bb8fbb713a2a428 --train-manifest ../data/manifests_ps/train.json --dev-manifest ../data/manifests_ps/dev.json --train-labels20 ../data/labels20/train.npz --dev-labels20 ../data/labels20/dev.npz --output ../result/cnn_finetune1 --device cuda
python -m cnn_ps.evaluate_cnn --run ../result/cnn_finetune1 --dev-manifest ../data/manifests_ps/dev.json --dev-labels20 ../data/labels20/dev.npz --eval-manifest ../data/manifests_ps/eval.json --eval-labels20 ../data/labels20/eval.npz --output ../result/cnn_finetune_eval1 --device cuda
```

上述命令可复现新的直接微调实验。已完成seed666结果见上表；重新运行请使用新的输出目录。实现启动前已做[独立微调核对](docs/verification_finetune_prelaunch.json)。

## 从头训练原CNN

原设置：seed666、dropout0.4、AdamW0.001、batch8、4秒且160 ms对齐随机裁剪、最多30轮、patience5；原训练只有160 ms BCE。历史最佳第7轮，第12轮停止。

```powershell
python -m cnn_ps.manifests full --audio-dir ../data/full/train/audio --labels-file ../data/full/train/segment_labels/train_seglab_0.16.npy --split train --output ../data/full_train.json
python -m cnn_ps.manifests full --audio-dir ../data/full/dev/audio --labels-file ../data/full/dev/segment_labels/dev_seglab_0.16.npy --split dev --output ../data/full_dev.json
python -m cnn_ps.baseline --train-manifest ../data/full_train.json --dev-manifest ../data/full_dev.json --output ../result/cnn_from_scratch1 --device cuda
```

按本机布局替换完整音频目录，并核对条数。历史训练没有完整源码快照，因此不保证重新训练逐位复现权重；旧权重加载及同环境推理已经[验证一致](docs/verification_backbone.json)。
