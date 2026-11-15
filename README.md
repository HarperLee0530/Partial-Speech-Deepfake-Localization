# PS 语音真假帧检测：CNN-BiGRU

当前主方案：**保留已训练的 CNN-BiGRU，用信道增强训练一个小型残差式逐帧判别头，提高 PS 上的真假帧识别表现。**

仓库只维护 CNN 路线：原 CNN 从头训练、残差头主方案、直接微调对照、推理和统一评估。旧 SAL、WavLM、语义、候选收益网络、生成器分组、事件长度权重和 LPS 入口已移出当前版本；历史提交仍可查阅。本地数据和历史实验不属于代码清理范围。

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

以下为**固定 3,562 条 PS eval 子集**、160 ms、假帧为正类；不是 71,237 条完整测试集成绩。新增头报告 seed666/667/668 的均值±样本标准差。

| 方法 | Frame F1 ↑ | EER ↓ | 真实帧误报率 ↓ |
|---|---:|---:|---:|
| 最初原 CNN，原阈值0 | 78.8702% | 19.7829% | 5.6269% |
| 冻结 CNN+头，原声训练 `clean_erm` | 79.2353±0.4258% | 19.4341±0.1361% | 5.3004% |
| **冻结 CNN+头+信道增强 `aug_erm`** | **81.1063±0.2762%** | **16.3033±0.1274%** | **5.9508%** |

主方案较原 CNN 的帧 F1 提高 **2.2361 个百分点**、EER 降低 **3.4796 个百分点**，同时真实帧误报率增加约0.324个百分点。没有证据支持“实现短事件保护”：≤320 ms事件召回14.3575%，原声头对照14.4572%；20 ms事件指标也未全面改善。残差连接和信道增强本身是已有方法，不能单独宣称为新算法。

[结果明细](docs/results_ps160.json)保存各seed、阈值、checkpoint哈希。[独立回放验证](docs/verification_evaluation.json)已用清理后代码重算全部3562条录音，8个模型/阈值组合的160 ms结果与原报告完全一致。

**直接微调对照尚待正式运行**：不加新头，增强音频后更新全部 CNN 参数。先跑seed666，与已有同seed结果配对；未完成时不列入结果表。学习率、batch和可训练参数范围不同，这是两种训练方案的比较。若要把增益单独归因于增强，还需要“原声直接微调”对照。

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

音频、标签、权重和大缓存不提交Git。使用组内已有文件，或按[PS公开资源清单](code/manifests/ps_sources.json)获取官方数据。不需要WavLM/SAL权重。

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

将上述原权重放到 `../data/models/original_cnn.pt` 并核对哈希；路径可替换。

```powershell
python -m cnn_ps.prepare --manifest ../data/manifests_ps/train.json --labels20 ../data/labels20/train.npz --labels20-encoding fake --checkpoint ../data/models/original_cnn.pt --output ../cache/train --channels --device cuda
python -m cnn_ps.prepare --manifest ../data/manifests_ps/dev.json --labels20 ../data/labels20/dev.npz --labels20-encoding fake --checkpoint ../data/models/original_cnn.pt --output ../cache/dev --device cuda
python -m cnn_ps.train_head --train-cache ../cache/train --dev-cache ../cache/dev --channel-cache ../cache/train --output ../result/head_run1 --arms aug_erm clean_erm --seeds 666 667 668

# 冻结训练结果后再准备并评估eval。
python -m cnn_ps.prepare --manifest ../data/manifests_ps/eval.json --labels20 ../data/labels20/eval.npz --labels20-encoding fake --checkpoint ../data/models/original_cnn.pt --output ../cache/eval --device cuda
python -m cnn_ps.evaluate --dev-cache ../cache/dev --eval-cache ../cache/eval --heads ../result/head_run1/aug_erm_seed666/b_model.pt ../result/head_run1/aug_erm_seed667/b_model.pt ../result/head_run1/aug_erm_seed668/b_model.pt --output ../result/head_eval1 --native20
```

缓存保留float16上下文及所有8N存储尾帧，真实时长由损失权重处理。跨设备/批大小/库版本重提取可能有微小差异；精确回放历史分数使用组内归档缓存，见[缓存验证](docs/verification_cache.json)。

统一160 ms主表EER为sklearn ROC最近交点处平均FPR/FNR。事件IoU≥0.5，先最大化一对一匹配数再最大化IoU；160 ms预测区间对真实20 ms事件，短事件≤320 ms。20 ms时长指标和160 ms帧计数指标不能混用。原CNN从头训练沿用历史genuine-positive EER选模，与主表EER分开记录。

```powershell
python -m cnn_ps.predict --audio ../data/example.wav --checkpoint ../data/models/original_cnn.pt --output ../result/example_original.json
python -m cnn_ps.predict --help
```

使用新增头时必须指定对应时间尺度的dev阈值。输出是logit，不是概率；音频要求16kHz单声道。

## 直接微调CNN的对照

加载同一原CNN并开放全部参数，不加新头。整段训练，不裁剪；同样三种信道及20/160 ms双尺度损失。默认seed666、8轮、batch8条录音、AdamW学习率0.0001，按上述20 ms dev规则选模，epoch0可选。

```powershell
python -m cnn_ps.finetune --checkpoint ../data/models/original_cnn.pt --expected-sha256 36899e66bcf84b1f5aa8a6f2d57128eb80b077bc28eaa94d9bb8fbb713a2a428 --train-manifest ../data/manifests_ps/train.json --dev-manifest ../data/manifests_ps/dev.json --train-labels20 ../data/labels20/train.npz --dev-labels20 ../data/labels20/dev.npz --output ../result/cnn_finetune1 --device cuda
python -m cnn_ps.evaluate_cnn --run ../result/cnn_finetune1 --dev-manifest ../data/manifests_ps/dev.json --dev-labels20 ../data/labels20/dev.npz --eval-manifest ../data/manifests_ps/eval.json --eval-labels20 ../data/labels20/eval.npz --output ../result/cnn_finetune_eval1 --device cuda
```

未完成时不能把dev成绩当eval结果。实现启动前已做[独立微调核对](docs/verification_finetune_prelaunch.json)。

## 从头训练原CNN

原设置：seed666、dropout0.4、AdamW0.001、batch8、4秒且160 ms对齐随机裁剪、最多30轮、patience5；原训练只有160 ms BCE。历史最佳第7轮，第12轮停止。

```powershell
python -m cnn_ps.manifests full --audio-dir ../data/full/train/audio --labels-file ../data/full/train/segment_labels/train_seglab_0.16.npy --split train --output ../data/full_train.json
python -m cnn_ps.manifests full --audio-dir ../data/full/dev/audio --labels-file ../data/full/dev/segment_labels/dev_seglab_0.16.npy --split dev --output ../data/full_dev.json
python -m cnn_ps.baseline --train-manifest ../data/full_train.json --dev-manifest ../data/full_dev.json --output ../result/cnn_from_scratch1 --device cuda
```

按本机布局替换完整音频目录，并核对条数。历史训练没有完整源码快照，因此不保证重新训练逐位复现权重；旧权重加载及同环境推理已经[验证一致](docs/verification_backbone.json)。
