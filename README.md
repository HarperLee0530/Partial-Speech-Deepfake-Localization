# Partial-Speech-Deepfake-Localization
Semantic-assisted reinspection and gain-based interval correction for partial speech deepfake localization.

## SAL 精简实验：组员使用指南

本项目用于课程实验：评估 SAL、训练 CNN–BiGRU 新基线，并探索语义信息对伪造语音检测的辅助作用。日常对照实验统一使用 **固定 10% 数据集**，保留完整数据的 SAL 入口和完整测试集评估方式。

## 组员先看：获取精简数据并运行

组员只使用固定 10% 数据开发、训练和测试；完整数据实验由项目负责人在其电脑执行。**`sal/full_data.py` 仍使用完整数据，不会自动退回精简数据。**

GitHub 提供代码和抽样 ID，不包含实际音频。由项目负责人通过组内网盘或移动硬盘提供下面的文件夹；目前尚未配置共享下载链接。不需要为此生成最终提交 ZIP。

| 工作 | 需要领取的资源 |
| --- | --- |
| CNN 固定 10% 实验 | 整个 `data/fixed10/`，约 2.73 GiB（2.93 GB），包含音频、标签和抽样清单；无需 SAL 权重。 |
| 固定 10% SAL 公开模型评估 | 上述数据及 `data/models/SAL_WavLM.ckpt`、`wavlm_large.pt`。 |
| 固定 10% SAL 重新训练 | 上述数据及 `data/models/wavlm_large.pt`；训练结束后用本次权重评估。 |
| 语义转写和错误统计 | 上述数据、对应 fixed10 预测目录中的 `predictions.sqlite`、`metrics.json`、`status.json`，以及本地 ASR/NER 模型或已有词级转录。分析已保存预测不需要加载 SAL 权重。 |

组员将收到的 `fixed10` 文件夹放到克隆仓库的 `data` 内，不要放成 `data/fixed10/fixed10`。音频复制到另一台电脑后可以是普通文件，不要求继续保持与完整数据的硬链接关系。

```text
Partial-Speech-Deepfake-Localization/
├── code/
├── data/
│   ├── fixed10/
│   │   ├── manifests/                       四份 JSON 和 TXT 清单
│   │   ├── PartialSpoof/database/
│   │   │   ├── train/                       2,538 条音频
│   │   │   ├── dev/                         2,484 条音频
│   │   │   ├── eval/                        7,124 条音频
│   │   │   └── segment_labels/              对应标签
│   │   └── LlamaPartialSpoof/
│   │       ├── cf/                          7,623 条音频
│   │       └── segment_labels/              对应标签
│   └── models/                             按任务准备权重
├── result/                                 运行生成；语义分析还需接收预测目录
└── visualization/
```

安装 Python 3.10 后，在克隆仓库的 `code` 目录打开 PowerShell：

```powershell
.\Setup.cmd
.\.venv\Scripts\python.exe -B tools/check_environment.py --task cnn --data-profile fixed10
.\.venv\Scripts\python.exe -B cnn/baseline.py --mode check
.\.venv\Scripts\python.exe -B cnn/baseline.py --mode train
.\.venv\Scripts\python.exe -B cnn/baseline.py --mode ps
.\.venv\Scripts\python.exe -B cnn/baseline.py --mode lps
```

Setup 仅检查安装环境；后续命令核验精简音频、标签和仓库清单，不要求完整数据、原始压缩包或解压标记。默认训练设备是 NVIDIA CUDA；无可用 GPU 时可在对应模型配置中选择 `cpu`，运行会更慢。不要复制他人的 `.venv`。

组员无需运行 `sal/full_data.py`、`--evaluation-data-profile full`、`data_tools.py --action prepare/audit/audit-full` 或整套原始数据下载脚本。这些完整数据准备、来源审计和完整测试由项目负责人执行。

没有 `result/cnn_grid_best.json` 时，CNN 使用默认参数，也可以自行搜索。复用最优参数指针需要它引用的搜索记录和权重，不能只传一个 JSON。组员实验后交回修改代码、配置、训练记录和权重，负责人再单独运行完整测试。

**语义模块目前能做转写、标注、对齐、错误统计、冻结统计和特征导出；尚未实现语义收益网络的训练。** 接手语义工作的同学可以先按第 5 节跑通数据流程，再按第 6 节开发后续模型，不能直接运行一个尚不存在的 `train` 命令。

## 1. 先看目录

```text
SAL_精简实验/
├── README.md                    本指南：文件说明、运行方法、组员交接
├── .gitignore                   排除大数据、权重、实验输出和本机环境
├── code/                        模型、数据工具、配置、测试
│   ├── sal/                     SAL 完整数据与固定 10% 两个入口
│   ├── cnn/                     CNN–BiGRU 基线及网格搜索
│   ├── semantic/                语义分析与特征准备
│   ├── tools/                   数据准备、检查和成绩汇总
│   ├── tests/                   功能测试
│   ├── vendor/SAL/              原作者模型及训练框架
│   ├── manifests/               资源来源、版本、校验值、固定抽样 ID
│   └── references/              资源平台的原始元数据
├── data/                        音频、标签、模型权重、语义中间数据
├── result/                      训练记录、指标、预测数据库、语义统计
└── visualization/               全部独立绘图代码、图表和文档导出
```

`data/`、`result/`、`visualization/outputs/` 和各个 `.venv/` 已被 Git 忽略，因此组员从 GitHub 下载后不会自动拥有它们。需要在本地创建目录，并另行获取实验资源。忽略规则不会删除本机文件。

### 主要代码文件

以下路径相对于项目根目录；各文件夹里的 `__init__.py` 只是 Python 包标记，不需要单独运行。

| 文件 | 功能 |
| --- | --- |
| [code/sal/full_data.py](code/sal/full_data.py) | 完整数据集上的 SAL 训练、验证、测试入口；默认只检查配置。 |
| [code/sal/fixed10.py](code/sal/fixed10.py) | 固定 10% 数据集上的 SAL 训练、验证、测试入口；与 CNN 的精简实验对照。 |
| [code/sal/pipeline.py](code/sal/pipeline.py) | 两个 SAL 入口共用的训练与推理实现。 |
| [code/sal/config.json](code/sal/config.json) | SAL 的学习率、批大小、训练轮数、设备等参数。 |
| [code/cnn/baseline.py](code/cnn/baseline.py) | 完整 CNN–BiGRU 基线：音频特征、网络、训练、验证、测试、单音频预测。 |
| [code/cnn/grid_search.py](code/cnn/grid_search.py) | 只根据 train/dev 搜索超参数，按 dev EER 选择最佳组，不用测试集选参。 |
| [code/cnn/config.json](code/cnn/config.json) | CNN 默认参数、数据档位和自动读取网格最优参数的开关。 |
| [code/cnn/grid_config.json](code/cnn/grid_config.json) | 学习率和 dropout 的候选值，目前共 4 组。 |
| [code/semantic/analysis.py](code/semantic/analysis.py) | 语义流程的统一入口，具体子命令见第 5 节。 |
| [code/semantic/config.json](code/semantic/config.json) | 语义类别优先级、英文词表、词区间覆盖阈值、置信度及统计平滑参数。 |
| [code/semantic/requirements.txt](code/semantic/requirements.txt) | 语义环境依赖；不包含 ASR/NER 模型文件。 |
| [code/semantic/README.txt](code/semantic/README.txt) | 语义输入格式、统计分母、时间对齐、数据隔离的详细说明。 |
| [code/tools/data_tools.py](code/tools/data_tools.py) | full/fixed10 路径管理、固定抽样、数据审计。 |
| [code/tools/reporting.py](code/tools/reporting.py) | SAL/CNN 成绩对照、指标复核、从全量预测提取固定子集成绩。 |
| [code/tools/download_resources.py](code/tools/download_resources.py) | 下载公开研究资源，支持恢复下载及校验；可能下载大量文件。 |
| [code/tools/prepare_data.py](code/tools/prepare_data.py) | 解压评估资源，包括 PS dev/eval 和 LPS；不负责全部训练数据的准备。 |
| [code/tools/check_environment.py](code/tools/check_environment.py) | 检查依赖、CUDA、数据和权重。 |
| [code/project_config.py](code/project_config.py) | 统一解析项目目录及公开 SAL 权重校验。 |
| [code/config.json](code/config.json) | data/result/解释器位置，路径以 code 为基准。 |
| [code/Setup.cmd](code/Setup.cmd) | Windows 下建立 Python 3.10 环境并安装锁定依赖。 |
| [code/requirements-lock.txt](code/requirements-lock.txt) | 完整环境锁定版本，供 Setup.cmd 使用。 |
| [code/requirements-evaluation.txt](code/requirements-evaluation.txt) | 主要评估依赖版本。 |
| [code/README.txt](code/README.txt) | 更详细的代码布局与运行说明。 |

### 测试、资料与可视化文件

| 文件或目录 | 功能 |
| --- | --- |
| `code/tests/test_baseline.py` | 网络、损失、时间对齐、训练与评估的合成数据检查。 |
| `code/tests/test_data.py` | 固定抽样、标签类型和结果可比性检查。 |
| `code/tests/test_grid.py` | 网格选优、失败保护、自动加载最优参数及 SAL 数据配置检查。 |
| `code/tests/test_semantic.py` | 语义标注、统计、时间对齐、特征回退和数据隔离检查。 |
| `code/vendor/SAL/` | 本项目依赖的第三方实现；原始说明和 LICENSE 保留在此处。 |
| `code/manifests/fixed10/` | train/dev/eval/lps 的 JSON 和 TXT 抽样清单，确保组员使用相同录音。 |
| `code/manifests/resource_sources.json` | 数据和模型资源来源及校验信息。 |
| `code/manifests/source_version.json` | 上游代码版本记录；其他 manifest 为历史校验和迁移记录。 |
| `code/references/` | `hf.json`、`wavlm_hf.json`、`zenodo.json`、`llama_zenodo.json` 等来源元数据。 |
| `visualization/diagnose_saved_predictions.py` | 从已保存预测生成诊断图和统计导出。 |
| `visualization/export_word_figures.py` | 导出适合放入 Word 的图表。 |
| `visualization/build_review_docx.py` | 根据已有图表生成审阅文档。 |
| `visualization/visualization_paths.py` | 管理绘图输入、输出和旧路径映射。 |
| `visualization/vendor_sal/gradcam.py`、`part_error.py` | 从上游移入的独立可视化脚本。 |
| `visualization/figures/` | 既有静态论文图。 |
| `visualization/outputs/` | 绘图及文档导出目录，不随 Git 上传。 |
| `visualization/requirements.txt`、`README.txt` | 可视化依赖和使用说明。 |

## 2. 数据与实验口径

固定抽样种子为 `20261107`。从官方划分中各自固定抽取约 10%，后续每次实验使用同一清单，训练时仅打乱顺序，**不会每轮重新随机抽 10%**。

| 划分 | 完整录音数 | 固定 10% 录音数 | 用途 |
| --- | ---: | ---: | --- |
| PS train | 25,380 | 2,538 | 训练 |
| PS dev | 24,844 | 2,484 | 验证、选模、调参 |
| PS eval | 71,237 | 7,124 | PS 测试 |
| LPS | 76,228 | 7,623 | 跨数据集测试 |

- 完整数据位于 `data/PartialSpoof/`、`data/LlamaPartialSpoof/`；精简数据位于 `data/fixed10/`。
- `code/manifests/fixed10/` 是可共享的 ID 清单，**清单本身不含音频**。运行时还需要 `data/fixed10/` 内的音频、标签和清单。
- 本机精简音频通过 NTFS 硬链接建立。不要原地改写音频，否则会同时影响对应完整数据；换电脑后需要重新准备本地数据。
- SAL 和 CNN 对照须使用相同划分、相同录音清单与指标定义。当前主表 F1 的正类为真实帧（bonafide）；PS 评价步长为 160 ms，LPS 为 20 ms。
- 精简训练完成后，可以在完整 **测试集** 上评估泛化表现；训练集、验证集上的分数应分别报告，不能混成一个测试成绩。

## 3. 新电脑准备

以下示例使用 Windows PowerShell。从下载后的项目根目录打开终端。项目不要求放在 D 盘。

```powershell
# 创建 GitHub 不包含的本地目录
New-Item -ItemType Directory -Force data, result, visualization/outputs | Out-Null
cd code
.\Setup.cmd
```

需要先安装带 Windows Python Launcher 的 Python 3.10。Setup 会安装锁定的 PyTorch/CUDA 依赖并仅检查环境；随后用 `tools/check_environment.py --task cnn --data-profile fixed10` 核验精简数据。`--task sal` 额外检查公开 SAL 评估所需的两个权重。

由组内另行提供所需的音频、标签、模型权重及实验预测结果，按原目录结构放置。公开资源来源见 `code/manifests/resource_sources.json`。从公开来源重建时，先查看下载/解压脚本再执行；训练数据另由 CNN 的 `--mode prepare` 入口准备。`data_tools.py --action prepare` 需要完整源数据，会建立或核验固定子集，不是只凭 ID 清单下载 10% 的工具。

在 `code` 目录下检查：

```powershell
.\.venv\Scripts\python.exe -B tools/check_environment.py
.\.venv\Scripts\python.exe -B sal/fixed10.py --mode check
.\.venv\Scripts\python.exe -B cnn/baseline.py --mode check
.\.venv\Scripts\python.exe -B semantic/analysis.py check
```

本项目模型配置默认使用 `cuda`，运行训练前需确认设备可用。不要复制他人的 `.venv` 作为本机环境。代码脚本参数中的相对资源路径通常以 `code/` 为基准；可视化脚本的相对路径以项目根目录为基准。

## 4. SAL 与 CNN 怎么运行

本节命令均从 `code` 目录执行。

### SAL

```powershell
# 公开权重在固定 10% 测试集上的推理评估
.\.venv\Scripts\python.exe -B sal/fixed10.py --mode test --checkpoint public --split eval
.\.venv\Scripts\python.exe -B sal/fixed10.py --mode test --checkpoint public --split lps

# 在固定 10% train/dev 上重新训练、选模，然后测试
.\.venv\Scripts\python.exe -B sal/fixed10.py --mode train
.\.venv\Scripts\python.exe -B sal/fixed10.py --mode validate --checkpoint latest
.\.venv\Scripts\python.exe -B sal/fixed10.py --mode test --checkpoint latest --split eval
.\.venv\Scripts\python.exe -B sal/fixed10.py --mode test --checkpoint latest --split lps
```

将入口换成 `sal/full_data.py` 即使用完整数据。`public` 表示公开 SAL 权重，`latest` 表示相同训练数据档位最近一次已完成训练的最佳权重；没有对应训练时不会回退到公开权重。也可传入具体权重路径。

若要用精简训练的 SAL 权重评估完整测试集，应在 `full_data.py` 中显式传入该精简训练权重路径，不能用 `latest` 代替，因为后者会寻找完整数据训练的权重。

公开权重推理与重新训练应分别报告。当前完成的是公开权重评估；固定 10% 的已有 SAL 指标由全量已保存预测按固定 ID 重新计算，**不是 SAL 在 10% 数据上重训的成绩**。新增 SAL 训练入口尚未完成真实重训实验；默认批大小为 8，和上游的 32 不同。

### CNN–BiGRU 与网格搜索

```powershell
# 先查看 4 组搜索计划，再正式搜索
.\.venv\Scripts\python.exe -B cnn/grid_search.py --mode plan
.\.venv\Scripts\python.exe -B cnn/grid_search.py --mode search

# 自动采用有效的网格最优参数，从头训练
.\.venv\Scripts\python.exe -B cnn/baseline.py --mode train
.\.venv\Scripts\python.exe -B cnn/baseline.py --mode ps
.\.venv\Scripts\python.exe -B cnn/baseline.py --mode lps

# 用精简训练的模型评估完整 PS 测试集
.\.venv\Scripts\python.exe -B cnn/baseline.py --mode ps --evaluation-data-profile full
```

搜索全部成功后发布 `result/cnn_grid_best.json`。默认 `use_grid_search_best=true`，基线会读取并校验最优参数、数据清单和权重。有有效结果就采用最优学习率/dropout；没有搜索结果则使用配置默认值；结果与当前配置不兼容时报错。训练命令会重新从头训练，不会自动续训。

`baseline.py --mode full` 表示“训练→PS 测试→LPS 测试→成绩对照”的完整流程，**不表示使用完整数据集**。

## 5. 负责语义的组员：先跑通这条流程

```text
完整录音 → ASR 词级时间戳 → 规则/NER 标注 → 对齐检测预测和真实标签
         → 错误统计 → 合法训练划分上冻结统计 → 新录音的词级统计特征
```

这里的“完整录音”指每条音频不按真假标签裁剪，不要求一次处理完整数据集。可以先用 32 条录音验证流程；这 32 条按预测保存顺序选取，不能据此推断全库分布。

### 5.1 环境、模型与输入

在 `code` 目录下建立独立语义环境：

```powershell
py -3.10 -m venv semantic/.venv
.\semantic\.venv\Scripts\python.exe -m pip install -r semantic/requirements.txt
.\semantic\.venv\Scripts\python.exe semantic/analysis.py check
```

另行准备模型文件，转写脚本不会自动下载：

- `data/models/whisper/`：faster-whisper 可加载的 **CTranslate2 格式** Whisper 模型目录。
- `data/models/ner_en/`：可由 spaCy 加载的英文模型目录，需要包含 `ner` 组件。
- 一个已完成的预测目录，包含 `predictions.sqlite`、`metrics.json`，并保留对应 `status.json`。只有汇总分数 JSON 不够。
- 对应录音音频。`check` 会列出本机已有预测目录；GitHub 克隆不包含这些实验结果。

`manifest` 根据预测目录中 `metrics.json` 的 `data_profile` 定位音频：`fixed10` 使用精简目录，`full` 使用完整目录；旧结果缺少字段时按完整数据处理。组员须领取对应 fixed10 预测结果，不要改字段把全量结果冒充为精简结果。也可手动准备音频清单 JSONL，每行格式如下，并把 audio 换成实际存在、相对于 code 的路径：

```json
{"uid":"实际录音ID","audio":"../data/fixed10/实际子目录/实际录音ID.flac","seconds":3.2}
```

不要照抄示例 ID、路径或时长；它们必须对应真实录音和预测记录。

### 5.2 小规模转写、标注和统计

下面 `$run` 是变量示例，必须替换为 `check` 列出的实际已完成预测目录。测试集结果在这里仅用于事后诊断。

```powershell
$run = '../result/替换为实际预测目录'
$semPython = '.\semantic\.venv\Scripts\python.exe'

# 按预测的数据档位自动查找音频；组员选择 fixed10 预测目录
& $semPython semantic/analysis.py manifest --run $run --output ../data/semantic/smoke_audio.jsonl --limit 32

# 默认 CPU int8；使用 GPU 需另加 --device cuda，并具备相应运行环境
& $semPython semantic/analysis.py transcribe --manifest ../data/semantic/smoke_audio.jsonl --asr-model ../data/models/whisper --output ../data/semantic/smoke_words.jsonl

& $semPython semantic/analysis.py annotate --transcripts ../data/semantic/smoke_words.jsonl --ner-model ../data/models/ner_en --output ../data/semantic/smoke_annotated.jsonl

& $semPython semantic/analysis.py analyze --run $run --transcripts ../data/semantic/smoke_annotated.jsonl
```

如已有人工审核的词级转录，可以跳过 ASR。转录每行一段录音，词时间是相对于原始音频的秒数，UID 必须和预测数据库一致：

```json
{"uid":"实际录音ID","language":"en","words":[{"text":"not","start":1.2,"end":1.4,"confidence":0.95}]}
```

不传 `--ner-model` 时仍可运行规则标注，但不能据此宣称实现了可靠的人物、地点和组织识别。当前词表针对英文。ASR 时间戳和 NER 类别也不等同于人工真值。

输出文件通常拒绝覆盖；重跑请使用新文件名。完成小规模检查后，生成正式清单时去掉 `--limit 32`，并记录 ASR/NER 模型版本和配置。

### 5.3 统计输出在哪里、看什么

`analyze` 自动创建 `result/semantic_diagnostics_时间戳/`：

| 文件 | 内容 |
| --- | --- |
| `word_diagnostics.csv` | 逐词时间、类别、真实/预测覆盖比例及错误状态，便于回查错例。 |
| `word_statistics.csv` | 各词的出现次数、漏检率、误报率等统计。 |
| `type_statistics.csv` | 各语义类型的错误统计。 |
| `region_type_distribution.csv` | 真实/伪造/漏检/误报等区间中的语义类型分布。 |
| `recording_coverage.csv` | 每条录音的词覆盖、无词错误时长等覆盖情况。 |
| `summary.json` | 来源、参数、覆盖情况、代码哈希和运行状态。 |
| `analyzed_uids.json` | 实际分析的录音 ID，供后续隔离检查。 |

优先检查转录缺失、低置信词、真假边界词以及未被词覆盖的错误时长，再解释统计。词级漏检率不等于事件召回率；主类别和多标签类别也不能混用。具体分母和过滤条件见 [语义详细说明](code/semantic/README.txt)。

新增语义绘图脚本放 `visualization/`，生成的图放 `visualization/outputs/`，统计表保留在 `result/`。

## 6. 语义训练：已具备什么，还需要做什么

当前 `analysis.py` 支持 `check / manifest / transcribe / annotate / analyze / freeze / features`，**没有神经网络训练子命令**。其中 `freeze` 拟合并保存统计先验，`features` 查询这些先验，不代表已经训练完成复查收益网络或区间修改网络。

### 冻结训练统计前的条件

需要按真实源录音组划分并记录 `base_train`、`correction_train`、`recheck_train`、`dev`、`test`，同源组不能跨角色；当前固定 10% 清单只保留官方划分，尚未完成这项阶段划分和源录音关联审计。

先用真实 `base_train` 训练基础检测器，再为 `correction_train` 导出预测、对齐转录并统计。当前入口没有一键完成这些阶段的数据划分和训练预测导出，组员需补齐相关数据加载/预测接口；不要把现成 eval 结果改成 `train` 就用于冻结。

`freeze` 要求报告来源的 `split=train`，全部统计 UID 属于 `correction_train`，分组清单必须包含实际 `base_train`。程序无法仅从权重判断模型是否见过这些录音，因此清单必须反映真实训练过程。

分组清单每行格式为：

```json
{"uid":"实际录音ID","source_group":"真实源录音组ID","role":"correction_train"}
```

满足上述条件后，以下命令才可使用；其中名称是占位符，需要替换：

```powershell
& $semPython semantic/analysis.py freeze --report ../result/实际修正训练统计目录 --partition-manifest ../data/semantic/partitions.jsonl --output ../data/semantic/frozen_statistics.json
& $semPython semantic/analysis.py features --transcripts ../data/semantic/新录音标注.jsonl --frozen ../data/semantic/frozen_statistics.json --output ../result/新录音语义特征.jsonl
```

特征查询会拒绝与拟合统计重复的 UID；下游必须读取 `prior_available` 等可用性信息。小样本或不可用时会回退，缺失统计的零占位值不能当成真实错误率。

### 后续开发建议顺序

1. 明确训练目标：复查哪些区间、如何定义修改收益，以及对应训练标签，先写清楚再实现网络。
2. 完成源录音分组和阶段数据划分，补齐训练分区预测导出；确保统计和后续模型的训练来源可追溯。
3. 跑通语义统计冻结与新录音特征导出，保存特征定义、模型版本和清单哈希。
4. 在 `code/semantic/` 实现下游训练入口。建议将一个模型的数据读取、网络、训练、验证与测试集中到一个完整文件；该文件目前尚未创建。
5. 只在训练部分拟合、dev 上选模型和阈值，方案确定后在固定测试集评估。对照纯声学基线与加入语义后的结果，图表统一导出至 visualization。

测试集词汇统计只用于事后分析，不能用于学习先验或调参。公开 SAL 权重的训练暴露也需核查，不能直接视为满足阶段隔离要求的基础检测器。不要直接把词汇错误率乘到 SAL 分数上并当成已验证的方法。

## 7. 组内交接与检查

GitHub 共享代码、配置和固定清单；音频、权重、预测数据库以及大规模转录结果另行共享。交接时记录所用数据档位、ID 清单、模型来源、权重、参数和输出目录，避免同名结果被误用。

负责语义的组员交回：转录及标注 JSONL、统计目录、所用模型与配置说明；开展训练后还应交回真实阶段划分清单、冻结统计、训练配置、权重和独立测试成绩。模型放 `data/models/`，运行记录放 `result/`，图表及绘图代码放 `visualization/`。

从 `code` 目录运行功能检查：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_*.py'
```

功能测试使用小规模或合成输入，不替代真实模型训练，也不代表 ASR/NER 在真实数据上的准确性已经验证。

上游 SAL 的原始文档和许可见 [vendor/SAL/README.md](code/vendor/SAL/README.md) 与 [LICENSE](code/vendor/SAL/LICENSE)。当前整理不生成提交 ZIP；最终提交包另行制作。
