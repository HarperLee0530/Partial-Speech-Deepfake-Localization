项目代码说明（最终模块布局，2026-11-07）

项目根目录固定为四个大文件夹：code、data、result、visualization。
code 放代码和配置；data 放音频、标签与模型权重；result 放训练记录、指标、日志和预测数据库；
visualization 放全部独立绘图脚本、图表和可视化导出文件。没有启动正式训练或生成提交 ZIP。

一、五个主要入口
1. sal/full_data.py：完整数据集的 SAL 训练、验证、测试入口。
2. sal/fixed10.py：固定 10% 数据集的 SAL 训练、验证、测试入口。
3. cnn/baseline.py：完整的 CNN–BiGRU 基线实现，包括网络、数据读取、训练、验证、测试与单音频预测。
4. cnn/grid_search.py：只使用固定 10% train/dev 搜索超参数，按验证集 EER 选择最佳组。
5. semantic/analysis.py：完整语义分析，包括 ASR、标注、时间对齐、统计、冻结统计和特征导出。

二、sal 文件夹
full_data.py：锁定 full 数据档位。默认 check，只打印配置和数据路径。
fixed10.py：锁定 fixed10 数据档位。默认 check，只打印配置和数据路径。
pipeline.py：两个入口共用的官方 SAL 训练与推理评估实现，不重复维护两份模型。
config.json：两种数据档位共用的 SAL 训练参数。
__init__.py：Python 包标记。

SAL 两个入口支持 --mode check/train/validate/test/all。
train：调用原作者 SALTrainer、模型、损失和数据增强，在对应 PS train 上训练、PS dev 上选模。
all：训练完成后，用本次最佳验证权重测试 --split 指定的测试集，默认 PS eval。
validate/test 必须明确 --checkpoint public、latest 或自定义权重路径：
  public：公开 SAL_WavLM 权重；属于公开模型推理复现。
  latest：同一训练数据档位最近一次已完成训练的最佳权重，不会回退到公开权重。
  路径：相对 code，例如 ../data/models/sal_train_.../best-....ckpt。
--split eval/lps 指定测试集；训练始终使用 PS train。
SAL 训练用预训练 WavLM 骨干和新初始化的 SAL 检测头，不加载公开 SAL 权重继续训练。
默认沿用官方模型、SPL/CSM、RawBoost、Adam、StepLR；batch size 从原配置 32 改为 8，workers 改为 0。
训练配置和实际数据清单保存到 result，权重保存到 data/models；这不等同于完全相同训练条件复现。
目前完成的是公开权重评估；新增训练入口已做配置和功能检查，尚未完成真实 SAL 重训实验。

三、cnn 文件夹
baseline.py：一个完整文件，按“网络与特征→数据与batch→训练和评估”排列。
  网络：16 kHz音频→80维Log-Mel→CNN→双向GRU→逐帧伪造logit。
  原生步长20ms；PS按8帧平均得到160ms输出，LPS保留20ms。
  默认 --mode train，从头训练；dev/ps/lps 分别评估验证集、PS、LPS。
  --mode full：训练→PS→LPS→成绩对照，full指完整流程，默认数据仍是固定10%。
  --mode check：检查配置；prepare：准备PS训练资源；predict：预测代码顶部AUDIO_FILE。
  最佳权重按dev EER选择；区间阈值也只用dev选择，不在测试集调参。
grid_search.py：读取网格范围，依次训练各组，记录每组验证指标并选出最优参数。
  默认 --mode plan 只显示搜索方案；--mode search 才启动正式网格搜索。
  不绑定旧实验、不自动运行测试集、不使用测试成绩选择参数。
config.json：CNN默认参数；training_data_profile/evaluation_data_profile默认为fixed10。
grid_config.json：学习率、dropout候选值；当前2×2共4组。
__init__.py：Python包标记。

baseline.py 与 grid_search.py 的关系：
  当前按最新要求先在fixed10训练/验证集上网格搜索；use_grid_search_best=true。
  网格搜索完整结束后，baseline.py自动读取 ../result/cnn_grid_best.json 中的最优参数。
  该文件仅在整个搜索成功完成后发布，记录验证EER最低组及其权重。
  baseline读取时复核搜索状态、最优组、训练/验证清单、固定配置和权重SHA256。
  有有效结果：自动采用最优learning_rate和dropout。
  没有结果：使用cnn/config.json默认参数。
  搜索失败：不发布半成品，不覆盖上一份有效最优结果。
  搜索与当前数据或其他固定配置不一致：报错，避免静默套用不相容参数。
  --mode train用最优参数重新从头训练；--mode ps/lps使用匹配参数的已完成训练权重，
  不把网格中最后一组当成最优组，也不会自动续训。
  如需完全手动使用config.json，将use_grid_search_best改为false。

四、semantic 文件夹
analysis.py：合并原common、annotation、alignment、statistics、adapters、pipeline。
  默认check；子命令manifest/transcribe/annotate/analyze/freeze/features覆盖整条语义流程。
  语音转写给出词及时间戳；规则/NER标记语义类别；与预测和标签对齐；统计漏检、误报；
  合法训练划分上拟合并冻结统计；为新录音导出词级语义统计特征。
config.json：词重叠阈值、置信度门槛、最小样本量、类别优先级和英文词表。
requirements.txt：可选faster-whisper/spaCy等依赖；ASR/NER模型仍须单独准备。
README.txt：详细输入格式、统计定义、运行命令与数据隔离要求。
__init__.py：包标记。
当前还没有复查收益网络、区间修改网络或实际声学复检；语义代码不等于完整论文方法。
测试集语义统计只能作事后分析，不能直接用于拟合训练统计或调参。

五、tools 文件夹
 data_tools.py：full/fixed10路径、固定抽样和数据审计。
  --action prepare：创建/核验固定子集，不更换已确定的抽样；同步可分发清单。
  --action audit（默认）：核验精简数据、标签、SAL/CNN一致性和已有SAL子集指标。
  --action audit-full：检查完整评估数据音频/标签覆盖。
 reporting.py：模型成绩汇总、对照、预测数据库独立复核，以及复用全量预测计算固定子集成绩。
  --action compare（默认）/summary/verify/subset；--data-profile fixed10/full。
  默认对照公开SAL结果；重训SAL可通过--sal-ps和--sal-lps指定两个结果JSON。
 download_resources.py：按来源清单下载数据和公开权重，支持恢复下载与校验。
 prepare_data.py：解压下载的数据并保存完成标记。
 check_environment.py：检查Python、依赖、CUDA、数据与可选权重哈希。
 __init__.py：包标记。
所有工具把相对路径解释为相对code，不依赖终端当前目录。

六、tests 文件夹
 test_baseline.py：网络、损失、时间对齐、指标与训练/评估流程的合成数据测试。
 test_data.py：固定抽样、标签类型和结果可比性测试。
 test_grid.py：网格选优、失败保护、基线自动读取最优参数、SAL两种数据配置测试。
 test_semantic.py：词语标注、对齐、统计回退、可移植路径和数据泄漏防护测试。
 __init__.py：测试包标记。
这些测试不启动正式数据集训练，不会把合成测试成绩当成研究成绩。

七、code根目录和资料目录
project_config.py：统一读取相对路径，提供data/result位置，并验证公开SAL权重。
config.json：整个项目的数据目录、结果目录和Python位置。
Setup.cmd：用Python3.10创建环境，安装锁定依赖，调用环境检查。
requirements-evaluation.txt：主要评估依赖版本。
requirements-lock.txt：完整锁定环境，供Setup.cmd使用。
README.txt：当前说明。
vendor/SAL：原作者模型、训练器、加载器、指标和配置。运行依赖，保留核心实现。
  原作者两个独立绘图脚本已移至 ../visualization/vendor_sal，Grad-CAM已适配新导入路径。
  原训练器内部的可视化选项在本项目配置中关闭；不把训练器拆到绘图目录。
manifests：官方来源、代码版本、校验值和fixed10数据清单。
references：资源平台原始元数据，方便追溯。
.venv：本机环境，不是研究代码，不应直接上传或打包分发。

八、固定数据说明
抽样种子20261107，CNN训练随机种子666；按录音固定抽样，每轮只打乱顺序。
train 2538条；dev 2484条；PS eval 7124条；LPS 7623条。
完整音频保留。精简音频是NTFS硬链接，不重复占用音频空间，不可原地改写。
保留官方train/dev/eval划分；源录音衍生关系未独立核验。
后续修正器/选择器的阶段训练数据还需单独划分，不能声称已经完成三阶段隔离。
主表F1正类为真实帧（bonafide），与SAL官方实现保持一致。

九、常用命令（code目录下PowerShell）
./.venv/Scripts/python.exe -B sal/full_data.py --mode check
./.venv/Scripts/python.exe -B sal/fixed10.py --mode check
./.venv/Scripts/python.exe -B sal/fixed10.py --mode test --checkpoint public
./.venv/Scripts/python.exe -B sal/fixed10.py --mode train
./.venv/Scripts/python.exe -B sal/fixed10.py --mode validate --checkpoint latest
./.venv/Scripts/python.exe -B sal/fixed10.py --mode test --checkpoint latest --split lps
./.venv/Scripts/python.exe -B cnn/grid_search.py --mode plan
./.venv/Scripts/python.exe -B cnn/grid_search.py --mode search
./.venv/Scripts/python.exe -B cnn/baseline.py --mode check
./.venv/Scripts/python.exe -B cnn/baseline.py --mode train
./.venv/Scripts/python.exe -B cnn/baseline.py --mode ps
./.venv/Scripts/python.exe -B cnn/baseline.py --mode lps
./.venv/Scripts/python.exe -B cnn/baseline.py --mode ps --evaluation-data-profile full
./.venv/Scripts/python.exe -B tools/reporting.py --action compare
./.venv/Scripts/python.exe -B tools/data_tools.py --action audit
./.venv/Scripts/python.exe -B semantic/analysis.py check
./.venv/Scripts/python.exe -B -m unittest discover -s tests -p 'test_*.py'

旧代码、旧说明和迁移记录备份在 ../result/code_cleanup_backup_20261107；不是运行入口。
已停止的旧训练和搜索没有重启，未上传GitHub，未生成提交ZIP。
