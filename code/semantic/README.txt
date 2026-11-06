语义路线：词语对齐、类别统计与冻结统计特征

一、当前实现对应研究方案的哪一步
对应《语义辅助复查与区间收益修正完整研究方案》第五节“建立词汇与语义错误统计”，
以及第九节“转写与时间对齐”的统计特征出口。

音频 → 本地 ASR 词级时间戳 → 词汇规则 / NER / 词性标注
     → 与已保存的 SAL 预测及真实标签对齐 → 词表、类型表、区间分布表
     → 仅在独立修正训练部分拟合并冻结统计 → 新录音查询统计特征与可用性标记。

这是首阶段语义路线，不是完整论文系统。复查收益网络、修改收益网络、实际声学复检
尚未在本模块实现。词频、错误率都不是实际复查收益；本模块不修改 SAL 分数或预测。
当前已有的 PS eval / LPS crossfade 只能做事后诊断，不能直接用于训练词表或调参。

二、模块（已合并）
analysis.py：完整入口和实现，包含转录、语义标注、词时间对齐、错误统计、冻结和特征导出。
config.json：类型优先级、词重叠阈值、ASR置信度、最小样本量与词表。
requirements.txt：可选ASR/NER依赖。__init__.py：包标记。
功能测试放在 ../tests/test_semantic.py；旧的小模块已合并，不需要逐个运行。
所有相对路径以code为基准，模型和数据在../data，表格及统计记录在../result。
独立绘图代码与图表统一放在项目根目录的visualization。

三、词语类型
主类别：negation 否定、quantity 数量、time 时间、person 人物、place 地点、
organization 组织、other 未归类。主类别按 config 的优先级唯一分配，可画堆叠图。
多标签类型：保留交叉语义，例如人物名字同时像时间词；multi_type 不能直接堆叠。
词性：spaCy 的 POS，如 NOUN / VERB / ADJ / PROPN；无模型时明确记录 UNKNOWN。

内置英文词表只识别部分否定、数字、时间表达，是可审计的规则基线，不是完整语义理解。
仅靠词表不能可靠识别人名、地名、组织；需要 NER 模型或已人工审核的 types 标注。
“May”“March”等存在歧义，词表结果需要抽查；NER 结果也不是人工真实标签。
other 代表当前提取器未归类，不代表该词没有语义。不同提取器的结果应分开对照。
统计单位是 ASR 提供的词时间单元；多词实体按词计数，不把一个实体误称为一个词。
英文规则不直接套用中文；其他语言必须补充对应词表和分词 / 对齐方案。

四、输入格式
JSONL，每行一段录音，uid 必须与 predictions.sqlite 完全一致，例如：
{"uid":"录音ID","language":"en","words":[{"text":"not","start":1.2,"end":1.4,"confidence":0.95},{"text":"Alice","start":1.4,"end":1.8,"confidence":0.9,"types":["person"]}]}

start/end 是原始完整录音中的秒；不是裁剪片段内的相对时间。
词区间应按时间排序且不重叠；不能用整句平均切分伪造词级时间戳。
confidence 可为 null，缺失会记录，不假装置信度为 1。
空 words 列表代表 ASR 未转出词，该录音仍计入覆盖审计。
未提供转录的录音不会被伪造为空转录，会在 summary 中报告 missing_transcript_recordings。
ASR 没识别出的词无法进入词汇计数，因此还报告真假和错误区间中未被词覆盖的时长。

五、运行步骤（下列命令从 code 目录执行，或给脚本填写完整路径）
1. 检查：
  .venv/Scripts/python.exe semantic/analysis.py check

2. 生成真实音频清单。首轮用 32 条只验证流程，不能据此前 32 条推断全库分布：
  .venv/Scripts/python.exe semantic/analysis.py manifest --run ../result/sal_wavlm_eval_full_20261106_115611_680558 --output ../data/semantic/ps_eval_smoke_audio.jsonl --limit 32
  .venv/Scripts/python.exe semantic/analysis.py manifest --run ../result/sal_wavlm_lps_full_20261106_124753_722237 --output ../data/semantic/lps_smoke_audio.jsonl --limit 32
正式完整清单省略 --limit，并使用新文件名，不覆盖已生成文件。

3. 可选安装转录/NER依赖。建议单独环境，避免改变正在训练的基线环境：
  python -m venv semantic/.venv
  semantic/.venv/Scripts/python.exe -m pip install -r semantic/requirements.txt
将 CTranslate2 格式 Whisper 模型放到 ../data/models/whisper/，spaCy 英文模型放到
../data/models/ner_en/。模型需另行准备；代码不会自动下载。

4. 在完整录音上转录，不按真实伪造标签裁剪：
  semantic/.venv/Scripts/python.exe semantic/analysis.py transcribe --manifest ../data/semantic/ps_eval_smoke_audio.jsonl --asr-model ../data/models/whisper --output ../data/semantic/ps_eval_words.jsonl
默认 CPU int8；显式 --device cuda 才用 GPU。ASR 时间另记，词时间戳是估计而非人工对齐。

5. 语义标注：
  semantic/.venv/Scripts/python.exe semantic/analysis.py annotate --transcripts ../data/semantic/ps_eval_words.jsonl --ner-model ../data/models/ner_en --output ../data/semantic/ps_eval_annotated.jsonl
不提供 --ner-model 则只用规则及输入已有 types，不能宣称完成人物/地点识别。

6. 对齐、统计（已有 numpy 的原环境也能运行）：
  .venv/Scripts/python.exe semantic/analysis.py analyze --run ../result/sal_wavlm_eval_full_20261106_115611_680558 --transcripts ../data/semantic/ps_eval_annotated.jsonl
LPS 使用对应 run 和其转录文件另跑一遍；不同评价网格和数据集不混成一个分母。
每次自动建立 result/semantic_diagnostics_时间戳；也可显式 --output 指定不存在的新目录。
词覆盖阈值默认为 0.5，低置信度门槛为 0.5；这些是待验证参数，不是优化后的结果。

六、结果怎样看
region_type_distribution.csv：回答“某种区间里，哪类词最多”。
  true_spoof：真实伪造区间；true_bonafide：真实语音区间。
  false_negative：真伪造且 SAL 判真的漏报时间段。
  false_positive：真真实且 SAL 判假的误报时间段。
  true_positive / true_negative：对应正确判别的时间段。
  complete_miss_event：整个真实事件都没有预测重叠。
  no_overlap_false_alarm_event：整个预测事件都没有真实伪造重叠。
  touched_words：与该区间有正时长交集的不同词数；同一词在同一行只计一次。
  fractional_word_count：按“相交时长/词时长”加权的词数，可用于跨边界敏感性分析。
  *_share：同一 scope、同一 region 内各唯一主类别的构成比例。
  不同 region 会重叠，例如 complete_miss_event 是 false_negative 的子集，不能把它们相加。

type_statistics.csv：回答“某类词出现时，多容易漏检或误报”。
word_statistics.csv：对每个规范化词给出同样统计。
  total：所有词，包括边界不确定、低置信和未完整评分的词。
  主分析只纳入完整被网格评分、通过置信门槛、真实标签纯净的词。
  tp/fn/fp/tn 按词与初检区间的时间覆盖阈值判断；不等同于事件 tIoU 指标。
  spoof_rate = (tp+fn)/(tp+fn+fp+tn)。
  miss_rate = fn/(tp+fn)，分母是该类真实伪造词。
  false_alarm_rate = fp/(fp+tn)，分母是该类真实词。
  分母为零时输出空值，不伪造为 0%。各比例有明确分母与描述性 Wilson 区间。
  partial_truth：词跨真假边界；主比例单独排除，threshold_* 给出按固定阈值纳入的对照计数。
  partial_prediction：预测跨词边界，保留标记，按阈值作主判断，不据此删掉难例。
  primary_type_word_duration：按词持续时间分层；不等同于按真实事件长度分层。
  scope=partial/genuine/full_spoof 分开统计；all 是合并视图，不与分组再相加。

word_diagnostics.csv：逐词位置、语义标签、分数、真实/预测覆盖比例及错误状态，可回查错例。
recording_coverage.csv：每段录音的网格、词覆盖、无词的错误时长和实际使用的模型。
summary.json：参数、数据来源、转录覆盖、失败信息、代码哈希；不会修改原预测库。
analyzed_uids.json：实际纳入录音，用于后续检查统计与特征输入是否重复。

可以用主类别数量画柱状图，用 primary_type 的 tp/fn/fp/tn 画互斥构成图。
不要将多标签类型堆叠成总体比例，也不要把词级漏检率当成事件召回率。
正式显著性检验应按录音重采样；当前 Wilson 区间没有处理同录音词语相关性。
本次没有生成语义图或虚构真实分布；有真实转录结果后再用 visualization 中的绘图流程。

七、后续冻结训练统计和导出特征
freeze 拒绝 dev/eval/LPS 的诊断表。它要求 run 的 split=train，且提供真实训练划分清单：
{"uid":"某录音","source_group":"源录音组","role":"correction_train"}
role 取 base_train/correction_train/recheck_train/dev/test。清单必须含实际 base_train，
统计中的所有 uid 都属于 correction_train，同源组不能跨划分。
注意：清单必须反映真实的基础模型训练过程；程序无法从权重推断是否见过录音。
当前公开预训练 SAL 的训练暴露情况、以及原 CNN 全训练集训练方式都需要单独审计，
不能给已有测试诊断改个文件名就称为无泄漏的训练统计。

  .venv/Scripts/python.exe semantic/analysis.py freeze --report ../result/修正训练语义统计 --partition-manifest ../data/semantic/partitions.jsonl --output ../data/semantic/frozen_statistics.json
  .venv/Scripts/python.exe semantic/analysis.py features --transcripts ../data/semantic/新录音标注.jsonl --frozen ../data/semantic/frozen_statistics.json --output ../result/新录音语义特征.jsonl

词和类型估计以总体平均平滑；每个条件比例按其各自分母检查最小样本量。
查询回退：可靠的词统计 → 可靠的主类型统计 → acoustic_fallback，输出 prior_available 标记。
缺失统计的 0 是占位值，不代表真实错误率为 0；下游必须读取可用性标记。
features 不打开预测标签，且拒绝与拟合录音 UID 重复；只输出允许的词与统计特征。
这些特征未来接入复查收益网络。不能直接把统计先验乘到 SAL 伪造分数上。

八、检查
  .venv/Scripts/python.exe tests/test_semantic.py
测试包含真实/预测四种组合、跨边界、尾部无标签、零 logit 平局、整段漏检区别、
转录遗漏、类型重叠、NER 字符映射、统计分母、小样本回退、测试先验禁用及同源隔离。
可选 ASR/NER 模型尚未安装时，它们的真实推理不在单元测试覆盖范围内。

接口依据
https://github.com/SYSTRAN/faster-whisper （word_timestamps，本地模型）
https://spacy.io/usage/linguistic-features （命名实体字符范围与词性）
