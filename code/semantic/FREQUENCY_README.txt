语义类别频率统计（不训练 SAL 或收益网络）

入口：semantic/frequency.py
解释器：semantic/.venv/Scripts/python.exe
范围：项目共享 fixed10/train 清单中的全部 2,538 条录音。
不混入 dev/eval/LPS 或其他试跑，不读取真假标签或 SAL 分数。

模型：
1. 本地 CTranslate2 Whisper base，faster-whisper，CPU int8，beam_size=5，英文。
   生成词级时间戳；不是人工逐词对齐。
2. spaCy en_core_web_sm（3.8.0）输出实体类别和词性。
3. semantic/config.json 的规则补充否定、时间和数量。

运行（相对路径以 code 为基准）：
semantic/.venv/Scripts/python.exe semantic/frequency.py --output ../result/semantic_frequency_train_fixed10_20261111 --asr-model 本地Whisper模型目录 --workers 2

输出：
run_config.json：模型文件哈希、版本、子集清单哈希、标注配置。
transcripts.jsonl：逐录音转写、词时间戳和置信度。
annotated.jsonl：逐词实体/语义类别、词性、标注来源。
status.json：已完成数量和状态。只有 complete 才表示全部完成。
frequency_summary.json：完整类别计数、占比、常见词、词性和质量统计。
统计结果.txt：便于直接阅读的汇总。

可用同一条命令继续中断的任务，已保存录音不会重新转写。
同一输出目录要求模型、清单及配置完全一致；换设置应使用新目录。
程序每完成一条保存一次，每 25 条更新进度。

解释口径：
- 每个 ASR 词时间单元出现一次计一次；不是不同词数，也不是完整实体数。
- 主语义类别按配置优先级唯一分配，因此可相加。
- 占全部词比例与“仅在六类已归类词中占比”分别保存。
- 额外报告置信度 >= 0.5 的子集，不把低置信度隐藏掉。
- 词性是另一种分类，不能与人物/地点等语义类别混加。
- other 表示未归入六种目标类别，不代表没有语义。
- 这些是模型和规则的预测，英文月份/人名等歧义仍可能误标。
- 高频词类不等于高伪造率、高漏检率或高复查收益。
- 当前任务只提供固定 10% 训练集的描述性统计。
