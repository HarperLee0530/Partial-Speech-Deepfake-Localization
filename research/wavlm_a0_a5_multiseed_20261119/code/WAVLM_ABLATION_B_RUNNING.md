# B组代码与运行说明

当前交付为代码准备：未训练B组、未生成真实数据特征缓存、未读取B组测试数据，也没有B组性能结果。检查仅使用合成输入做前向计算，不反向传播或更新参数。

## 与原方案对应

| 编号 | 干预 | 仍保留的部分 |
|---|---|---|
| B0_full | 无干预，参照A0完整模型 | 冻结原始WavLM及完整A0任务后端 |
| B1_no_conv_position | 卷积位置嵌入输出置零 | 波形卷积特征编码器、Transformer、相对位置偏置及门控 |
| B2_position_gate_one | 相对位置偏置的输入依赖门控固定为1 | 原有相对位置偏置、卷积位置嵌入、注意力Q/K/V |
| B3_no_relative_bias | 注意力中的相对位置偏置置零 | 卷积位置嵌入、内容注意力Q/K/V；不再计算无作用的位置门控 |

B1关闭的是Transformer之前的**卷积位置嵌入**，并非提取原始波形声学特征的卷积编码器。用零输出而非Identity，避免把内容特征错误地加两次。

B2将门控严格固定为1，不是将门控参数清零（清零后的公式并不产生1）。B3不改变内容注意力，也不移除Transformer层。

三组均提取干预后第6/9/12层输出，重新学习三层融合和完整A0后端。沿用可学习下采样、局部注意力、可学习门控和所有损失；不换成A5，也不继承A0已经训练的后端参数。编码器权重保持冻结且eval状态。对现有预训练权重进行干预的结果，不能代表“移除此模块后从头预训练”的效果。

## 新增文件

- `cnn_ps/wavlm_ablation_b.py`：编码器实例开关、B组后端与缓存读取校验。
- `cnn_ps/cache_wavlm_ablation_b.py`：逐组逐split重新提取特征，eval访问需先冻结全部B组。
- `cnn_ps/train_wavlm_ablation_b.py`：独立训练及评估入口，沿用A0预算和选模规则。
- `run_wavlm_ablation_b.py`：顺序缓存/训练3组，全部冻结后统一评估，接着执行推理核对和报告。
- `benchmark_wavlm_ablation_b.py`：最佳权重的32条固定开发音频原始推理核对、可训练参数与耗时。
- `report_wavlm_ablation_b.py`：完整指标、相对B0的配对bootstrap及中文报告。
- `tests/test_wavlm_ablation_b.py`：8项仅前向及流程保护测试。
- `verify_wavlm_ablation_b.py`：本机公开预训练权重在合成波形上的仅前向检查。

原A组代码、结果和第三方transformers文件均不修改。实例方法替换只存在于对应Python进程中；需要先加载原始权重，再应用相应干预，不能只凭相同state_dict判断启用了哪个方案。

## 当前已检查的范围

- 未干预B0与原始WavLM输出逐元素相同。
- 三个开关保持所有参数字节不变，全部编码器参数冻结；输出有限且保持相同网格。
- B1被绕过的卷积位置参数、B2被绕过的门控参数、B3被绕过的偏置及门控参数，不再影响前向结果。
- B2/B3与显式传入原始偏置/零偏置的注意力计算一致。
- 各B组后端初始化、前向输出、损失系数与A0一致。
- 缓存拒绝跨组、旧版schema、不同源码/预训练权重和错误split。
- 无冻结计划时，在打开eval数据之前拒绝操作。
- 默认CLI不创建训练目录、不生成缓存、不调用训练。

尚未执行完整缓存、后端训练及测试评估，不能把这些代码检查当作性能验证。当前验证限定本机transformers 4.51.3、标准WavLMEncoder与指定WavLM-Base-Plus权重；其他实现需重新核验。

## 只检查代码，不训练

在项目`code`目录使用已有虚拟环境：

```powershell
$env:PYTHONIOENCODING='utf-8'
.venv/Scripts/python.exe -m unittest discover -s tests -p test_wavlm_ablation_b.py -v
.venv/Scripts/python.exe verify_wavlm_ablation_b.py
.venv/Scripts/python.exe run_wavlm_ablation_b.py
```

最后一条默认只打印计划。验证记录在工作区`handoff_audit/ablation_b_20261119_code_checks`，不属于训练结果。不要将旧A组测试套件整体运行作为“仅前向检查”，其中包含优化器更新测试。

## 将来明确开始训练时

以下命令只作使用说明，本次没有执行：

```powershell
.venv/Scripts/python.exe -u run_wavlm_ablation_b.py --execute --discard-caches
```

结果默认保存到工作区`handoff_audit/ablation_b_20261119`。建议后台执行时使用隐藏的独立进程并重定向标准输出/错误到日志，避免长训练依赖交互工具会话。本次不创建任何后台进程。

`--discard-caches`在每组训练/测试完成后只删除本次生成的B组特征缓存，先保留完整摘要；不会删除共享A组缓存、标签、原始音频或模型。不同B组特征不能复用。每组train/dev约需35GB量级；若保留三个组的完整缓存，需要超过100GB额外空间，按实际metadata计算为准。程序每次建缓存检查剩余空间，空间不足时停止，不自动删除其他文件。

调度顺序：B1缓存train/dev→B1训练→B2缓存/训练→B3缓存/训练→冻结三组模型与阈值→依次提取eval缓存及评价→对照A0→原始音频推理核对→报告。

固定条件：seed666、20轮、batch8、AdamW lr0.001、weight decay0.0001、梯度裁剪5、相同三视图增强与样本顺序；WavLM不更新。每组开发集真实时长FPR≤5%校准阈值，按开发事件F1选轮次（允许epoch0，平分选较早轮次）。不同组使用各自开发阈值，禁止测试集调阈值或选模型。

运行期间调度器阻止Windows自动睡眠，正常结束或异常退出后释放请求；不会修改永久电源设置，也不能阻止人为关机。只记录本地状态，不主动发送进度消息。

完整输出包含`experiment_plan.json`、`training/<组别>/best.pt`、开发/测试指标、冻结记录、配对区间、原始音频核对、推理耗时和`B组消融验证报告.md`。

## 中断与复现

已完成组可跳过，但原训练配置和源码必须不变；损坏或未完成目录不会被静默覆写。当前训练器沿用A组checkpoint格式，未保存优化器/RNG恢复点；若某组中途退出，需保留其中断记录、将失败目录隔离后从头重训，不能仅加载best.pt并声称精确续训。缓存不完整时也需先保留诊断再重新生成。

性能结论仍是单种子探索性对照，配对区间不代表重训波动。eval已被项目历史查看，不能称为全新盲测。B0沿用经哈希核验的A0完整方案，避免根据A组测试排名改换共同参照。
