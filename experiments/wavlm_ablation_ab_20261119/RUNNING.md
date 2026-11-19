# 代码、归档复核与运行说明

本目录是独立快照。不要覆盖复制到仓库根目录，也不要把原来的训练缓存换成另一实验的特征。历史日志中的Windows绝对路径仅作来源证据。

## 无GPU复核与查看指标

在本目录运行，默认只核验归档，不训练、不联网：

```powershell
python release_tools.py verify
python release_tools.py summarize
```

核验检查文件SHA256、14个不同最佳权重、冻结开发集指标、协议与对应源码。B0复用A0权重，文件不重复存储。完整模型与阈值均由dev选择，测试分数未用于选择训练轮次。

## 环境

原运行环境：Python3.10.18、torch2.2.1+cu121、torchaudio2.2.1、transformers4.51.3、RTX4080 Laptop。依赖见`code/requirements-release.txt`；如使用GPU，需安装与本机兼容的PyTorch CUDA构建。B组内部干预仅对已验证的transformers4.51.3实现开放。

```powershell
python -m pip install -r code/requirements-release.txt
```

公开WavLM-Base-Plus权重没有重复打包。进入`code`目录后，可用原有下载入口准备固定版本编码器：

```powershell
python -m cnn_ps.download_wavlm --output YOUR_MODEL_FOLDER
```

固定revision为`4c66d4806a428f2e922ccfa1a962776e232d487b`，模型SHA256为`3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51`。

## 单音频推理

回到本归档目录，输入16kHz单声道音频。以下是使用说明，本次发布不会自动执行训练：

```powershell
python release_tools.py predict --group A --arm A5_fixed_pool --model-folder YOUR_MODEL_FOLDER --audio YOUR_AUDIO.wav --output prediction_a5.json --device cuda
python release_tools.py predict --group B --arm B2_position_gate_one --model-folder YOUR_MODEL_FOLDER --audio YOUR_AUDIO.wav --output prediction_b2.json --device cuda
```

支持全部A/B组，可用`--device cpu`。阈值直接读取各组best.pt，不用音频标签或测试集重新校准。输出原生20ms logits及连续正帧区间，无额外平滑或短段删除。输出文件已存在时拒绝覆写。

## 重新训练的入口

完整缓存和音频不在发布包中。必须另行准备同一固定划分、细粒度真实20ms标签，以及对应干预的特征缓存。核心模块使用显式路径，可从归档`code`目录运行：

```powershell
# A组：输入为原版冻结WavLM的6/9/12层缓存
python -m cnn_ps.train_wavlm_ablation_a --arm A5_fixed_pool --train-cache REFERENCE_TRAIN --dev-cache REFERENCE_DEV --train-ssl ORIGINAL_WAVLM_TRAIN --dev-ssl ORIGINAL_WAVLM_DEV --output NEW_A_OUTPUT --epochs 20 --batch-size 8 --seed 666 --threads 4

# B组：每个组需要自己重新生成的缓存；--execute才会执行
python -m cnn_ps.cache_wavlm_ablation_b --execute --arm B1_no_conv_position --split train --reference REFERENCE_TRAIN --model-folder YOUR_MODEL_FOLDER --output NEW_B1_TRAIN_CACHE
python -m cnn_ps.cache_wavlm_ablation_b --execute --arm B1_no_conv_position --split dev --reference REFERENCE_DEV --model-folder YOUR_MODEL_FOLDER --output NEW_B1_DEV_CACHE
python -m cnn_ps.train_wavlm_ablation_b --execute --arm B1_no_conv_position --train-cache REFERENCE_TRAIN --dev-cache REFERENCE_DEV --train-ssl NEW_B1_TRAIN_CACHE --dev-ssl NEW_B1_DEV_CACHE --output NEW_B_OUTPUT --epochs 20 --batch-size 8 --seed 666 --threads 4
```

所有训练结果必须写入新的目录。B组冻结/评价入口要求B1/B2/B3共同完成，不能提前用eval选择设置；每组的eval缓存应置于`EVAL_SSL_ROOT/<组名>/eval`，并由已有冻结计划授权生成。

原始`run_*`、`launch_*`、`verify_*`和`benchmark_*`等本机调度脚本作为实验来源保留，其中部分路径依赖原工作区布局，不能当作换机器后的一键复现实验。换机器优先使用上述显式路径核心入口；数据路径变化会影响元数据哈希，不能伪造旧来源或直接复用旧冻结计划。完整复现应重新建立自身缓存/协议，并为共同基线提供同条件验证或重训。

## 复现限制与产物

主结果位于`reports/A`和`reports/B`。训练源码快照字节保持原样，依赖字段和历史绝对路径没有为美观而重写。新发布工具仅提供便携验证/推理，不改变已完成训练的计算路径。

A组曾在第20轮中途被终止，最终表格使用从头重跑后完成20轮的结果；中断模型不用于比较。各组最佳检查点没有优化器/RNG状态，不支持把best.pt当作精确断点续训。单音频推理跨设备/库版本可能有微小数值差异，以归档分数定义本次结果。

14组最佳权重约64MB，全部小于GitHub单文件限制，不需要Git LFS。发布包不含语音、完整特征缓存、全部epoch权重、虚拟环境或训练进程。
