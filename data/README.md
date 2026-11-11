# 配套运行数据

向交接人取得 `reduced_data_and_wavlm.zip`，按 package_manifest.json 校验 SHA256 后解压到**项目根目录**。正确结果为 `data/reduced` 与 `data/models/wavlm_large.pt`。

ZIP 约 3.53 GB；包含 17,460 条固定音频、原生标签、清单和通用 WavLM 权重，不含公开 SAL 成品模型或旧训练检查点，不含可选 ASR/NER 模型。

解压后在 code 目录执行 `python data.py --action audit --deep`。所有角色清单与仓库 code/manifests/reduced 必须一致。不得重新随机抽样、按模型错误挑选数据、修改音频或把旧数据重新混入。

本仓库不上传大型音频和模型，下载代码时还须取得此配套 ZIP。ZIP 内所有数据保留原始研究使用范围和出处，不因此获得额外再分发许可。
