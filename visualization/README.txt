可视化入口（缩减数据项目，2026-11-11）

文件职责
- diagnose_saved_predictions.py：只读已保存的 SAL predictions.sqlite，生成诊断图、CSV和来源JSON；不训练、不重新推理、不调阈值。
- export_word_figures.py：将已有的PS/LPS诊断导出为三张便于放入Word的图；不生成新的Word文档。
- visualization_paths.py：统一路径、当前缩减实验身份校验与跨盘来源记录。
旧 build_review_docx.py 已移除；项目交接Word由项目根目录交付文件提供。

无参数运行两个入口只显示帮助，不启动绘图。使用 code/Setup.cmd 配好的 Python 3.10 环境即可。
以下命令从项目根目录运行；命令中的相对路径也以项目根目录为基准。

1. 仅检查诊断计算基本逻辑：
   code/.venv/Scripts/python.exe visualization/diagnose_saved_predictions.py --check

2. 自动选择当前缩减模型的PS/LPS结果：
   code/.venv/Scripts/python.exe visualization/diagnose_saved_predictions.py --latest
   自动选择必须同时满足：complete、reduced、非smoke、当前eval/LPS清单、当前train_detector/dev训练身份、reduced_trained权重；PS/LPS checkpoint_sha256必须一致，否则要求显式选择。
   没有符合条件的新结果时明确报错，不会自动选旧全量/旧fixed10/公开SAL权重结果。

3. 显式诊断一个或两个已完成结果；--run也是--runs的别名：
   code/.venv/Scripts/python.exe visualization/diagnose_saved_predictions.py --runs result/<PS运行目录> result/<LPS运行目录>
   可显式读取历史结果；图标题显示实际profile、录音数和保存网格，不能当作新缩减模型成绩。
   Windows外部盘绝对路径也支持；来源跨盘时保留绝对路径，不调用会失败的跨盘相对路径转换。

4. 重绘某次诊断（沿用它的已统计数据）：
   code/.venv/Scripts/python.exe visualization/diagnose_saved_predictions.py --render-existing visualization/outputs/<诊断目录>

5. 导出Word用图（需要该诊断同时包含PS_eval和LPS_crossfade）：
   code/.venv/Scripts/python.exe visualization/export_word_figures.py --analysis-dir visualization/outputs/<诊断目录>
   或使用 --latest，只选择当前缩减训练身份的最近完整诊断。默认不会生成Word。

当前输出：visualization/outputs/sal_diagnostics_<时间戳>/。
历史图片：visualization/figures/ 和 visualization/historical_evidence/，保留供原报告追溯，不代表新模型结果。
需要重绘时必须提供对应预测数据库或完整诊断CSV/JSON；仅图片不能还原逐帧预测。
输出仅写入visualization目录。读取预测不需要原音频、模型权重或GPU。
PS默认原生160ms、LPS默认原生20ms；每张图使用结果实际保存网格，不把粗标签当精细边界真值。
边界、完全漏检、碎片化等类别可能重叠；不能将不同分母的比例相加。历史结论不会自动套用到新结果。
