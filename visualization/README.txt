【2026-11-07整理】绘图代码和图表均在visualization；默认输出为visualization/outputs。
result仅提供模型成绩和预测数据作为绘图输入。旧诊断图目录已迁入outputs。

SAL 可视化与导师审阅文档

一、文件用途
diagnose_saved_predictions.py：读取已完成 SAL 实验的 SQLite 分数和标签，生成两套诊断图、单例图及统计 CSV。
export_word_figures.py：从已有诊断结果导出三张适合 Word 排版的图，使用指定蓝紫色卡。
build_review_docx.py：将正文和三张图生成 Word 文档。
visualization_paths.py：集中处理相对路径，以及旧结果元数据中记录的绝对路径。

二、目录关系
项目根目录/
  code/
    config.json
  data/
  result/
    sal_wavlm_.../predictions.sqlite
    sal_wavlm_.../metrics.json
    sal_wavlm_.../status.json
  visualization/
    本文件和全部独立可视化脚本
    outputs/sal_diagnostics_.../  图表和诊断导出
    figures/  既有论文图
    vendor_sal/  迁入的原作者独立绘图脚本

脚本按自身所在位置查找相邻的 code 和 result，不依赖 D 盘、用户名或终端当前目录。
图中文字使用 Times New Roman；目标电脑需安装该字体才能得到相同外观，数据计算不依赖字体。
绘图区与画布底色均为白色 #FFFFFF；热力图颜色仍按原数值映射。
坐标轴保留四边框，柱形和热力图色块使用 #4A4A4A 轮廓；样式色号集中在 diagnose_saved_predictions.py 的 COLORS 中。
命令参数中的相对路径统一相对于项目根目录，而非当前终端目录。
旧 summary.json 中的原机器绝对路径会优先映射到本包 result 下的同名实验文件夹。
code/config.json 的 result_root 应保持相对于 code 的路径，例如 ../result。

三、首次运行
推荐使用 Python 3.11 或 3.12。在项目根目录打开终端，创建本机环境；不要将原电脑的 .venv 当作可移植环境。

Windows：
  python -m venv visualization/.venv
  visualization/.venv/Scripts/python.exe -m pip install -r visualization/requirements.txt
  visualization/.venv/Scripts/python.exe visualization/diagnose_saved_predictions.py

macOS / Linux：
  python3 -m venv visualization/.venv
  visualization/.venv/bin/python -m pip install -r visualization/requirements.txt
  visualization/.venv/bin/python visualization/diagnose_saved_predictions.py

如果原项目 code/.venv 已安装绘图依赖，可继续使用 code/.venv/Scripts/python.exe 运行绘图脚本。
下文 python 均指已安装上述依赖的解释器。

四、常用命令（从项目根目录运行）
完整诊断，默认选择最新完成的 PS eval 和 LPS 实验，并创建新的时间戳结果目录：
  python visualization/diagnose_saved_predictions.py

只重绘已有图：
  python visualization/diagnose_saved_predictions.py --render-existing visualization/outputs/sal_diagnostics_figures_1_2_20261106

导出适合 Word 的三张图（默认选择最新完整诊断目录）：
  python visualization/export_word_figures.py

指定诊断目录：
  python visualization/export_word_figures.py --analysis-dir visualization/outputs/sal_diagnostics_figures_1_2_20261106

生成 Word（先运行上一条导出三张图）：
  python visualization/build_review_docx.py

Word 生成器会重新写入默认审阅文档；已手工修改的版本请另存，或用 --output 指定新文件：
  python visualization/build_review_docx.py --output result/新的导师审阅文档.docx

五、打包时必须带上什么
1. visualization 文件夹，以及 code/config.json（也可保留完整 code）。
2. result 中两次 SAL 实验的 predictions.sqlite、metrics.json、status.json。
3. 若要直接重绘或生成 Word，同时带上完整 sal_diagnostics_... 文件夹（包括 CSV、JSON、导师审阅/figures）。
4. 不需要音频、模型权重或 GPU 即可使用已保存的结果画图。
5. 不要只打包 Python 文件而遗漏结果数据库；脚本需要真实预测数据。
6. 官方 vendor/SAL 内的原始 Grad-CAM 等工具仍属于第三方模型源码，不是本次新增的离线诊断脚本。

统计口径与限制见结果目录的 阅读说明.txt。历史 manifest 中的旧路径用于记录当时来源，不要求目标电脑具备该盘符。
