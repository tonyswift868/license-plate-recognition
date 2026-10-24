# 派单 02 · 员工 B · 第三批：图 5 车牌识别六阶段流水线图

```text
你是“停车场收费系统”M2 车牌识别模块的执行者【员工 B，工位 02】。
工作目录：D:\keshe2\workers\02    分支：feature/m2-wpb-locate（已检出，不要切换分支）
环境：工作目录自带 .venv，用 .venv\Scripts\python.exe。
必读：docs/ref/停车场收费系统-AI开发说明.md 11.3 节（图表清单、图片引用格式，draw.io 导出 150dpi PNG）；
      docs/M2-车牌识别-开发文档.md 4.1、4.2 节；你自己的 algorithm/lpr_tools/viz.py
任务：
  1. 画图 5 流水线图的 draw.io 源文件 docs/img/src/pipeline.drawio（手写 draw.io XML 即可，能被 draw.io 正常打开）：
     六个阶段（预处理→定位→校正→分割→识别→后处理）及各阶段输入/输出、候选回退与多版式尝试分支、
     多帧投票与重复抓拍判定的旁路、最终 RecognitionResult；中文标注，配色清爽。
  2. 各阶段中间图素材：用 lpr_tools.viz 在 algorithm/data/raw_mock 的仿 CCPD 图上导出一套
     （原图、颜色掩膜、边缘掩膜、候选框、校正图、二值图、分割框、字符拼图），挑一张效果好的，
     缩小到宽 ≤800 px，存到 docs/img/stages/（总大小 ≤2 MB）。仿 CCPD 数据若不存在，先用
     python -m lpr_tools.synth_ccpd --out algorithm/data/raw_mock --per-subset 20 --seed 42 生成。
  3. docs/img/README-图5.md：说明图 5 的构成、素材来源（仿真数据，CCPD 到位后替换为真实图）、
     以及 PNG 导出方法（本机没有 draw.io 桌面版：用 app.diagrams.net 打开 .drawio → 导出 PNG 150dpi →
     保存为 docs/img/pipeline.png）。本批不要求生成 pipeline.png。
只允许新建或修改：docs/img/src/pipeline.drawio、docs/img/stages/ 下的图片、docs/img/README-图5.md
禁止：修改任何代码、配置、测试和其他文档；引入新依赖或 pip install；切换或合并分支；改动 D:\keshe2 主检出或其他 workers 目录；
      提交 algorithm/data/、algorithm/out/ 下的文件。
协作方式：积极派子代理并行（如一个画 draw.io，一个导出素材）；看图先缩小到宽 ≤800 px，不在对话里打开大图。
提交：docs(m2): 前缀。
汇报方式：完整汇报写到 algorithm/out/receipts/第三批.md；聊天里只回一行：回执 algorithm/out/receipts/第三批.md，commit <最后提交号>
```
